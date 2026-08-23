# One-vs-rest linear SVM over SoA feature columns: per epoch, compute each
# class's hinge-loss factor, reduce it to a gradient per feature, update the
# weights on the host, then evaluate accuracy with a single argmax reduction.
#
# Mirrors the C++ benchmark. Reductions remain independent until read, allowing
# the runtime to fuse them.

using PolymerPIM
using Printf
include(joinpath(@__DIR__, "../bench.jl"))
using .Bench

const _gen = joinpath(@__DIR__, "Param.generated.jl")
include(isfile(_gen) ? _gen : joinpath(@__DIR__, "Param.jl"))

const LABEL = "julia"

# Ports of multitask_classifier_common.h.
const SVM_MARGIN = 12
const SVM_WEIGHT_DECAY = 8

flow_class_for_row(row, classes) = Int32((row * 5 + 1) & (classes - 1))

function flow_feature_value(row, feature, classes)
    mixed = UInt32(row % (1 << 32)) * 0x41c64e6d + 0x3039 +
            UInt32(feature) * 0x9e3779b1
    mixed ⊻= mixed >> 16
    center = flow_class_for_row(row, classes) == feature ÷ 2 ? Int32(2) : Int32(-1)
    noise = Int32(mixed % 3) - Int32(1)
    return center + noise
end

function svm_div_round_closest(value::Int64, divisor::Int64)
    value < 0 && return -((-value + divisor ÷ 2) ÷ divisor)
    return (value + divisor ÷ 2) ÷ divisor
end

function svm_update_weight(weight::Int32, gradient::Int64, rows::Integer)
    divisor = max(Int64(rows ÷ 2), Int64(1))
    data_step = svm_div_round_closest(gradient, divisor)
    decay_step = svm_div_round_closest(Int64(weight), Int64(SVM_WEIGHT_DECAY))
    return Int32(Int64(weight) - data_step - decay_step)
end

function main()
    T = Param.T
    N = Param.N
    FEATURES = Param.FEATURES
    CLASSES = Param.CLASSES

    stages = BenchStages()
    warm_stages = BenchStages()

    @printf("[VERIFY_TAG] Starting julia_multitask_classifier with N=%d F=%d C=%d\n",
            N, FEATURES, CLASSES)

    stage_begin!(stages, :init)
    PolymerPIM.sync()
    stage_end!(stages)

    stage_begin!(stages, :alloc)
    host_features = [Vector{T}(undef, N) for _ in 1:FEATURES]
    host_classes = Vector{T}(undef, N)
    stage_end!(stages)

    stage_begin!(stages, :load)
    for i in 0:(N - 1)
        host_classes[i + 1] = flow_class_for_row(i, CLASSES)
        for d in 0:(FEATURES - 1)
            host_features[d + 1][i + 1] = flow_feature_value(i, d, CLASSES)
        end
    end
    stage_end!(stages)

    stage_begin!(stages, :write)
    features = [DPUVector(host_features[d]) for d in 1:FEATURES]
    class_ids = DPUVector(host_classes)
    PolymerPIM.sync()
    stage_end!(stages)

    weights = zeros(T, CLASSES * FEATURES)
    metrics = Dict(:margin_violations => Int64(0), :correct_predictions => Int64(0))

    function run_epoch(st)
        stage_begin!(st, :kernel)
        violation_futures = DpuFuture[]
        gradient_futures = DpuFuture[]
        factors = DPUVector[]

        for c in 0:(CLASSES - 1)
            model = Int32[weights[c * FEATURES + d] for d in 1:FEATURES]
            score = features[1] .* model[1]
            for d in 2:FEATURES
                score = score .+ features[d] .* model[d]
            end
            label = (class_ids .== T(c)) .* T(2) .- T(1)
            active = label .* score .< T(SVM_MARGIN)
            factor = DPUVector(active .* (-label))

            # ||factor||^2 counts the margin violations for this class.
            push!(violation_futures, sum(abs2.(factor)))
            # and factor . feature_d is the gradient for that weight.
            for d in 1:FEATURES
                push!(gradient_futures, sum(factor .* features[d]))
            end
            push!(factors, factor)
        end
        PolymerPIM.sync()
        stage_end!(st)

        stage_begin!(st, :read)
        metrics[:margin_violations] = sum(Int64(get(f)) for f in violation_futures)
        gradients = Int64[Int64(get(f)) for f in gradient_futures]
        stage_end!(st)
        # C++ drops each per-class factor at scope exit; without the same here
        # their MRAM lives until Julia's GC runs, which it cannot schedule off
        # DPU memory pressure -- over an epoch count the DPUs fill up and OOM.
        for f in factors
            release!(f)
        end

        stage_begin!(st, :merge)
        for i in 1:(CLASSES * FEATURES)
            weights[i] = svm_update_weight(weights[i], gradients[i], N)
        end
        stage_end!(st)

        stage_begin!(st, :kernel)
        scores = DpuLazy[]
        for c in 0:(CLASSES - 1)
            s = features[1] .* weights[c * FEATURES + 1]
            for d in 2:FEATURES
                s = s .+ features[d] .* weights[c * FEATURES + d]
            end
            push!(scores, s)
        end
        best = argmax.(zip(scores...)) .- 1
        correct_future = sum(best .== class_ids)
        PolymerPIM.sync()
        stage_end!(st)

        stage_begin!(st, :read)
        metrics[:correct_predictions] = Int64(get(correct_future))
        stage_end!(st)
    end

    # release! covers the vectors this file names; the fused kernels' anonymous
    # intermediates are only reclaimed by the GC, which cannot see MRAM pressure.
    # Collect per pass (outside the timed region) or a long run OOMs.
    warm = BenchStats()
    for _ in 1:Param.warmup_iterations
        t0 = time_ns()
        run_epoch(warm_stages)
        stats_update!(warm, elapsed_us(t0))
        GC.gc()
    end
    Param.warmup_iterations > 0 && stats_print("$(LABEL)_warmup", warm)

    stats = BenchStats()
    for _ in 1:Param.iterations
        t0 = time_ns()
        run_epoch(stages)
        stats_update!(stats, elapsed_us(t0))
        GC.gc()
    end
    stats_print(LABEL, stats)
    stages_report(LABEL, stages)
    stages_report("$(LABEL)_cold", warm_stages)

    @printf("margin_violations: %d\n", metrics[:margin_violations])
    @printf("correct_predictions: %d\n", metrics[:correct_predictions])

    if Param.check_correctness != 0
        # Recompute the final epoch's evaluation on the host from the weights
        # the DPU run ended with.
        correct = 0
        for i in 1:N
            best_c, best_s = 0, typemin(Int64)
            for c in 0:(CLASSES - 1)
                s = Int64(0)
                for d in 1:FEATURES
                    s += Int64(host_features[d][i]) *
                         Int64(weights[c * FEATURES + d])
                end
                if s > best_s
                    best_s, best_c = s, c
                end
            end
            correct += (best_c == host_classes[i]) ? 1 : 0
        end
        if correct != metrics[:correct_predictions]
            @printf("Mismatch: CPU correct = %d, DPU correct = %d\n",
                    correct, metrics[:correct_predictions])
            exit(1)
        end
        println("the result is correct")
    end
end

main()
