# Linear regression: one gradient-descent step per iteration over SoA columns.
#
# Mirrors benchmarks/main-benchmarks/polymerpim/linreg/run.cc. Weights stay at zero (the reference
# gradients are taken from the first step), and the DIM gradient reductions are
# left unread so they fuse into as few kernel passes as possible.

using PolymerPIM
using Printf
include(joinpath(@__DIR__, "../bench.jl"))
using .Bench

# Param.jl holds the defaults between the harness's marker comments; the sweep
# writes substituted values to Param.generated.jl.  The choice lives here rather
# than in Param.jl because `module` cannot be parsed inside an if/else.
const _gen = joinpath(@__DIR__, "Param.generated.jl")
include(isfile(_gen) ? _gen : joinpath(@__DIR__, "Param.jl"))

const LABEL = "julia"

function main()
    T = Param.T
    N = Param.N
    DIM = Param.DIM
    scaling_shift = Int32(Param.scaling_shift)
    s_half = scaling_shift ÷ 2

    stages = BenchStages()
    warm_stages = BenchStages()

    @printf("[VERIFY_TAG] Starting julia_linreg with N=%d, DIM=%d\n", N, DIM)

    stage_begin!(stages, :init)
    PolymerPIM.sync()
    stage_end!(stages)

    stage_begin!(stages, :alloc)
    host_x = [Vector{T}(undef, N) for _ in 1:DIM]
    host_y = Vector{T}(undef, N)
    stage_end!(stages)

    expected_grads = Int64[]
    stage_begin!(stages, :load)
    if Param.load_ref != 0
        @printf("Loading reference data from %s...\n", Param.ref_path)
        for j in 1:DIM
            load_bin!(joinpath(Param.ref_path, "SoA", "x_col_$(j-1).bin"), host_x[j])
        end
        load_bin!(joinpath(Param.ref_path, "SoA", "y.bin"), host_y)
        if Param.check_correctness != 0
            expected_grads = Vector{Int64}(undef, DIM)
            load_bin!(joinpath(Param.ref_path, "ref_grads.bin"), expected_grads)
        end
    else
        # Same synthetic pattern as the C++ variant, so the two agree.
        for i in 0:(N - 1)
            for j in 0:(DIM - 1)
                host_x[j + 1][i + 1] = T((i * (DIM + 1) + j) % 256)
            end
            host_y[i + 1] = T((i * (DIM + 1) + DIM) % 256)
        end
    end
    stage_end!(stages)

    stage_begin!(stages, :write)
    dy = DPUVector(host_y)
    dx = [DPUVector(host_x[j]) for j in 1:DIM]
    PolymerPIM.sync()
    stage_end!(stages)

    weights = zeros(T, DIM)
    grads = zeros(Int64, DIM)

    function run_iter(st)
        stage_begin!(st, :kernel)
        # error = -y + sum_j x_j * w_j, pre-shifted so the 32-bit product below
        # cannot overflow.  Dotted operators stay lazy, so the loop is one kernel.
        bc = .-dy
        for j in 1:DIM
            bc = bc .+ dx[j] .* weights[j]
        end
        # Materialised, not left lazy: DIM reductions share it, and inlining
        # would re-derive the whole error expression inside each of them.
        err_shifted = DPUVector(bc .>> (scaling_shift - s_half))
        # Unread, so the DIM reductions fuse into one pass.
        pending = [sum((dx[j] .>> s_half) .* err_shifted) for j in 1:DIM]
        PolymerPIM.sync()
        stage_end!(st)

        stage_begin!(st, :read)
        for j in 1:DIM
            grads[j] = pending[j][]
        end
        stage_end!(st)
        # C++ drops err_shifted at scope exit; without the same here its MRAM
        # lives until Julia's GC runs, which it cannot schedule off DPU memory
        # pressure -- over a long iteration count the DPUs fill up and OOM.
        release!(err_shifted)
    end

    # release! covers the vectors this file names; the fused kernels' anonymous
    # intermediates are only reclaimed by the GC, which cannot see MRAM pressure.
    # Collect per pass (outside the timed region) or a long run OOMs.
    warm = BenchStats()
    for _ in 1:Param.warmup_iterations
        t0 = time_ns()
        run_iter(warm_stages)
        stats_update!(warm, elapsed_us(t0))
        GC.gc()
    end
    Param.warmup_iterations > 0 && stats_print("$(LABEL)_warmup", warm)

    stats = BenchStats()
    for _ in 1:Param.iterations
        t0 = time_ns()
        run_iter(stages)
        stats_update!(stats, elapsed_us(t0))
        GC.gc()
    end
    stats_print(LABEL, stats)
    stages_report(LABEL, stages)
    stages_report("$(LABEL)_cold", warm_stages)

    @printf("Final gradients: %s\n", join(grads, " "))

    if Param.check_correctness != 0
        want = if !isempty(expected_grads)
            expected_grads
        else
            [sum(Int64(host_x[j][i] >> s_half) *
                 Int64((-host_y[i]) >> (scaling_shift - s_half))
                 for i in 1:N) for j in 1:DIM]
        end
        bad = findfirst(j -> abs(grads[j] - want[j]) > 1, 1:DIM)
        if bad !== nothing
            @printf("Mismatch at gradient %d: got %d, expected %d\n",
                    bad - 1, grads[bad], want[bad])
            exit(1)
        end
        @printf("All results match after %d iterations.\n", Param.iterations)
    end
end

main()
