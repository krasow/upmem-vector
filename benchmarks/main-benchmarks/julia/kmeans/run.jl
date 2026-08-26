# K-means: one Lloyd iteration over SoA columns.
#
# Mirrors benchmarks/main-benchmarks/polymerpim/kmeans/run.cc. Assignment and accumulation share a
# single scan: the RPN program picks the nearest centroid with one argmin over K
# candidate distances, then scatters a count and DIM coordinate sums into a
# per-DPU accumulator.  All DIM+1 scatters share the index prefix (the winning
# label times DIM+1), which the program emits once and re-uses with OP_DUP.

using PolymerPIM
using Printf
include(joinpath(@__DIR__, "../bench.jl"))
using .Bench

const _gen = joinpath(@__DIR__, "Param.generated.jl")
include(isfile(_gen) ? _gen : joinpath(@__DIR__, "Param.jl"))

const LABEL = "julia"

# Bound as constants, not locals in main(): a local holding Param.T is only a
# DataType to inference, which makes every host array abstractly typed and
# turns each element store into a dynamic dispatch.
const T = Param.T
const N = Param.N
const DIM = Param.DIM
const K = Param.K

div_round_closest(n::Integer, d::Integer) =
    ((n < 0) != (d < 0)) ? ((n - d ÷ 2) ÷ d) : ((n + d ÷ 2) ÷ d)

function main()

    stages = BenchStages()
    warm_stages = BenchStages()

    @printf("[VERIFY_TAG] Starting julia_kmeans with N=%d DIM=%d K=%d\n", N, DIM, K)

    stage_begin!(stages, :init)
    PolymerPIM.sync()
    stage_end!(stages)

    stage_begin!(stages, :load)
    centroids_init = Vector{T}(undef, K * DIM)
    if Param.load_ref != 0
        load_bin!(joinpath(Param.ref_path, "ref_c_init.bin"), centroids_init)
    else
        for j in 0:(K - 1), d in 0:(DIM - 1)
            centroids_init[j * DIM + d + 1] = T((j + d) % 1000)
        end
    end
    stage_end!(stages)

    cols = Vector{DPUVector}()
    host_cols = Vector{Vector{T}}()
    for d in 0:(DIM - 1)
        stage_begin!(stages, :alloc)
        col = Vector{T}(undef, N)
        stage_end!(stages)
        stage_begin!(stages, :load)
        if Param.load_ref != 0
            load_bin!(joinpath(Param.ref_path, "SoA", "col_$(d).bin"), col)
        else
            @inbounds for i in 0:(N - 1)
                col[i + 1] = T((i + d) % 1000)
            end
        end
        stage_end!(stages)
        stage_begin!(stages, :write)
        push!(cols, DPUVector(col))
        fence(cols[end])
        stage_end!(stages)
        Param.check_correctness != 0 && push!(host_cols, col)
    end

    centroids = copy(centroids_init)
    operands = cols[2:end]

    function run_kmeans(st)
        stage_begin!(st, :kernel)
        local_stats = DPULocalVector(K * (DIM + 1))

        # Squared distance to each centroid. Lazy scalar operands are launch
        # parameters automatically, so every iteration reuses one compiled
        # kernel while capturing this iteration's centroid values.
        dists = Vector{Any}(undef, K)
        for j in 0:(K - 1)
            acc = abs2.(cols[1] .- centroids[j * DIM + 1])
            for d in 2:DIM
                acc = acc .+ abs2.(cols[d] .- centroids[j * DIM + d])
            end
            dists[j + 1] = acc
        end

        # The nearest centroid, times the stride of one centroid's slots.  Every
        # update below shares this index, and the scatter emits it once (OP_DUP).
        base = (argmin.(zip(dists...)) .- 1) .* T(DIM + 1)
        local_stats[base] .+= 1
        for d in 1:DIM
            local_stats[base .+ T(d)] .+= cols[d]
        end
        PolymerPIM.sync()
        stage_end!(st)

        stage_begin!(st, :read)
        flat = Array(local_stats)
        counts = [Int64(flat[j * (DIM + 1) + 1]) for j in 0:(K - 1)]
        sums = [Int64(flat[j * (DIM + 1) + d + 1]) for j in 0:(K - 1)
                                                  for d in 1:DIM]
        stage_end!(st)

        stage_begin!(st, :merge)
        for j in 0:(K - 1)
            counts[j + 1] <= 0 && continue
            for d in 1:DIM
                s = sums[j * DIM + d]
                centroids[j * DIM + d] = T(div_round_closest(s, counts[j + 1]))
            end
        end
        stage_end!(st)
        return counts
    end

    warm = BenchStats()
    for _ in 1:Param.warmup_iterations
        centroids .= centroids_init
        t0 = time_ns()
        run_kmeans(warm_stages)
        stats_update!(warm, elapsed_us(t0))
    end
    Param.warmup_iterations > 0 && stats_print("$(LABEL)_warmup", warm)

    stats = BenchStats()
    counts = Int64[]
    for _ in 1:Param.iterations
        centroids .= centroids_init
        t0 = time_ns()
        counts = run_kmeans(stages)
        stats_update!(stats, elapsed_us(t0))
    end
    stats_print(LABEL, stats)
    stages_report(LABEL, stages)
    stages_report("$(LABEL)_cold", warm_stages)

    if Param.check_correctness != 0
        # Repeat the assignment step on the host from the initial centroids.
        want_counts = zeros(Int64, K)
        want_sums = zeros(Int64, K * DIM)
        for i in 1:N
            best_j, best_dist = 0, typemax(Int64)
            for j in 0:(K - 1)
                dist = Int64(0)
                for d in 1:DIM
                    diff = Int64(host_cols[d][i]) -
                           Int64(centroids_init[j * DIM + d])
                    dist += diff * diff
                end
                if dist < best_dist
                    best_dist, best_j = dist, j
                end
            end
            want_counts[best_j + 1] += 1
            for d in 1:DIM
                want_sums[best_j * DIM + d] += Int64(host_cols[d][i])
            end
        end
        want = copy(centroids_init)
        for j in 0:(K - 1)
            want_counts[j + 1] <= 0 && continue
            for d in 1:DIM
                # The DPU accumulates in a 32-bit local vector, so a large
                # cluster wraps; generate_ref.cc truncates the same way via
                # `(int)sums[...]`.  Reproduce it rather than flag it.
                s = Int64(want_sums[j * DIM + d] % Int32)
                want[j * DIM + d] =
                    T(div_round_closest(s, want_counts[j + 1]))
            end
        end
        if want_counts != counts
            @printf("Mismatch in counts: CPU = %s, DPU = %s\n",
                    string(want_counts), string(counts))
            exit(1)
        end
        bad = findfirst(i -> want[i] != centroids[i], 1:(K * DIM))
        if bad !== nothing
            @printf("Mismatch at centroid %d: CPU = %d, DPU = %d\n",
                    bad - 1, want[bad], centroids[bad])
            exit(1)
        end
        println("the result is correct")
    end
end

main()
