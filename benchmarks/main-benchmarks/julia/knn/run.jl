# KNN: minimum over rows of the squared distance to a query point.
#
# Mirrors benchmarks/main-benchmarks/polymerpim/knn/run.cc. Columns are stored SoA, one DPUVector
# per dimension, and the whole distance-and-min becomes a single RPN reduction.

using PolymerPIM
using Printf
include(joinpath(@__DIR__, "../bench.jl"))
using .Bench

const _gen = joinpath(@__DIR__, "Param.generated.jl")
include(isfile(_gen) ? _gen : joinpath(@__DIR__, "Param.jl"))

const LABEL = "julia"

function main()
    T = Param.T
    N = Param.N
    DIM = Param.DIM

    stages = BenchStages()
    warm_stages = BenchStages()

    @printf("[VERIFY_TAG] Starting julia_knn with N=%d DIM=%d\n", N, DIM)

    stage_begin!(stages, :init)
    PolymerPIM.sync()
    stage_end!(stages)

    query = Vector{T}(undef, DIM)
    stage_begin!(stages, :load)
    if Param.load_ref != 0
        load_bin!(joinpath(Param.ref_path, "ref_query.bin"), query)
    else
        for d in 0:(DIM - 1)
            query[d + 1] = T((d * 17) % 128)
        end
    end
    stage_end!(stages)

    # One column at a time, so the host never holds DIM*N at once.
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
            for i in 0:(N - 1)
                col[i + 1] = T((i * (DIM + 1) + d) % 256)
            end
        end
        stage_end!(stages)
        stage_begin!(stages, :write)
        push!(cols, DPUVector(col))
        fence(cols[end])
        stage_end!(stages)
        Param.check_correctness != 0 && push!(host_cols, col)
    end

    result = Ref{Int64}(0)

    function run_knn(st)
        stage_begin!(st, :kernel)
        # sum_d (col_d - query_d)^2, then min -- one fused reduction kernel.
        # abs2 lowers to sqr, so each difference is loaded once (OP_DUP).
        acc = abs2.(cols[1] .- query[1])
        for d in 2:DIM
            acc = acc .+ abs2.(cols[d] .- query[d])
        end
        pending = minimum(acc)
        PolymerPIM.sync()
        stage_end!(st)
        stage_begin!(st, :read)
        result[] = pending[]
        stage_end!(st)
    end

    warm = BenchStats()
    for _ in 1:Param.warmup_iterations
        t0 = time_ns()
        run_knn(warm_stages)
        stats_update!(warm, elapsed_us(t0))
    end
    Param.warmup_iterations > 0 && stats_print("$(LABEL)_warmup", warm)

    stats = BenchStats()
    for _ in 1:Param.iterations
        t0 = time_ns()
        run_knn(stages)
        stats_update!(stats, elapsed_us(t0))
    end
    stats_print(LABEL, stats)
    stages_report(LABEL, stages)
    stages_report("$(LABEL)_cold", warm_stages)

    if Param.check_correctness != 0
        expected = if Param.load_ref != 0
            buf = Vector{Int64}(undef, 1)
            load_bin!(joinpath(Param.ref_path, "ref_res.bin"), buf)
            buf[1]
        else
            best = typemax(Int64)
            for i in 1:N
                acc = Int64(0)
                for d in 1:DIM
                    diff = Int64(host_cols[d][i]) - Int64(query[d])
                    acc += diff * diff
                end
                best = min(best, acc)
            end
            best
        end
        if expected != result[]
            @printf("Mismatch: CPU result = %d, DPU result = %d\n",
                    expected, result[])
            exit(1)
        end
        println("the result is correct")
    end
end

main()
