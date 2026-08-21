# Reduction: sum a single vector.
#
# Mirrors benchmarks/variants/polymerpim/red/run.cc, including the per-stage fences -- the
# runtime is lazy, so without them every cost lands in the blocking read.

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

    stages = BenchStages()       # steady loop (+ one-time setup)
    warm_stages = BenchStages()  # cold warmup loop

    @printf("[VERIFY_TAG] Starting julia_red with N=%d\n", N)

    stage_begin!(stages, :init)
    PolymerPIM.sync()           # forces runtime init
    stage_end!(stages)

    stage_begin!(stages, :alloc)
    a = Vector{T}(undef, N)
    stage_end!(stages)

    stage_begin!(stages, :load)
    if Param.load_ref != 0
        @printf("Loading reference data from %s...\n", Param.ref_path)
        load_bin!(joinpath(Param.ref_path, "ref_t1.bin"), a)
    else
        for i in 1:N
            a[i] = T((i - 1) % 10)
        end
    end
    stage_end!(stages)

    result = Ref{Int64}(0)

    function round_trip(st)
        stage_begin!(st, :write)
        da = DPUVector(a)
        PolymerPIM.sync()
        stage_end!(st)
        stage_begin!(st, :kernel)
        pending = sum(da)
        PolymerPIM.sync()
        stage_end!(st)
        stage_begin!(st, :read)
        result[] = pending[]
        stage_end!(st)
        # C++ drops da at scope exit; without the same here its MRAM lives until
        # Julia's GC runs, which it cannot schedule off DPU memory pressure -- a
        # long run then trips retry_on_oom and pays a major collect mid-loop.
        release!(da)
    end

    warm = BenchStats()
    for _ in 1:Param.warmup_iterations
        t0 = time_ns()
        round_trip(warm_stages)
        stats_update!(warm, elapsed_us(t0))
    end
    Param.warmup_iterations > 0 && stats_print("$(LABEL)_warmup", warm)

    stats = BenchStats()
    for _ in 1:Param.iterations
        t0 = time_ns()
        round_trip(stages)
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
            sum(Int64.(a))
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
