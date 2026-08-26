# Elementwise: apply Param.operation to two vectors.
#
# Mirrors benchmarks/main-benchmarks/polymerpim/elementwise/run.cc, including the per-stage fences --
# the runtime is lazy, so without them every cost lands in the blocking read.

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

# Bound as constants, not locals in main(): a local holding Param.T is only a
# DataType to inference, which makes every host array abstractly typed and
# turns each element store into a dynamic dispatch.
const T = Param.T
const N = Param.N

# `result` is assigned inside round_trip, so Julia boxes it and indexing it from
# a closure dispatches per element -- 42x slower over N.  Compare in a typed
# function instead.
function first_mismatch(got::Vector{T}, want::Vector{T}) where {T}
    @inbounds for i in eachindex(got, want)
        got[i] != want[i] && return i
    end
    return nothing
end

function main()

    stages = BenchStages()
    warm_stages = BenchStages()

    @printf("[VERIFY_TAG] Starting julia_elementwise with N=%d\n", N)

    stage_begin!(stages, :init)
    PolymerPIM.sync()
    stage_end!(stages)

    stage_begin!(stages, :alloc)
    a = Vector{T}(undef, N)
    b = Vector{T}(undef, N)
    stage_end!(stages)

    expected = T[]
    stage_begin!(stages, :load)
    if Param.load_ref != 0
        @printf("Loading reference data from %s...\n", Param.ref_path)
        load_bin!(joinpath(Param.ref_path, "ref_a.bin"), a)
        load_bin!(joinpath(Param.ref_path, "ref_b.bin"), b)
        if Param.check_correctness != 0
            expected = Vector{T}(undef, N)
            load_bin!(joinpath(Param.ref_path, "ref_res.bin"), expected)
        end
    else
        @inbounds for i in 1:N
            a[i] = T((i - 1) % 10)
            b[i] = T((i * 2) % 10)
        end
    end
    stage_end!(stages)

    result = T[]

    function round_trip(st)
        stage_begin!(st, :write)
        da = DPUVector(a)
        db = DPUVector(b)
        PolymerPIM.sync()
        stage_end!(st)
        stage_begin!(st, :kernel)
        # Lazy: sync() runs it, since nothing else will, so the kernel is
        # timed here rather than in the read stage below.
        res = Param.operation(da, db)
        PolymerPIM.sync()
        stage_end!(st)
        stage_begin!(st, :read)
        result = Array(res)
        stage_end!(st)
        # C++ destroys each dpu_vector at scope exit; without the same here the
        # DPU memory lives until Julia's GC runs, which it cannot schedule off
        # MRAM pressure -- the next pass then OOMs and pays a major collection.
        release!(da); release!(db); release!(DPUVector(res))
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
        # Without a reference file the host applies the same operation
        # elementwise -- Param.operation is written to work on scalars too.
        want = isempty(expected) ?
            T[Param.operation(a[i], b[i]) for i in 1:N] : expected
        bad = first_mismatch(result::Vector{T}, want::Vector{T})
        if bad !== nothing
            @printf("Mismatch at index %d: got %d, expected %d\n",
                    bad - 1, result[bad], want[bad])
            exit(1)
        end
        @printf("All results match after %d iterations.\n", Param.iterations)
    end
end

main()
