# Histogram: bucket each input into one of BINS and count.
#
# Mirrors benchmarks/main-benchmarks/polymerpim/hist/run.cc. The bucket index is computed in the
# same RPN program that scatters into the per-DPU accumulator, so the whole
# histogram is one kernel pass and the bucket vector is never materialised.

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
const BINS = Param.BINS
const DEPTH = Param.DEPTH

function main()

    stages = BenchStages()
    warm_stages = BenchStages()

    @printf("[VERIFY_TAG] Starting julia_hist with N=%d BINS=%d\n", N, BINS)

    stage_begin!(stages, :init)
    PolymerPIM.sync()
    stage_end!(stages)

    stage_begin!(stages, :alloc)
    a = Vector{T}(undef, N)
    stage_end!(stages)

    stage_begin!(stages, :load)
    if Param.load_ref != 0
        @printf("Loading reference data from %s...\n", Param.ref_path)
        load_bin!(joinpath(Param.ref_path, "ref_t1.bin"), a)
    else
        @inbounds for i in 0:(N - 1)
            a[i + 1] = T(i % (1 << DEPTH))
        end
    end
    stage_end!(stages)

    result = Int32[]

    function round_trip(st)
        stage_begin!(st, :write)
        da = DPUVector(a)
        PolymerPIM.sync()
        stage_end!(st)

        stage_begin!(st, :kernel)
        bins = DPULocalVector(BINS)
        # The bucket stays lazy, so it is computed inside the scatter kernel;
        # sync() flushes the queued update, keeping the work in this stage.
        bins[(da .* T(BINS)) .>> T(DEPTH)] .+= 1
        PolymerPIM.sync()
        stage_end!(st)

        stage_begin!(st, :read)
        result = Array(bins)
        stage_end!(st)
        # Julia GC problems
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
        want = if Param.load_ref != 0
            buf = Vector{Int32}(undef, BINS)
            load_bin!(joinpath(Param.ref_path, "ref_res.bin"), buf)
            buf
        else
            h = zeros(Int32, BINS)
            for v in a
                h[((Int64(v) * BINS) >> DEPTH) + 1] += 1
            end
            h
        end
        bad = findfirst(i -> result[i] != want[i], 1:BINS)
        if bad !== nothing
            @printf("Mismatch at bin %d: CPU = %d, DPU = %d\n",
                    bad - 1, want[bad], result[bad])
            exit(1)
        end
        println("the result is correct")
    end
end

main()
