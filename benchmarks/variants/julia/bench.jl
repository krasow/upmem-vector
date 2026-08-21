# Julia counterpart of benchmarks/benchmark.h.
#
# The output formats here are load-bearing: benchmarks/core/benchmark.py parses
# them with regexes keyed on the label, so they must match benchmark.h exactly.
#   <label> (ms): mean=%.3f stddev=%.3f min=%.3f max=%.3f n=%d
#   <label>_stage_<stage> (ms): %f

module Bench

using Printf

export BenchStats, stats_update!, stats_print,
       BenchStages, stage_begin!, stage_end!, stages_report,
       load_bin!, nr_dpus, elapsed_us

# Stage names and order match BENCH_STAGE_NAMES in benchmark.h.
const STAGE_NAMES = ("alloc", "load", "transpose", "init",
                     "write", "kernel", "read", "merge")

# ---- running statistics (Welford, as bench_stats_update) ----

mutable struct BenchStats
    mean::Float64
    M2::Float64
    min_us::Float64
    max_us::Float64
    count::Int
end
BenchStats() = BenchStats(0.0, 0.0, 0.0, 0.0, 0)

function stats_update!(s::BenchStats, elapsed_us::Float64)
    s.count += 1
    delta = elapsed_us - s.mean
    s.mean += delta / s.count
    s.M2 += delta * (elapsed_us - s.mean)
    if s.count == 1
        s.min_us = elapsed_us
        s.max_us = elapsed_us
    else
        s.min_us = min(s.min_us, elapsed_us)
        s.max_us = max(s.max_us, elapsed_us)
    end
    return s
end

function stats_print(label::AbstractString, s::BenchStats)
    stddev_ms = s.count > 1 ? sqrt(s.M2 / (s.count - 1)) / 1000.0 : 0.0
    @printf("%s (ms): mean=%.3f stddev=%.3f min=%.3f max=%.3f n=%d\n",
            label, s.mean / 1000.0, stddev_ms,
            s.min_us / 1000.0, s.max_us / 1000.0, s.count)
end

# ---- per-stage accumulators ----

mutable struct BenchStages
    us::Vector{Float64}
    current::Int          # index into STAGE_NAMES, 0 when idle
    started_ns::UInt64
end
BenchStages() = BenchStages(zeros(Float64, length(STAGE_NAMES)), 0, UInt64(0))

function stage_begin!(s::BenchStages, stage::Symbol)
    idx = findfirst(==(String(stage)), STAGE_NAMES)
    idx === nothing && error("unknown stage $stage")
    s.current = idx
    s.started_ns = time_ns()
    return nothing
end

function stage_end!(s::BenchStages)
    s.current == 0 && return nothing
    s.us[s.current] += (time_ns() - s.started_ns) / 1000.0
    s.current = 0
    return nothing
end

function stages_report(label::AbstractString, s::BenchStages)
    for (i, name) in enumerate(STAGE_NAMES)
        @printf("%s_stage_%s (ms): %f\n", label, name, s.us[i] / 1000.0)
    end
end

# Time one region into a stage and return the block's value.
macro stage(stages, name, body)
    quote
        stage_begin!($(esc(stages)), $(esc(name)))
        local v = $(esc(body))
        stage_end!($(esc(stages)))
        v
    end
end

elapsed_us(from_ns::UInt64) = (time_ns() - from_ns) / 1000.0

# ---- inputs ----

nr_dpus() = parse(Int, get(ENV, "NR_DPUS", "64"))

"""
    load_bin!(path, data)

Read `length(data)` elements into `data`. A file holding 32-bit values is
widened when `data` is 64-bit, matching how the C++ benchmarks reuse a single
reference dataset across bit widths.
"""
function load_bin!(path::AbstractString, data::AbstractVector)
    open(path, "r") do io
        want = length(data) * sizeof(eltype(data))
        have = stat(path).size
        if have == want
            read!(io, data)
        elseif eltype(data) === Int64 && have == want ÷ 2
            tmp = Vector{Int32}(undef, length(data))
            read!(io, tmp)
            @. data = Int64(tmp)
        else
            error("reference file $path has size $have, expected $want")
        end
    end
    return data
end

end # module Bench
