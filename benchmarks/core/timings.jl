const APP_TIME_FORMAT = "APP_TIME real=%e user=%U sys=%S maxrss=%M"
const APP_TIME_RE = r"APP_TIME real=([\d.]+) user=([\d.]+) sys=([\d.]+) maxrss=(\d+)"
const STAGE_NAMES = ("alloc", "load", "transpose", "init", "write", "kernel",
                     "read", "merge")
const MEASURE_COLUMNS = [
    "time", "stddev", "min", "max", "warmup_ms",
    "real_s", "user_s", "sys_s", "max_rss_kb",
    ("$(stage)_ms" for stage in STAGE_NAMES)...,
    ("$(stage)_cold_ms" for stage in STAGE_NAMES)...,
]
const RUN_COLUMNS = [
    "timestamp", "invocation", "benchmark", "variant", "phase", "status",
    "command_status", "exit_code", "elapsed_s", "detail",
    "elements_per_dpu", "total_elements", "dpus", "warmup", "iterations",
    "check", "seed", "operation", "parameters", "fusion_flags",
    MEASURE_COLUMNS...,
]

struct CommandResult
    status::Symbol
    exit_code::Union{Nothing,Int}
    stdout::String
    stderr::String
    elapsed_s::Float64
    detail::String
end

successful(result::CommandResult) = result.status == :success

struct VariantResult
    status::Symbol
    phase::Symbol
    command::CommandResult
    timing::Dict{String,Any}
end

successful(result::VariantResult) = result.status == :complete

function execute_command(config::RunnerConfig, command::AbstractString,
                         directory::AbstractString, dpus::Int;
                         timeout::Union{Nothing,Int} = nothing,
                         timed::Bool = false, echo::Bool = false)
    executable = timed ? "/usr/bin/time -f '$APP_TIME_FORMAT' $command" : command
    wrapped = "source \"$(config.paths.environment)\" && $executable"
    cmd = `/bin/bash -lc $wrapped`
    timeout === nothing || (cmd = `/usr/bin/timeout --signal=TERM --kill-after=10 $timeout $cmd`)
    cmd = Cmd(cmd; dir = directory)
    output, errors = IOBuffer(), IOBuffer()
    process = nothing
    started = time_ns()
    status, exit_code, detail = try
        process = run(pipeline(addenv(cmd, "NR_DPUS" => string(dpus));
                               stdout = output, stderr = errors); wait = false)
        wait(process)
        code = process.exitcode
        state = code == 0 ? :success :
                timeout !== nothing && code in (124, 137) ? :timed_out : :failed
        state, code, ""
    catch exception
        if exception isa InterruptException
            process === nothing || try
                kill(process, Base.SIGKILL)
            catch
            end
            rethrow()
        end
        message = hasproperty(exception, :msg) ?
                  string(getproperty(exception, :msg)) : string(nameof(typeof(exception)))
        :launch_failed, nothing, message
    end
    elapsed = (time_ns() - started) / 1.0e9
    out, err = String(take!(output)), String(take!(errors))
    if echo || status != :success
        print(stdout, out)
        print(stderr, err)
        isempty(detail) || println(stderr, detail)
    end
    return CommandResult(status, exit_code, out, err, elapsed, detail)
end

function parsed_value(pattern::Regex, text::AbstractString)
    found = match(pattern, text)
    return found === nothing ? "" : parse(Float64, found.captures[1])
end

function parse_timings(label::AbstractString, out::AbstractString,
                       err::AbstractString)
    escaped = "\\Q$(label)\\E"
    values = Dict{String,Any}(
        "time" => parsed_value(
            Regex("$escaped\\s*\\(ms\\):\\s*(?:mean=)?([0-9.]+)"), out),
        "stddev" => parsed_value(
            Regex("$escaped\\s*\\(ms\\):.*?stddev=([0-9.]+)"), out),
        "min" => parsed_value(
            Regex("$escaped\\s*\\(ms\\):.*?min=([0-9.]+)"), out),
        "max" => parsed_value(
            Regex("$escaped\\s*\\(ms\\):.*?max=([0-9.]+)"), out),
        "warmup_ms" => parsed_value(
            Regex("$(escaped)_warmup\\s*\\(ms\\):\\s*(?:mean=)?([0-9.]+)"), out),
    )
    app = match(APP_TIME_RE, err)
    for (index, name) in enumerate(("real_s", "user_s", "sys_s", "max_rss_kb"))
        values[name] = app === nothing ? "" : parse(Float64, app.captures[index])
    end
    for stage in STAGE_NAMES
        values["$(stage)_ms"] = parsed_value(
            Regex("$(escaped)_stage_$stage\\s*\\(ms\\):\\s*([0-9.]+)"), out)
        values["$(stage)_cold_ms"] = parsed_value(
            Regex("$(escaped)_cold_stage_$stage\\s*\\(ms\\):\\s*([0-9.]+)"), out)
    end
    return values
end

const PASS_MARKERS = ("All results match", "the result is correct")
const FAIL_MARKERS = ("Mismatch at index", "result mismatch at position",
                      "Mismatch at gradient", "Mismatch at coeff",
                      "Mismatch at centroid", "Mismatch: got")

function parse_verification(output::AbstractString)
    text = lowercase(output)
    any(marker -> occursin(lowercase(marker), text), FAIL_MARKERS) && return false
    any(marker -> occursin(lowercase(marker), text), PASS_MARKERS) && return true
    return nothing
end

function command_failure(phase::Symbol, command::CommandResult)
    return command.status == :timed_out ? Symbol(phase, "_timed_out") :
           command.status == :launch_failed ? Symbol(phase, "_launch_failed") :
           Symbol(phase, "_failed")
end

function assess_run(variant::AbstractString, case::RunCase,
                    command::CommandResult)
    label = variant == "cpu" ? "cpu_baseline" : variant
    timing = parse_timings(label, command.stdout, command.stderr)
    successful(command) || return VariantResult(
        command_failure(:runtime, command), :run, command, timing)
    timing["time"] isa Number ||
        return VariantResult(:timing_missing, :run, command, timing)
    if case.check && variant != "cpu"
        verified = parse_verification(command.stdout * "\n" * command.stderr)
        verified === true || return VariantResult(
            verified === false ? :verification_failed : :verification_missing,
            :verify, command, timing)
    end
    return VariantResult(:complete, :run, command, timing)
end

csv_cell(value) = occursin(r"[\",\n]", string(value)) ?
    "\"" * replace(string(value), "\"" => "\"\"") * "\"" : string(value)

results_csv(options::Options) = something(
    options.csv, joinpath(dirname(options.state), "runs.csv"))

function record_timing(path::AbstractString, case::RunCase, variant::AbstractString,
                       result::VariantResult; invocation::Int = 0,
                       build = nothing)
    timing = result.timing
    row = Dict{String,Any}(
        "timestamp" => Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"),
        "invocation" => invocation,
        "benchmark" => case.benchmark,
        "variant" => variant,
        "phase" => result.phase,
        "status" => result.status,
        "command_status" => result.command.status,
        "exit_code" => something(result.command.exit_code, ""),
        "elapsed_s" => round(result.command.elapsed_s; digits = 6),
        "detail" => result.command.detail,
        "elements_per_dpu" => case.elements_per_dpu,
        "total_elements" => total_elements(case),
        "dpus" => case.dpus,
        "warmup" => case.warmup,
        "iterations" => case.iterations,
        "check" => case.check,
        "seed" => case.seed,
        "operation" => something(case.operation, ""),
        "parameters" => repr(sort(collect(case.parameters); by = first)),
        "fusion_flags" => build === nothing ? "" :
                          join(("$knob=$(build[knob])" for knob in
                                FUSION_BUILD_KNOBS), " "),
    )
    merge!(row, timing)
    mkpath(dirname(path))
    exists = isfile(path)
    if exists
        readline(path) == join(RUN_COLUMNS, ',') || error("unexpected CSV schema in $path")
    end
    open(path, "a") do io
        exists || println(io, join(RUN_COLUMNS, ','))
        println(io, join((csv_cell(get(row, column, "")) for column in RUN_COLUMNS), ','))
    end
    return timing
end
