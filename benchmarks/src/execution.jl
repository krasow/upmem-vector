function render_template(value, context::Dict{String,String})
    result = string(value)
    for (name, replacement) in context
        result = replace(result, "{$name}" => replacement)
    end
    unresolved = match(r"\{[A-Za-z_][A-Za-z0-9_]*\}", result)
    unresolved === nothing || error("unknown template field $(unresolved.match)")
    return result
end

function variant_context(config::RunnerConfig, case::RunCase)
    return Dict(
        "repo" => config.paths.repo,
        "benchmarks" => config.paths.benchmarks,
        "install" => joinpath(config.paths.repo, "install"),
        "julia" => Base.julia_cmd().exec[1],
        "benchmark" => case.benchmark,
        "dpus" => string(case.dpus),
        "elements" => string(total_elements(case)),
        "fusion_flags" => "",
    )
end

function variant_directory(config::RunnerConfig, variant::VariantSpec,
                           case::RunCase)
    path = abspath(config.paths.benchmarks,
                   render_template(variant.directory,
                                   variant_context(config, case)))
    root = config.paths.benchmarks * "/"
    startswith(path, root) || error("variant directory leaves benchmark tree: $path")
    return path
end

function write_parameters(config::RunnerConfig, source::AbstractString,
                          text::String; dry_run::Bool)
    output = generated_path(source)
    if dry_run
        println("  generate ", relpath(output, config.paths.benchmarks))
    else
        open(output, "w") do io
            write(io, text)
        end
    end
    return output
end

function print_command(config::RunnerConfig, command::AbstractString,
                       directory::AbstractString)
    relative = relpath(directory, config.paths.benchmarks)
    println("  [$relative] $command")
end

function is_implemented(config::RunnerConfig, variant::VariantSpec, case::RunCase)
    return isdir(variant_directory(config, variant, case))
end

function prepare_variant(config::RunnerConfig, variant::VariantSpec,
                         case::RunCase; dry_run::Bool = false)
    context = variant_context(config, case)
    directory = variant_directory(config, variant, case)
    isdir(directory) || error("missing benchmark directory $directory")

    if !dry_run
        for relative in variant.prepare
            path = abspath(directory, render_template(relative, context))
            startswith(path, directory * "/") ||
                error("prepare directory leaves variant tree: $path")
            mkpath(path)
        end
    end

    if variant.parameter_file !== nothing
        source = joinpath(directory, variant.parameter_file)
        isfile(source) || error("missing parameter template $source")
        generated = generated_parameters(
            source; elements = total_elements(case), dpus = case.dpus,
            warmup = case.warmup, iterations = case.iterations,
            check = case.check, seed = case.seed,
            fixed = case.parameters, operation = case.operation)
        write_parameters(config, source, generated; dry_run)
    end
    return directory, context
end

function execute_variant(config::RunnerConfig, variant::VariantSpec,
                         case::RunCase, directory::AbstractString, context;
                         timeout::Int, build_timeout::Int, echo::Bool = true,
                         build_variant::Bool = true)
    if build_variant
        for raw in variant.build
            result = execute_command(
                config, render_template(raw, context), directory, case.dpus;
                timeout = build_timeout, echo)
            successful(result) || return VariantResult(
                command_failure(:build, result), :build, result,
                Dict{String,Any}())
        end
    end
    result = execute_command(
        config, render_template(variant.run, context), directory, case.dpus;
        timeout, timed = true, echo)
    outcome = assess_run(variant.name, case, result)
    !echo && !successful(outcome) && print_command_output(result)
    return outcome
end

function run_variant(config::RunnerConfig, variant::VariantSpec, case::RunCase,
                     options::Options; profile = nothing, invocation::Int = 0,
                     trial::Int = 1, build_variant::Bool = true)
    if build_variant
        directory, context = prepare_variant(
            config, variant, case; dry_run = options.dry_run)
    else
        context = variant_context(config, case)
        directory = variant_directory(config, variant, case)
    end
    empty = CommandResult(:success, 0, "", "", 0.0, "")
    options.generate_only && return VariantResult(
        :complete, :generate, empty, Dict{String,Any}())
    build = profile === nothing ? nothing : profile.build

    if options.dry_run
        if build_variant
            for raw in variant.build
                print_command(config, render_template(raw, context), directory)
            end
        end
        print_command(config, render_template(variant.run, context), directory)
        return VariantResult(:complete, :dry_run, empty, Dict{String,Any}())
    end
    outcome = execute_variant(
        config, variant, case, directory, context;
        timeout = options.timeout, build_timeout = options.build_timeout,
        echo = options.verbose, build_variant)
    record_timing(results_csv(options), case, variant.name, outcome;
                  invocation, trial, build)
    return outcome
end

function setup_context(config::RunnerConfig, profile)
    return Dict(
        "repo" => config.paths.repo,
        "benchmarks" => config.paths.benchmarks,
        "install" => joinpath(config.paths.repo, "install"),
        "julia" => Base.julia_cmd().exec[1],
        "benchmark" => "",
        "dpus" => "1",
        "elements" => "1",
        "fusion_flags" => fusion_flags(profile),
    )
end

function run_setup(config::RunnerConfig, active_variants::Vector{String},
                   options::Options, profile = nothing)
    options.skip_setup && return nothing
    config.setup === nothing && return nothing
    isempty(intersect(Set(config.setup.variants), Set(active_variants))) && return nothing
    context = setup_context(config, profile)
    for raw in config.setup.commands
        command = render_template(raw, context)
        if options.dry_run
            print_command(config, command, config.paths.benchmarks)
            continue
        end
        result = execute_command(config, command, config.paths.benchmarks, 1;
                                 timeout = options.build_timeout,
                                 echo = options.verbose)
        successful(result) || return result
    end
    return nothing
end


function benchmark_profile(spec::BenchmarkSpec, active::Vector{String},
                           options::Options)
    options.use_profiles || return nothing
    isempty(intersect(Set(active), Set(("polymerpim", "julia")))) && return nothing
    return load_fusion_profile(options.profiles, spec.name)
end

function resolved_case(spec::BenchmarkSpec, defaults::RunnerDefaults,
                       options::Options, dpus::Int, elements_per_dpu::Int)
    return RunCase(
        spec.name, dpus, elements_per_dpu,
        something(options.warmup, spec.warmup, defaults.warmup),
        something(options.iterations, spec.iterations, defaults.iterations),
        options.check, something(spec.seed, defaults.seed), spec.parameters,
        spec.operation)
end

function selected_variants(spec::BenchmarkSpec, requested::Vector{String})
    allowed = Set(spec.variants)
    return filter(in(allowed), requested)
end

mutable struct RunState
    path::String
    completed::Set{String}
    records::Vector{Dict{String,Any}}
    enabled::Bool
end

function save_state(state::RunState)
    state.enabled || return
    mkpath(dirname(state.path))
    temporary = state.path * ".tmp"
    open(temporary, "w") do io
        write_toml(io, Dict("version" => 3, "completed" => state.records))
    end
    mv(temporary, state.path; force = true)
end

function archive_runner_state(path::AbstractString)
    archived = path * ".legacy"
    index = 1
    while ispath(archived)
        index += 1
        archived = path * ".legacy.$index"
    end
    mv(path, archived)
    println("Archived legacy runner checkpoint: $archived")
end

function run_state(options::Options)
    enabled = !(options.dry_run || options.generate_only)
    records = Dict{String,Any}[]
    fresh = !options.resume
    if enabled && options.resume && isfile(options.state)
        raw = TOML.parsefile(options.state)
        version = get(raw, "version", 0)
        if version in (1, 2)
            archive_runner_state(options.state)
            fresh = true
        elseif version == 3
            append!(records, (Dict{String,Any}(entry) for entry in
                              get(raw, "completed", Dict{String,Any}[])))
        else
            error("unsupported runner state $(options.state)")
        end
    end
    state = RunState(options.state, Set(run_key.(records)), records, enabled)
    fresh && save_state(state)
    return state
end

function run_record(case::RunCase, variant::AbstractString, profile,
                    trial::Int = 1)
    record = Dict{String,Any}(
        "benchmark" => case.benchmark,
        "variant" => variant,
        "dpus" => case.dpus,
        "elements_per_dpu" => case.elements_per_dpu,
        "warmup" => case.warmup,
        "iterations" => case.iterations,
        "trial" => trial,
        "check" => case.check,
        "seed" => case.seed,
    )
    isempty(case.parameters) || (record["parameters"] = case.parameters)
    case.operation === nothing || (record["operation"] = case.operation)
    profile === nothing || (record["fusion_build"] = profile.build)
    return record
end

function run_key(record)
    buffer = IOBuffer()
    TOML.print(buffer, record; sorted = true)
    return bytes2hex(sha256(take!(buffer)))
end

function remove_benchmark_rows(path::AbstractString, names)
    isfile(path) || return 0
    lines = readlines(path)
    isempty(lines) && return 0
    columns = split(first(lines), ',')
    benchmark_column = findfirst(==("benchmark"), columns)
    benchmark_column === nothing && error("$path has no benchmark column")
    selected = Set(names)
    kept = String[first(lines)]
    removed = 0
    for line in Iterators.drop(lines, 1)
        fields = split(line, ','; limit = benchmark_column + 1)
        if length(fields) >= benchmark_column && fields[benchmark_column] in selected
            removed += 1
        else
            push!(kept, line)
        end
    end
    removed == 0 && return 0
    temporary = path * ".tmp"
    open(temporary, "w") do io
        foreach(line -> println(io, line), kept)
    end
    mv(temporary, path; force = true)
    return removed
end

function reset_runs(names, options::Options)
    records = Dict{String,Any}[]
    if isfile(options.state)
        raw = TOML.parsefile(options.state)
        version = get(raw, "version", 0)
        if version in (1, 2)
            archive_runner_state(options.state)
        elseif version == 3
            append!(records, (Dict{String,Any}(entry) for entry in
                              get(raw, "completed", Dict{String,Any}[])))
        else
            error("unsupported runner state $(options.state)")
        end
    end
    selected = Set(names)
    before = length(records)
    filter!(record -> !(get(record, "benchmark", "") in selected), records)
    state = RunState(options.state, Set(run_key.(records)), records, true)
    save_state(state)
    csv = results_csv(options)
    rows = remove_benchmark_rows(csv, names)
    sections = remove_benchmark_rows(sections_csv(csv), names)
    options.resume = true
    println("[runner] reset ", join(names, ", "), ": ",
            before - length(records), " checkpoints, ", rows,
            " runs, ", sections, " sections")
end

function trial_summary(timing, iterations::Int)
    metric(key) = get(timing, key, "") isa Number ?
                  @sprintf("%10.3f", timing[key]) : lpad("-", 10)
    wall = get(timing, "real_s", "")
    wall_text = wall isa Number ? @sprintf("%8.2f", wall) : lpad("-", 8)
    return @sprintf("iterations=%4d", iterations) *
           "  mean=$(metric("time")) ms" *
           "  stddev=$(metric("stddev")) ms" *
           "  min=$(metric("min")) ms" *
           "  max=$(metric("max")) ms" *
           "  wall=$wall_text s"
end

function run_benchmarks(config::RunnerConfig, benchmark_names::Vector{String},
                        requested::Vector{String}, options::Options;
                        invocation::Int = 0)
    failures = String[]
    setup_key = nothing
    state = run_state(options)

    for name in benchmark_names, spec in config.benchmarks[name]
        variants = selected_variants(spec, requested)
        profile = benchmark_profile(spec, variants, options)
        profile === nothing || println(
            "Using fusion profile ", relpath(profile.path, config.paths.benchmarks),
            ": ", fusion_flags(profile))
        dpus_list = something(options.dpus, spec.dpus, config.defaults.dpus)
        sizes = something(options.elements_per_dpu, spec.elements_per_dpu)
        ntrials = options.generate_only ? 1 :
                  something(options.ntrials, config.defaults.ntrials)
        for dpus in dpus_list, elements_per_dpu in sizes
            case = resolved_case(spec, config.defaults, options, dpus,
                                 elements_per_dpu)
            @info "Benchmark case" benchmark = name dpus elements_per_dpu total_elements = total_elements(case) iterations = case.iterations ntrials
            ordered = options.check ? unique(["cpu"; variants]) : variants
            for variant_name in ordered
                variant = config.variants[variant_name]
                if !is_implemented(config, variant, case)
                    println("  skip $variant_name (not implemented)")
                    continue
                end
                keyed_profile = variant_name in ("polymerpim", "julia") ? profile : nothing
                pending = Tuple{Int,Dict{String,Any},String}[]
                for trial in 1:ntrials
                    record = run_record(case, variant_name, keyed_profile, trial)
                    key = run_key(record)
                    if options.resume && key in state.completed
                        println("  skip $variant_name trial $trial/$ntrials (completed)")
                    else
                        push!(pending, (trial, record, key))
                    end
                end
                if isempty(pending)
                    println("  skip $variant_name (completed)")
                    continue
                end
                needs_setup = config.setup !== nothing &&
                    variant_name in config.setup.variants
                next_setup_key = fusion_flags(profile)
                if needs_setup && next_setup_key != setup_key
                    description = isempty(next_setup_key) ?
                                  "default fusion parameters" : next_setup_key
                    println("-- setup ($description)")
                    setup_failure = run_setup(
                        config, [variant_name], options, profile)
                    setup_failure === nothing || error(
                        "benchmark setup $(setup_failure.status)" *
                        (setup_failure.exit_code === nothing ? "" :
                         " (exit $(setup_failure.exit_code))"))
                    setup_key = next_setup_key
                end
                println("-- $variant_name")
                built = false
                for (trial, record, key) in pending
                    println("   trial $trial/$ntrials")
                    outcome = run_variant(
                        config, variant, case, options;
                        profile = keyed_profile, invocation, trial,
                        build_variant = !built)
                    built = outcome.phase != :build
                    if successful(outcome)
                        get(outcome.timing, "time", "") isa Number &&
                            println("      ", trial_summary(
                                outcome.timing, case.iterations))
                        push!(state.completed, key)
                        push!(state.records, record)
                        save_state(state)
                        continue
                    end
                    println("      $(outcome.status)")
                    push!(failures,
                          "$name/$variant_name/$dpus/$elements_per_dpu/trial-$trial: " *
                          string(outcome.status))
                    options.keep_going || error(
                        "benchmark $name/$variant_name trial $trial failed: " *
                        string(outcome.status) *
                        (outcome.command.exit_code === nothing ? "" :
                         " (exit $(outcome.command.exit_code))"))
                end
            end
        end
    end
    isempty(failures) || error("failed cases:\n  " * join(failures, "\n  "))
    return nothing
end
