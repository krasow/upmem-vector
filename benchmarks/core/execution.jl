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
                         timeout::Int, build_timeout::Int)
    for raw in variant.build
        result = execute_command(
            config, render_template(raw, context), directory, case.dpus;
            timeout = build_timeout, echo = true)
        successful(result) || return VariantResult(
            command_failure(:build, result), :build, result,
            Dict{String,Any}())
    end
    result = execute_command(
        config, render_template(variant.run, context), directory, case.dpus;
        timeout, timed = true, echo = true)
    return assess_run(variant.name, case, result)
end

function run_variant(config::RunnerConfig, variant::VariantSpec, case::RunCase,
                     options::Options; profile = nothing, invocation::Int = 0)
    directory, context = prepare_variant(
        config, variant, case; dry_run = options.dry_run)
    empty = CommandResult(:success, 0, "", "", 0.0, "")
    options.generate_only && return VariantResult(
        :complete, :generate, empty, Dict{String,Any}())
    build = profile === nothing ? nothing : profile.build

    if options.dry_run
        for raw in variant.build
            print_command(config, render_template(raw, context), directory)
        end
        print_command(config, render_template(variant.run, context), directory)
        return VariantResult(:complete, :dry_run, empty, Dict{String,Any}())
    end
    outcome = execute_variant(
        config, variant, case, directory, context;
        timeout = options.timeout, build_timeout = options.build_timeout)
    record_timing(results_csv(options), case, variant.name, outcome;
                  invocation, build)
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
                                 timeout = options.build_timeout, echo = true)
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
    enabled::Bool
end

function save_state(state::RunState)
    state.enabled || return
    mkpath(dirname(state.path))
    temporary = state.path * ".tmp"
    open(temporary, "w") do io
        write_toml(io, Dict("version" => 1,
                            "completed" => sort(collect(state.completed))))
    end
    mv(temporary, state.path; force = true)
end

function run_state(options::Options)
    enabled = !(options.dry_run || options.generate_only)
    completed = Set{String}()
    if enabled && options.resume && isfile(options.state)
        raw = TOML.parsefile(options.state)
        get(raw, "version", 0) == 1 || error("unsupported runner state $(options.state)")
        union!(completed, string.(get(raw, "completed", String[])))
    end
    state = RunState(options.state, completed, enabled)
    options.resume || save_state(state)
    return state
end

function run_key(case::RunCase, variant::AbstractString, profile)
    parameters = sort(collect(case.parameters); by = first)
    return repr((case.benchmark, variant, case.dpus, case.elements_per_dpu,
                 case.warmup, case.iterations, case.check, case.seed,
                 parameters, case.operation, fusion_flags(profile)))
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
        for dpus in dpus_list, elements_per_dpu in sizes
            case = resolved_case(spec, config.defaults, options, dpus,
                                 elements_per_dpu)
            println("\n== $name: $dpus DPUs × $elements_per_dpu elements/DPU = ",
                    total_elements(case), " ==")
            ordered = options.check ? unique(["cpu"; variants]) : variants
            for variant_name in ordered
                variant = config.variants[variant_name]
                if !is_implemented(config, variant, case)
                    println("  skip $variant_name (not implemented)")
                    continue
                end
                keyed_profile = variant_name in ("polymerpim", "julia") ? profile : nothing
                key = run_key(case, variant_name, keyed_profile)
                if options.resume && key in state.completed
                    println("  skip $variant_name (completed)")
                    continue
                end
                needs_setup = config.setup !== nothing &&
                    variant_name in config.setup.variants
                next_setup_key = fusion_flags(profile)
                if needs_setup && next_setup_key != setup_key
                    setup_failure = run_setup(
                        config, [variant_name], options, profile)
                    setup_failure === nothing || error(
                        "benchmark setup $(setup_failure.status)" *
                        (setup_failure.exit_code === nothing ? "" :
                         " (exit $(setup_failure.exit_code))"))
                    setup_key = next_setup_key
                end
                println("-- $variant_name")
                outcome = run_variant(config, variant, case, options;
                                      profile = keyed_profile, invocation)
                if successful(outcome)
                    push!(state.completed, key)
                    save_state(state)
                    continue
                end
                push!(failures,
                      "$name/$variant_name/$dpus/$elements_per_dpu: $(outcome.status)")
                options.keep_going || error(
                    "benchmark $name/$variant_name failed: $(outcome.status)" *
                    (outcome.command.exit_code === nothing ? "" :
                     " (exit $(outcome.command.exit_code))"))
            end
        end
    end
    isempty(failures) || error("failed cases:\n  " * join(failures, "\n  "))
    return nothing
end
