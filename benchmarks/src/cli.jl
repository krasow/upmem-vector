function usage(io::IO = stdout)
    println(io, """
    Usage: julia runner.jl [options] [benchmark ...]

    Options:
      --list                       List configured benchmarks and variants
      --variant NAME[,NAME...]     Select variants (repeatable)
      --dpus N[,N...]              Override configured DPU counts
      --elements-per-dpu N[,N...] Override configured problem sizes
      --warmup N                   Override warmup iterations
      --iterations N               Override measured iterations
      --ntrials N                  Launch each benchmark process N times
      --timeout N                  Runtime timeout in seconds (default: 1800)
      --build-timeout N            Build timeout in seconds
      --check                      Generate CPU references and verify results
      --skip-setup                 Skip the shared setup commands
      --generate-only              Write parameters without building or running
      --dry-run                    Print actions without writing or running
      --keep-going                 Continue after a failed command
      --verbose                    Print build and benchmark subprocess output
      --resume                     Skip cases completed in --state
      --state PATH                 Runner checkpoint path
      --csv PATH                   Append per-run timings here
      --config PATH                Read another benchmark TOML file
      --profiles PATH              Read per-benchmark fusion profiles here
      --no-profile                 Ignore fusion profiles
      --reset                      Discard selected benchmarks' saved runs
      -h, --help                   Show this help
    """)
end

function int_list(value::AbstractString)
    values = parse.(Int, filter(!isempty, split(value, ',')))
    isempty(values) && error("expected a comma-separated integer list")
    all(>(0), values) || error("all list values must be positive")
    return values
end

function option_value(args, index::Int, option::AbstractString)
    index == length(args) && error("$option needs a value")
    return args[index + 1]
end

function parse_args(args)
    options = Options()
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("-h", "--help")
            options.action = :help
        elseif arg == "--list"
            options.action = :list
        elseif arg in ("--check", "--skip-setup", "--generate-only",
                       "--dry-run", "--keep-going", "--resume", "--reset",
                       "--verbose")
            setproperty!(options, Symbol(replace(arg[3:end], '-' => '_')), true)
        elseif arg == "--no-profile"
            options.use_profiles = false
        elseif arg in ("--variant", "--dpus", "--elements-per-dpu",
                       "--warmup", "--iterations", "--timeout", "--build-timeout",
                       "--ntrials", "--config", "--profiles", "--csv")
            value = option_value(args, index, arg)
            index += 1
            if arg == "--variant"
                append!(options.variants, filter(!isempty, split(value, ',')))
            elseif arg == "--dpus"
                options.dpus = int_list(value)
            elseif arg == "--elements-per-dpu"
                options.elements_per_dpu = int_list(value)
            elseif arg in ("--warmup", "--iterations")
                parsed = parse(Int, value)
                parsed >= 0 || error("$arg must be non-negative")
                setproperty!(options, Symbol(arg[3:end]), parsed)
            elseif arg == "--ntrials"
                options.ntrials = parse(Int, value)
                options.ntrials > 0 || error("--ntrials must be positive")
            elseif arg in ("--timeout", "--build-timeout")
                parsed = parse(Int, value)
                parsed > 0 || error("$arg must be positive")
                setproperty!(options, Symbol(replace(arg[3:end], '-' => '_')), parsed)
            elseif arg == "--config"
                options.config = abspath(value)
            elseif arg == "--profiles"
                options.profiles = abspath(value)
            else
                options.csv = abspath(value)
            end
        elseif arg == "--state"
            value = option_value(args, index, arg)
            index += 1
            options.state = abspath(value)
        elseif startswith(arg, "-")
            error("unknown option $arg")
        else
            push!(options.benchmarks, arg)
        end
        index += 1
    end
    return options
end

function print_config(config::RunnerConfig)
    for name in config.benchmark_names, spec in config.benchmarks[name]
        sizes = join(spec.elements_per_dpu, ", ")
        variants = join(spec.variants, ", ")
        println(rpad(name, 28), " sizes=[", sizes,
                "] variants=[", variants, "]")
    end
end

function resolve_selection(config::RunnerConfig, options::Options)
    names = isempty(options.benchmarks) ? config.benchmark_names :
            unique(options.benchmarks)
    unknown = setdiff(names, config.benchmark_names)
    isempty(unknown) || error("unknown benchmark(s): $(join(unknown, ", "))")

    variants = isempty(options.variants) ? config.defaults.variants :
               unique(options.variants)
    unknown = setdiff(variants, collect(keys(config.variants)))
    isempty(unknown) || error("unknown variant(s): $(join(unknown, ", "))")
    return names, variants
end

function runner_manifest_entry(config::RunnerConfig, names, requested,
                               options::Options, args)
    benchmarks = Dict{String,Any}[]
    for name in names, spec in config.benchmarks[name]
        variants = selected_variants(spec, requested)
        profile = benchmark_profile(config, spec, variants, options)
        dpus_list = something(options.dpus, spec.dpus, config.defaults.dpus)
        sizes = something(options.elements_per_dpu, spec.elements_per_dpu)
        selected = options.check ? unique(["cpu"; variants]) : variants
        cases = [resolved_case(spec, config.defaults, options, dpus, size)
                 for dpus in dpus_list for size in sizes]
        push!(benchmarks, manifest_dimensions(cases, selected; profile))
    end
    entry = Dict{String,Any}(
        "arguments" => string.(args),
        "config" => config.paths.config,
        "state" => options.state,
        "csv" => results_csv(options),
        "timeout_seconds" => options.timeout,
        "build_timeout_seconds" => options.build_timeout,
        "ntrials" => something(options.ntrials, config.defaults.ntrials),
        "check" => options.check,
        "resume" => options.resume,
        "reset" => options.reset,
        "verbose" => options.verbose,
        "benchmarks" => benchmarks,
    )
    options.use_profiles && (entry["profiles"] = options.profiles)
    return entry
end

function run_cli(args = ARGS)
    options = parse_args(args)
    if options.action == :help
        usage()
        return nothing
    end

    config = load_config(options.config)
    if options.action == :list
        print_config(config)
        return nothing
    end

    names, variants = resolve_selection(config, options)
    if options.reset
        if options.dry_run
            println("[runner] would reset ", join(names, ", "))
        else
            reset_runs(names, options)
        end
    end
    if options.dry_run || options.generate_only
        run_benchmarks(config, names, variants, options)
        return nothing
    end

    path = joinpath(dirname(options.state), "Manifest.toml")
    manifest = begin_manifest(
        path, "benchmark", runner_manifest_entry(config, names, variants, options, args))
    try
        run_benchmarks(config, names, variants, options;
                       invocation = manifest.index)
        finish_manifest(manifest, "complete")
    catch exception
        try
            finish_manifest(manifest, "failed"; failure = sprint(showerror, exception))
        catch manifest_error
            @error "could not update benchmark manifest" path = manifest.path message =
                sprint(showerror, manifest_error)
        end
        rethrow()
    end
    return nothing
end
