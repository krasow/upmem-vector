using Test
using TOML

include(joinpath(@__DIR__, "..", "core", "BenchmarkRunner.jl"))
using .BenchmarkRunner

const BENCHMARKS = normpath(joinpath(@__DIR__, ".."))

@testset "configuration" begin
    config = load_config()
    @test "elementwise-interpreter" in config.benchmark_names
    @test config.benchmarks["elementwise-interpreter"][1].variants == ["baseline"]
    @test sort(collect(keys(config.variants))) ==
          ["baseline", "cpu", "julia", "polymerpim", "simplepim"]
end

@testset "arguments" begin
    options = parse_args([
        "elementwise", "--variant", "julia,baseline", "--dpus", "2,4",
        "--elements-per-dpu", "64", "--warmup", "0", "--iterations", "1",
        "--timeout", "7", "--build-timeout", "9",
        "--check", "--skip-setup", "--csv", "/tmp/benchmark-times.csv",
    ])
    @test options.benchmarks == ["elementwise"]
    @test options.variants == ["julia", "baseline"]
    @test options.dpus == [2, 4]
    @test options.elements_per_dpu == [64]
    @test options.warmup == 0
    @test options.iterations == 1
    @test options.timeout == 7
    @test options.build_timeout == 9
    @test options.check && options.skip_setup
    @test options.csv == "/tmp/benchmark-times.csv"
end

@testset "timing capture" begin
    output = """
    polymerpim_warmup (ms): mean=12.500 stddev=0.200 min=12.300 max=12.700 n=2
    polymerpim (ms): mean=3.250 stddev=0.125 min=3.100 max=3.400 n=4
    polymerpim_stage_alloc (ms): 1.500
    polymerpim_stage_kernel (ms): 13.000
    polymerpim_cold_stage_kernel (ms): 9.750
    polymerpim_cold_stage_merge (ms): 2.250
    """
    errors = "trace output\nAPP_TIME real=1.23 user=0.80 sys=0.10 maxrss=4567\n"
    timing = BenchmarkRunner.parse_timings("polymerpim", output, errors)
    @test timing["time"] == 3.25
    @test timing["stddev"] == 0.125
    @test timing["min"] == 3.1
    @test timing["max"] == 3.4
    @test timing["warmup_ms"] == 12.5
    @test timing["real_s"] == 1.23
    @test timing["max_rss_kb"] == 4567.0
    @test timing["alloc_ms"] == 1.5
    @test timing["kernel_cold_ms"] == 9.75
    @test timing["merge_cold_ms"] == 2.25
    @test timing["read_ms"] == ""

    mktempdir() do directory
        case = BenchmarkRunner.RunCase(
            "elementwise", 2, 64, 2, 4, false, 1,
            Dict{String,Any}(), "a + b")
        command = BenchmarkRunner.CommandResult(:success, 0, output, errors, 1.3, "")
        result = BenchmarkRunner.assess_run("polymerpim", case, command)
        path = joinpath(directory, "runs.csv")
        BenchmarkRunner.record_timing(path, case, "polymerpim", result;
                                      invocation = 3,
                                      build = BenchmarkRunner.DEFAULT_FUSION_BUILD)
        lines = readlines(path)
        @test length(lines) == 2
        @test startswith(lines[1], "timestamp,invocation,benchmark,variant,phase,status")
        @test occursin("polymerpim,run,complete,success,0", lines[2])
        @test occursin(",FUSION_LOOKAHEAD,MAX_HFUSE_CHAINS,JIT_BATCH_SIZE,", lines[1])
        @test occursin(",128,10,16,128,11,4,", lines[2])
        sections = readlines(joinpath(directory, "runs.sections.csv"))
        @test length(sections) == 5
        @test occursin(",alloc,measured,1.5", sections[2])
        @test any(line -> occursin(",kernel,cold,9.75", line), sections)
    end

    unchecked = BenchmarkRunner.RunCase(
        "elementwise", 1, 1, 0, 1, false, 1, Dict{String,Any}(), nothing)
    checked = BenchmarkRunner.RunCase(
        "elementwise", 1, 1, 0, 1, true, 1, Dict{String,Any}(), nothing)
    empty = BenchmarkRunner.CommandResult(:success, 0, "", "", 0.1, "")
    mismatch = BenchmarkRunner.CommandResult(
        :success, 0,
        "baseline (ms): 1.0\nMismatch at index 0\n", "", 0.1, "")
    @test BenchmarkRunner.assess_run("baseline", unchecked, empty).status ==
          :timing_missing
    @test BenchmarkRunner.assess_run("baseline", checked, mismatch).status ==
          :verification_failed
    @test BenchmarkRunner.command_failure(
        :build, BenchmarkRunner.CommandResult(:timed_out, 124, "", "", 2.0, "")) ==
          :build_timed_out

    config = load_config()
    failed = BenchmarkRunner.execute_command(
        config, "exit 7", BENCHMARKS, 1; timeout = 2)
    @test failed.status == :failed
    @test failed.exit_code == 7
    @test isempty(failed.detail)

    timed_out = BenchmarkRunner.execute_command(
        config, "sleep 2", BENCHMARKS, 1; timeout = 1)
    @test timed_out.status == :timed_out
    @test timed_out.exit_code == 124
end

@testset "variant failures" begin
    config = load_config()
    case = BenchmarkRunner.RunCase(
        "elementwise", 1, 1, 0, 1, false, 1, Dict{String,Any}(), nothing)
    directory = "variants/baseline/elementwise"

    mktempdir() do temporary
        path = joinpath(temporary, "build.csv")
        variant = BenchmarkRunner.VariantSpec(
            "broken", directory, nothing, String[], ["exit 7"], "true")
        result = BenchmarkRunner.run_variant(
            config, variant, case, BenchmarkRunner.Options(csv = path);
            invocation = 4)
        @test result.status == :build_failed
        @test result.phase == :build
        @test occursin("broken,build,build_failed,failed,7", read(path, String))
    end

    mktempdir() do temporary
        path = joinpath(temporary, "missing.csv")
        variant = BenchmarkRunner.VariantSpec(
            "broken", directory, nothing, String[], String[], "true")
        result = BenchmarkRunner.run_variant(
            config, variant, case, BenchmarkRunner.Options(csv = path))
        @test result.status == :timing_missing
        @test occursin("broken,run,timing_missing,success,0", read(path, String))
    end

    mktempdir() do temporary
        path = joinpath(temporary, "mismatch.csv")
        checked = BenchmarkRunner.RunCase(
            "elementwise", 1, 1, 0, 1, true, 1, Dict{String,Any}(), nothing)
        variant = BenchmarkRunner.VariantSpec(
            "broken", directory, nothing, String[], String[],
            "printf 'broken (ms): 1.0\\nMismatch at index 0\\n'")
        result = BenchmarkRunner.run_variant(
            config, variant, checked, BenchmarkRunner.Options(csv = path))
        @test result.status == :verification_failed
        @test result.phase == :verify
        @test occursin("broken,verify,verification_failed,success,0",
                       read(path, String))
    end
end

@testset "parameter generation" begin
    c_source = joinpath(BENCHMARKS, "variants", "baseline", "elementwise",
                        "Param.h")
    c_params = generated_parameters(
        c_source; elements = 128, dpus = 2, warmup = 0, iterations = 1,
        check = true, seed = 7, fixed = Dict{String,Any}(),
        operation = "a + b")
    @test occursin("const uint32_t dpu_number = 2;", c_params)
    @test occursin("uint64_t nr_elements = 128;", c_params)
    @test occursin("#define OPERATION(a, b) a + b", c_params)

    julia_source = joinpath(BENCHMARKS, "variants", "julia", "elementwise",
                            "Param.jl")
    julia_params = generated_parameters(
        julia_source; elements = 128, dpus = 2, warmup = 0, iterations = 1,
        check = false, seed = 7, fixed = Dict{String,Any}(),
        operation = "a + b")
    @test occursin("const N = 128", julia_params)
    @test occursin("const check_correctness = 0", julia_params)
    @test occursin("operation(a, b) = @. a + b", julia_params)
end

@testset "dry run" begin
    text = mktemp() do _, output
        redirect_stdout(output) do
            run_cli([
                "elementwise-interpreter", "--variant", "baseline", "--dpus", "2",
                "--elements-per-dpu", "64", "--warmup", "0", "--iterations", "1",
                "--check", "--skip-setup", "--dry-run",
            ])
        end
        flush(output)
        seekstart(output)
        read(output, String)
    end
    @test occursin("skip cpu (not implemented)", text)
    @test occursin("variants/baseline/elementwise-interpreter", text)
    @test occursin("generate variants/baseline/elementwise-interpreter", text)
end

@testset "fusion search" begin
    seed = copy(BenchmarkRunner.DEFAULT_FUSION_BUILD)
    search = deepcopy(BenchmarkRunner.DEFAULT_FUSION_SEARCH)
    search["FUSION_LOOKAHEAD"] = [1, 2, 3]
    for knob in BenchmarkRunner.CORE_FUSION_KNOBS[2:end]
        search[knob] = [seed[knob]]
    end
    evaluations = Ref(0)
    evaluate = build -> begin
        evaluations[] += 1
        objective = Float64((build["FUSION_LOOKAHEAD"] - 2)^2 + 1)
        BenchmarkRunner.TuneResult("ok", objective, [objective], [(1, 1)])
    end
    best, result = BenchmarkRunner.coordinate_descent(seed, search, 3, evaluate)
    @test best["FUSION_LOOKAHEAD"] == 2
    @test result.objective == 1.0
    @test evaluations[] < 3 * sum(length, values(search))
end

@testset "fusion profiles" begin
    mktempdir() do directory
        profile = Dict(
            "version" => 1,
            "benchmark" => "elementwise",
            "build" => Dict(knob => index for
                            (index, knob) in enumerate(BenchmarkRunner.FUSION_BUILD_KNOBS)),
        )
        open(joinpath(directory, "elementwise.toml"), "w") do io
            TOML.print(io, profile)
        end
        loaded = BenchmarkRunner.load_fusion_profile(directory, "elementwise")
        @test loaded.build["FUSION_LOOKAHEAD"] == 1

        text = mktemp() do _, output
            redirect_stdout(output) do
                run_cli([
                    "elementwise", "--variant", "polymerpim", "--dpus", "2",
                    "--elements-per-dpu", "64", "--warmup", "0", "--iterations", "1",
                    "--profiles", directory, "--dry-run",
                ])
            end
            flush(output)
            seekstart(output)
            read(output, String)
        end
        @test occursin("FUSION_LOOKAHEAD=1", text)
        @test occursin("Using fusion profile", text)
    end
end

@testset "fusion resume" begin
    config = load_config()
    spec = config.benchmarks["elementwise"][1]
    @test BenchmarkRunner.TuneOptions().resume
    reset = BenchmarkRunner.parse_tune_args(["elementwise", "--reset"])
    @test reset.reset && !reset.resume
    mktempdir() do directory
        options = BenchmarkRunner.TuneOptions(
            dpus = [2], elements_per_dpu = [64], warmup = 0, iterations = 1,
            checkpoints = directory)
        build = copy(BenchmarkRunner.DEFAULT_FUSION_BUILD)
        result = BenchmarkRunner.TuneResult("ok", 4.5, [4.5], [(2, 64)])
        checkpoint = BenchmarkRunner.load_checkpoint(config, spec, options)
        @test BenchmarkRunner.record!(checkpoint, build, result, 1,
                                      "FUSION_LOOKAHEAD", 128)
        resumed = BenchmarkRunner.load_checkpoint(
            config, spec, BenchmarkRunner.TuneOptions(
                dpus = [2], elements_per_dpu = [64], warmup = 0, iterations = 1,
                checkpoints = directory, resume = true))
        @test resumed.cache[BenchmarkRunner.config_key(build)].objective == 4.5
        @test isfile(joinpath(directory, "elementwise.csv"))
        @test length(BenchmarkRunner.tuning_source_fingerprint(config, spec)) == 64
        checkpoint_text = read(checkpoint.path, String)
        @test occursin("\n[signature.search]\n", checkpoint_text)
        @test !occursin(r"(?m)^[ \t]+\[", checkpoint_text)

        mismatched = BenchmarkRunner.TuneOptions(
            dpus = [4], elements_per_dpu = [64], warmup = 0, iterations = 1,
            checkpoints = directory, resume = true)
        @test_throws ErrorException BenchmarkRunner.load_checkpoint(
            config, spec, mismatched)
    end

    mktempdir() do directory
        options = BenchmarkRunner.TuneOptions(
            dpus = [2], elements_per_dpu = [64], warmup = 0, iterations = 1,
            checkpoints = directory)
        spec = config.benchmarks["elementwise"][1]
        build = copy(BenchmarkRunner.DEFAULT_FUSION_BUILD)
        checkpoint = BenchmarkRunner.load_checkpoint(config, spec, options)
        failed = BenchmarkRunner.TuneResult(
            "runtime_failed", Inf, Float64[], Tuple{Int,Int}[])
        @test BenchmarkRunner.record!(checkpoint, build, failed, 1,
                                      "FUSION_LOOKAHEAD", 128)
        resumed = BenchmarkRunner.load_checkpoint(config, spec, options)
        @test isempty(resumed.cache)
        @test length(resumed.trials) == 1
    end

    mktempdir() do directory
        path = joinpath(directory, "elementwise.toml")
        open(path, "w") do io
            TOML.print(io, Dict(
                "version" => 1, "benchmark" => "elementwise",
                "complete" => false, "trials" => [],
                "signature" => Dict("stale" => true)))
        end
        options = BenchmarkRunner.TuneOptions(
            dpus = [2], elements_per_dpu = [64], warmup = 0, iterations = 1,
            checkpoints = directory)
        refreshed = BenchmarkRunner.load_checkpoint(config, spec, options)
        @test isempty(refreshed.trials)
        @test refreshed.signature ==
              BenchmarkRunner.checkpoint_signature(config, spec, options)
    end

    mktempdir() do directory
        profiles = joinpath(directory, "profiles")
        checkpoints = joinpath(directory, "checkpoints")
        mkpath(profiles)
        mkpath(checkpoints)
        for path in (joinpath(profiles, "elementwise.toml"),
                     joinpath(checkpoints, "elementwise.toml"),
                     joinpath(checkpoints, "elementwise.csv"),
                     joinpath(checkpoints, "runs.csv"))
            write(path, "saved")
        end
        options = BenchmarkRunner.TuneOptions(
            profiles = profiles, checkpoints = checkpoints, reset = true)
        BenchmarkRunner.reset_tuning(["elementwise"], options)
        @test !isfile(joinpath(profiles, "elementwise.toml"))
        @test !isfile(joinpath(checkpoints, "elementwise.toml"))
        @test !isfile(joinpath(checkpoints, "elementwise.csv"))
        @test isfile(joinpath(checkpoints, "runs.csv"))
    end
end

@testset "runner resume" begin
    mktempdir() do directory
        path = joinpath(directory, "runner.toml")
        state = BenchmarkRunner.run_state(BenchmarkRunner.Options(state = path))
        push!(state.completed, "case-one")
        BenchmarkRunner.save_state(state)
        resumed = BenchmarkRunner.run_state(
            BenchmarkRunner.Options(state = path, resume = true))
        @test resumed.completed == Set(["case-one"])
    end
end

@testset "run manifests" begin
    mktempdir() do directory
        path = joinpath(directory, "Manifest.toml")
        first = BenchmarkRunner.begin_manifest(
            path, "benchmark", Dict("arguments" => ["elementwise"]))
        BenchmarkRunner.finish_manifest(first, "complete")
        second = BenchmarkRunner.begin_manifest(
            path, "benchmark", Dict("arguments" => ["hist"]))
        BenchmarkRunner.finish_manifest(second, "failed"; failure = "boom")

        document = TOML.parsefile(path)
        @test document["version"] == 1
        @test document["kind"] == "benchmark"
        @test getindex.(document["invocations"], "status") ==
              ["complete", "failed"]
        @test document["invocations"][2]["error"] == "boom"
        @test haskey(document["invocations"][1], "started_at")
        @test haskey(document["invocations"][1], "finished_at")
        @test !occursin(r"(?m)^[ \t]+\[", read(path, String))
    end

    config = load_config()
    options = BenchmarkRunner.Options(
        benchmarks = ["elementwise"], variants = ["baseline"], dpus = [2],
        elements_per_dpu = [64], warmup = 0, iterations = 1,
        use_profiles = false)
    entry = BenchmarkRunner.runner_manifest_entry(
        config, ["elementwise"], ["baseline"], options, ["elementwise"])
    @test length(entry["cases"]) == 1
    @test entry["cases"][1]["total_elements"] == 128
    @test entry["cases"][1]["variants"] == ["baseline"]
end
