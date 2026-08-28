@testset "configuration and CLI" begin
    config = load_config()
    @test "elementwise-interpreter" in config.benchmark_names
    @test only(config.benchmarks["elementwise-interpreter"]).variants == ["baseline"]
    @test config.defaults.ntrials == 5
    @test config.defaults.benchmark_root == "main-benchmarks"
    @test Set(keys(config.variants)) ==
          Set(["baseline", "cpu", "julia", "polymerpim", "simplepim",
               "simplepim-patched", "polymerpim-jit", "polymerpim-pipeline",
               "polymerpim-eager", "polymerpim-hybrid"])
    @test BenchmarkRunner.Options().profiles ==
          joinpath(BENCHMARKS, "results", "fusion", "profiles")
    @test BenchmarkRunner.TuneOptions().profiles ==
          joinpath(BENCHMARKS, "results", "fusion", "profiles")
    @test BenchmarkRunner.Options().timeout == 1800
    @test BenchmarkRunner.TuneOptions().timeout == 1800

    options = parse_args([
        "elementwise", "--variant", "julia,baseline", "--dpus", "2,4",
        "--elements-per-dpu", "64", "--warmup", "0", "--iterations", "1",
        "--ntrials", "3", "--timeout", "7", "--build-timeout", "9",
        "--check", "--skip-setup", "--verbose", "--reset", "--csv",
        "/tmp/benchmark-times.csv",
    ])
    @test options.benchmarks == ["elementwise"]
    @test options.variants == ["julia", "baseline"]
    @test options.dpus == [2, 4]
    @test options.elements_per_dpu == [64]
    @test (options.warmup, options.iterations, options.ntrials) == (0, 1, 3)
    @test (options.timeout, options.build_timeout) == (7, 9)
    @test options.check && options.skip_setup && options.verbose && options.reset
    @test options.csv == "/tmp/benchmark-times.csv"

    # Each suite TOML owns its own results, so a second suite cannot append to
    # the main runs.csv.  An explicit --state still wins.
    modes = joinpath(BENCHMARKS, "main-benchmarks", "polymerpim-modes.toml")
    @test BenchmarkRunner.results_csv(parse_args(String[])) ==
          joinpath(BENCHMARKS, "results", "runs.csv")
    @test BenchmarkRunner.results_csv(parse_args(["--config", modes])) ==
          joinpath(BENCHMARKS, "results", "polymerpim-modes", "runs.csv")
    @test BenchmarkRunner.results_csv(
              parse_args(["--config", modes, "--state", "/tmp/pp/state.toml"])) ==
          joinpath("/tmp/pp", "runs.csv")

    @test_throws ErrorException parse_args(["--ntrail", "3"])
    output = IOBuffer()
    process = run(pipeline(
        ignorestatus(`bash $(joinpath(BENCHMARKS, "run.sh")) elementwise --defualt-params`),
        stdout = output, stderr = output))
    @test process.exitcode == 2
    @test occursin("unknown shared option: --defualt-params",
                   String(take!(output)))

    modes = load_config(joinpath(
        BENCHMARKS, "main-benchmarks", "polymerpim-modes.toml"))
    @test modes.benchmark_names == ["elementwise", "knn", "linreg"]
    @test modes.defaults.variants ==
          ["polymerpim-jit", "polymerpim-pipeline", "polymerpim-eager"]
    @test modes.defaults.group_by_variant
    @test all(modes.variants[name].timing_label == "polymerpim"
              for name in modes.defaults.variants)
    @test length(unique(BenchmarkRunner.setup_key(
        modes, modes.variants[name], nothing)
        for name in modes.defaults.variants)) == 3

    dynamic = load_config(joinpath(
        BENCHMARKS, "dynamic-benchmarks", "benchmark.toml"))
    @test dynamic.benchmark_names == ["adaptive_image", "dynamic_query"]
    @test dynamic.defaults.warmup == 0
    @test dynamic.defaults.benchmark_root == "dynamic-benchmarks"
    @test "polymerpim-hybrid" in dynamic.defaults.variants
    @test "simplepim" ∉ only(dynamic.benchmarks["adaptive_image"]).variants
    @test "simplepim" ∈ only(dynamic.benchmarks["dynamic_query"]).variants
    @test BenchmarkRunner.selected_variants(
        only(dynamic.benchmarks["dynamic_query"]), String[]) ==
          ["polymerpim-jit", "polymerpim-hybrid", "polymerpim-pipeline",
           "simplepim"]
    @test BenchmarkRunner.selected_variants(
        only(dynamic.benchmarks["dynamic_query"]),
        ["simplepim", "polymerpim-jit", "polymerpim-eager"]) ==
          ["simplepim", "polymerpim-jit"]
end
