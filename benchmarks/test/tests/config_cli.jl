@testset "configuration and CLI" begin
    config = load_config()
    @test "elementwise-interpreter" in config.benchmark_names
    @test only(config.benchmarks["elementwise-interpreter"]).variants == ["baseline"]
    @test config.defaults.ntrials == 5
    @test Set(keys(config.variants)) ==
          Set(["baseline", "cpu", "julia", "polymerpim", "simplepim"])
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

    @test_throws ErrorException parse_args(["--ntrail", "3"])
    output = IOBuffer()
    process = run(pipeline(
        ignorestatus(`bash $(joinpath(BENCHMARKS, "run.sh")) elementwise --defualt-params`),
        stdout = output, stderr = output))
    @test process.exitcode == 2
    @test occursin("unknown shared option: --defualt-params",
                   String(take!(output)))
end
