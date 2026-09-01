@testset "command failures" begin
    config = load_config()
    failed = quietly() do
        BenchmarkRunner.execute_command(
            config, "exit 7", BENCHMARKS, 1; timeout = 2)
    end
    @test (failed.status, failed.exit_code) == (:failed, 7)

    timed_out = quietly() do
        BenchmarkRunner.execute_command(
            config, "sleep 2", BENCHMARKS, 1; timeout = 1)
    end
    @test (timed_out.status, timed_out.exit_code) == (:timed_out, 124)
end

@testset "variant failure recording" begin
    config = load_config()
    case = test_case(dpus = 1, elements_per_dpu = 1)
    directory = "main-benchmarks/baseline/elementwise"
    scenarios = [
        (
            BenchmarkRunner.VariantSpec(
                "broken", directory, nothing, String[], ["exit 7"], "true"),
            :build_failed, :build, "failed", "7",
        ),
        (
            BenchmarkRunner.VariantSpec(
                "broken", directory, nothing, String[], String[], "true"),
            :timing_missing, :run, "success", "0",
        ),
        (
            BenchmarkRunner.VariantSpec(
                "broken", directory, nothing, String[], String[],
                "printf 'broken (ms): 1.0\\nMismatch at index 0\\n'"),
            :verification_failed, :verify, "success", "0",
        ),
    ]

    for (variant, status, phase, command_status, exit_code) in scenarios
        mktempdir() do directory
            path = joinpath(directory, "runs.csv")
            run_case = status == :verification_failed ?
                       test_case(dpus = 1, elements_per_dpu = 1, check = true) : case
            result = quietly() do
                BenchmarkRunner.run_variant(
                    config, variant, run_case, BenchmarkRunner.Options(csv = path))
            end
            row = only(csv_records(path))
            @test (result.status, result.phase) == (status, phase)
            @test row["status"] == string(status)
            @test row["phase"] == string(phase)
            @test row["command_status"] == command_status
            @test row["exit_code"] == exit_code
        end
    end
end

@testset "dry-run plan" begin
    text = captured_output() do
        run_cli([
            "elementwise-interpreter", "--variant", "baseline",
            "--dpus", "2", "--elements-per-dpu", "64",
            "--warmup", "0", "--iterations", "1", "--check",
            "--skip-setup", "--dry-run",
        ])
    end
    @test occursin("check = true", text)
    @test occursin("main-benchmarks/baseline/elementwise-interpreter", text)
    @test occursin("generate main-benchmarks/baseline/elementwise-interpreter", text)
    @test occursin("total_elements = 128", text)
    @test length(findall("trial ", text)) == 5
    @test length(findall("make clean", text)) == 1
end

@testset "mode suite groups setup" begin
    text = captured_output() do
        run_cli([
            "knn", "--config",
            joinpath(BENCHMARKS, "main-benchmarks", "polymerpim-modes.toml"),
            "--dpus", "2", "--elements-per-dpu", "64,128",
            "--warmup", "0", "--iterations", "1", "--ntrials", "1",
            "--dry-run", "--no-profile",
        ])
    end
    @test length(findall("-- setup", text)) == 3
    @test length(findall("-- polymerpim-jit", text)) == 2
    @test length(findall("-- polymerpim-pipeline", text)) == 2
    @test length(findall("-- polymerpim-eager", text)) == 2
end
