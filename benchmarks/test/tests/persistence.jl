@testset "TOML formatting" begin
    output = IOBuffer()
    BenchmarkRunner.write_toml(output, Dict(
        "version" => 1,
        "signature" => Dict("search" => Dict("A" => [1, 2])),
        "trials" => [Dict("candidate" => 1)],
    ))
    text = String(take!(output))
    @test !occursin(r"(?m)^[ \t]+\S", text)
    @test !occursin(r"(?m)\S\n\[", text)
    @test TOML.parse(text)["signature"]["search"]["A"] == [1, 2]
end

@testset "runner checkpoints" begin
    mktempdir() do directory
        path = joinpath(directory, "runner.toml")
        case = test_case(operation = "a + b")
        records = [BenchmarkRunner.run_record(case, "baseline", nothing, trial)
                   for trial in 1:2]
        state = BenchmarkRunner.RunState(
            path, Set(BenchmarkRunner.run_key.(records)), records, true)
        BenchmarkRunner.save_state(state)

        resumed = BenchmarkRunner.run_state(
            BenchmarkRunner.Options(state = path, resume = true))

        @test resumed.records == records
        @test resumed.completed == Set(BenchmarkRunner.run_key.(records))
        @test length(resumed.completed) == 2
        raw = TOML.parsefile(path)
        @test raw["version"] == 3
        @test getindex.(raw["completed"], "trial") == [1, 2]
    end
end

@testset "runner reset" begin
    mktempdir() do directory
        state_path = joinpath(directory, "runner.toml")
        csv_path = joinpath(directory, "runs.csv")
        options = BenchmarkRunner.Options(state = state_path, csv = csv_path)
        records = [
            BenchmarkRunner.run_record(
                test_case(benchmark = "elementwise"), "baseline", nothing),
            BenchmarkRunner.run_record(
                test_case(benchmark = "elementwise"), "julia", nothing),
            BenchmarkRunner.run_record(
                test_case(benchmark = "hist",
                          parameters = Dict{String,Any}("bins" => 16)),
                "baseline", nothing),
        ]
        BenchmarkRunner.save_state(BenchmarkRunner.RunState(
            state_path, Set(BenchmarkRunner.run_key.(records)), records, true))
        write(csv_path,
              "timestamp,invocation,benchmark,variant\n" *
              "now,1,elementwise,baseline\n" *
              "now,1,elementwise,julia\nnow,1,hist,baseline\n")
        sections_path = BenchmarkRunner.sections_csv(csv_path)
        write(sections_path,
              "timestamp,invocation,benchmark,variant,section\n" *
              "now,1,elementwise,baseline,kernel\n" *
              "now,1,elementwise,julia,kernel\n" *
              "now,1,hist,baseline,kernel\n")

        captured_output() do
            BenchmarkRunner.reset_runs(["elementwise"], ["baseline"], options)
        end

        @test options.resume
        resumed = BenchmarkRunner.run_state(
            BenchmarkRunner.Options(state = state_path, resume = true))
        @test getindex.(resumed.records, "variant") == ["julia", "baseline"]
        @test getindex.(csv_records(csv_path), "variant") == ["julia", "baseline"]
        @test getindex.(csv_records(sections_path), "variant") ==
              ["julia", "baseline"]
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
        @test document["version"] == 2
        @test document["kind"] == "benchmark"
        @test getindex.(document["invocations"], "status") ==
              ["complete", "failed"]
        @test document["invocations"][2]["error"] == "boom"
        @test all(haskey(invocation, "started_at") &&
                  haskey(invocation, "finished_at")
                  for invocation in document["invocations"])
    end

    config = load_config()
    options = BenchmarkRunner.Options(
        benchmarks = ["elementwise"], variants = ["baseline"], dpus = [2, 4],
        elements_per_dpu = [64, 128], warmup = 0, iterations = 1,
        use_profiles = false)
    entry = BenchmarkRunner.runner_manifest_entry(
        config, ["elementwise"], ["baseline"], options, ["elementwise"])
    benchmark = only(entry["benchmarks"])
    @test benchmark["dpus"] == [2, 4]
    @test benchmark["elements_per_dpu"] == [64, 128]
    @test benchmark["variants"] == ["baseline"]
    @test !haskey(benchmark, "total_elements")
    @test !haskey(benchmark, "parameters")
    @test entry["ntrials"] == 5
    @test !entry["check"]
end
