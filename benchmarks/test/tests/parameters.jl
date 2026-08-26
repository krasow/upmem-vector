@testset "parameter generation" begin
    c_source = joinpath(
        BENCHMARKS, "main-benchmarks", "baseline", "elementwise", "Param.h")
    c_params = generated_parameters(
        c_source; elements = 128, dpus = 2, warmup = 0, iterations = 1,
        check = true, load_ref = true, seed = 7, fixed = Dict{String,Any}(),
        operation = "a + b")
    @test occursin("const uint32_t dpu_number = 2;", c_params)
    @test occursin("uint64_t nr_elements = 128;", c_params)
    @test occursin("#define OPERATION(a, b) a + b", c_params)

    julia_source = joinpath(
        BENCHMARKS, "main-benchmarks", "julia", "elementwise", "Param.jl")
    julia_params = generated_parameters(
        julia_source; elements = 128, dpus = 2, warmup = 0, iterations = 1,
        check = false, load_ref = true, seed = 7, fixed = Dict{String,Any}(),
        operation = "a + b")
    @test occursin("const N = 128", julia_params)
    @test occursin("const check_correctness = 0", julia_params)
    @test occursin("const load_ref = 1", julia_params)

    no_ref = generated_parameters(
        julia_source; elements = 128, dpus = 2, warmup = 0, iterations = 1,
        check = false, load_ref = false, seed = 7, fixed = Dict{String,Any}(),
        operation = "a + b")
    @test occursin("const load_ref = 0", no_ref)
    @test occursin("operation(a, b) = @. a + b", julia_params)

    @test BenchmarkRunner.format_parameters(
        Dict{String,Any}("k" => 10, "dim" => 4)) == "dim=4;k=10"
end
