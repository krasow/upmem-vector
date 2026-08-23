using Test
using TOML

const BENCHMARKS = normpath(joinpath(@__DIR__, ".."))
pushfirst!(LOAD_PATH, BENCHMARKS)
using BenchmarkRunner

const TESTS = joinpath(@__DIR__, "tests")

include(joinpath(TESTS, "support.jl"))

@testset "benchmark runner" begin
    for file in sort(readdir(TESTS; join = true))
        basename(file) == "support.jl" || include(file)
    end
end
