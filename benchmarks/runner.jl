#!/usr/bin/env julia

include(joinpath(@__DIR__, "core", "BenchmarkRunner.jl"))

exit(BenchmarkRunner.main(ARGS))
