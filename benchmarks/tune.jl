#!/usr/bin/env julia

include(joinpath(@__DIR__, "core", "BenchmarkRunner.jl"))

exit(BenchmarkRunner.tune_main(ARGS))
