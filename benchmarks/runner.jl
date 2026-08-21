#!/usr/bin/env julia

pushfirst!(LOAD_PATH, @__DIR__)
using BenchmarkRunner

exit(BenchmarkRunner.main(ARGS))
