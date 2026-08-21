module BenchmarkRunner

using TOML
using Dates
using SHA
using Printf

const BENCHMARK_DIR = dirname(@__DIR__)
const REPO_ROOT = dirname(BENCHMARK_DIR)
const VARIANT_DIR = joinpath(BENCHMARK_DIR, "variants")
const DEFAULT_CONFIG = joinpath(BENCHMARK_DIR, "benchmark.toml")
const DEFAULT_ENV = joinpath(BENCHMARK_DIR, "env.sh")

include("config.jl")
include("parameters.jl")
include("manifest.jl")
include("timings.jl")
include("execution.jl")
include("cli.jl")
include("tuning.jl")

export generated_parameters, generated_path, load_config, main, parse_args,
       run_cli, tune_main

function main(args = ARGS)
    try
        run_cli(args)
        return 0
    catch exception
        exception isa InterruptException && rethrow()
        println(stderr, "error: ", sprint(showerror, exception))
        return 1
    end
end

end
