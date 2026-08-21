# Benchmarks

[`benchmark.toml`](benchmark.toml) defines each workload's sizes, parameters,
and variants. Implementations and build commands live in `variants/`.

`make` and `run.sh` install SimplePIM and Perfetto into `../opt/` on first use.

## Run

Tune first, then run the suite:

```bash
./run.sh
./run.sh elementwise --resume
./run.sh elementwise --reset-tune
./run.sh elementwise --tune --passes 2 --runner --variant polymerpim,julia
```

Arguments before `--tune` or `--runner` apply to both phases. Arguments after a
marker apply only to that phase. Set `JULIA` to choose the Julia executable.

Run either phase directly when needed:

```bash
julia tune.jl elementwise --dpus 256 --passes 2 --check
julia runner.jl elementwise --variant julia --dpus 2 \
  --elements-per-dpu 4096 --warmup 1 --iterations 1 --check
julia runner.jl --list
julia test/runtests.jl
```

Tuning resumes automatically and skips completed profiles. Use `--reset-tune`
to retune the selected benchmarks. The runner prints and applies tuned values
by default; use `--no-profile` to disable them. Tuning output is concise by
default; pass `--verbose` after `--tune` for complete subprocess output.
By default tuning uses the smallest configured problem size; pass
`--elements-per-dpu` to tune against one or more explicit sizes.

## Outputs

- `fusion/<benchmark>.toml`: best fusion parameters
- `results/Manifest.toml`: expanded invocations and status
- `results/runs.csv`: run status, parameters, and total timings
- `results/runs.sections.csv`: printed stage timings in long form
- `results/fusion/`: tuning manifest, measurements, trials, and resume state

Benchmark commands run under `/usr/bin/time`. Generated parameters, binaries,
reference data, profiles, and results are untracked.

Build, timeout, runtime, timing, and verification failures have distinct CSV
statuses. Only successful runs enter checkpoints; `--keep-going` records later
cases after a failure.

The UPMEM SDK is loaded from `$UPMEM_ENV` or `/usr/upmem_env.sh`. `.localenv`
points `SIMPLE_PIM_LIB` at the installed `opt/SimplePIM` checkout.
