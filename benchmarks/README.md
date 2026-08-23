# Benchmarks

[`benchmark.toml`](benchmark.toml) defines each workload's sizes, parameters,
and variants. Implementations and build commands live in `variants/`.
Pass `--config` to select another suite.

`make` and `run.sh` install SimplePIM and Perfetto into `../opt/` on first use.

## Run

Tune first, then run the suite:

```bash
./run.sh
./run.sh elementwise --resume
./run.sh elementwise --reset --default-params
./run.sh elementwise --reset-tune
./run.sh elementwise --default-params
./run.sh elementwise --tune --passes 2 --runner --variant polymerpim,julia
./run.sh --config polymerpim-modes.toml --default-params
```

Arguments before `--tune` or `--runner` apply to both phases. Arguments after a
marker apply only to that phase. Set `JULIA` to choose the Julia executable.
`--resume` resumes tuning first, then retries unfinished benchmark trials.
`--default-params` skips tuning and ignores saved fusion profiles.
`--reset` removes the selected benchmarks from run CSVs and checkpoints;
`--reset-tune` discards their tuning checkpoints and profiles.

[`polymerpim-modes.toml`](polymerpim-modes.toml) compares JIT, interpreted
pipeline, and eager PolymerPIM on elementwise, k-NN, and linear regression.
The three variants reuse the same benchmark sources and appear separately in
the CSV. `--default-params` keeps the comparison on Makefile defaults.

`ntrials` in `benchmark.toml` launches each benchmark process independently;
`iterations` remains the workload's in-process loop count. Override trials with
`--ntrials`.

Run either phase directly when needed:

```bash
julia --project=. tune.jl elementwise --dpus 256 --passes 2 --check
julia --project=. runner.jl elementwise --variant julia --dpus 2 \
  --elements-per-dpu 4096 --warmup 1 --iterations 1 --check
julia --project=. runner.jl --list
julia --project=. test/runtests.jl
```

Tuning resumes automatically and skips completed profiles. Use `--reset-tune`
to retune the selected benchmarks. The runner prints and applies tuned values
by default; use `--no-profile` to disable them. Tuning output is concise by
default, as is the benchmark runner; pass `--verbose` for complete subprocess
output.
By default tuning uses the smallest configured problem size; pass
`--elements-per-dpu` to tune against one or more explicit sizes.

## Outputs

- `results/Manifest.toml`: resolved benchmark dimensions and run status
- `results/runs.csv`: run status, parameters, and total timings
- `results/runs.sections.csv`: printed stage timings in long form
- `results/fusion/profiles/<benchmark>.toml`: best fusion parameters
- `results/fusion/`: tuning manifest, measurements, trials, and resume state

Benchmark commands run under `/usr/bin/time`. Generated parameters, binaries,
reference data, profiles, and results are untracked.

Build, timeout, runtime, timing, and verification failures have distinct CSV
statuses. Failures do not stop the suite. Only successes enter the checkpoint,
so `--resume` retries missing trials without repeating completed ones.

The UPMEM SDK is loaded from `$UPMEM_ENV` or `/usr/upmem_env.sh`. `.localenv`
points `SIMPLE_PIM_LIB` at the installed `opt/SimplePIM` checkout.
