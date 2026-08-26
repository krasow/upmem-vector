# Benchmarks

Benchmark suites are TOML files. Workloads live in `main-benchmarks/` and
`dynamic-benchmarks/`; backend definitions live in `backends/`. Dependencies
are installed into `../opt/` on first use.

## Run

`run.sh` tunes fusion parameters, then runs the selected benchmarks:

```bash
./run.sh
./run.sh elementwise --resume
./run.sh elementwise --reset --default-params
./run.sh --config main-benchmarks/polymerpim-modes.toml --default-params
./run.sh --config dynamic-benchmarks/benchmark.toml --default-params
```

- `--resume` continues tuning and retries unfinished trials.
- `--reset` discards saved runs for the selection.
- `--reset-tune` discards its tuning profile.
- `--default-params` skips tuning and ignores profiles.
- `--verbose` prints subprocess output.

Use `./run.sh --help` for runner options and `--help-all` for advanced options.
Arguments after `--tune` or `--runner` apply only to that phase.

`ntrials` launches independent processes; `iterations` controls each process's
workload loop.

## Suites

- `main-benchmarks/benchmark.toml`: standard benchmark suite.
- `main-benchmarks/polymerpim-modes.toml`: JIT, Pipeline, and Eager comparison.
- `dynamic-benchmarks/benchmark.toml`: cold-start `adaptive_image` and
  `dynamic_query` workloads without warmup.

Dynamic workloads keep independent results:

```bash
./dynamic-benchmarks/adaptive_image.sh --reset
./dynamic-benchmarks/query_sweep.sh --reset
```

Both write their CSV, sections, checkpoint, summary, and plot under
`results/dynamic/`.

## Direct commands

```bash
julia --project=. tune.jl elementwise --dpus 256 --passes 2 --check
julia --project=. runner.jl elementwise --variant julia --check
julia --project=. test/runtests.jl
```

## Results

- `results/Manifest.toml`: resolved invocation and status.
- `results/runs.csv`: run status and total timings.
- `results/runs.sections.csv`: parsed timed sections.
- `results/fusion/`: tuning profiles, measurements, and checkpoints.

Runs use `/usr/bin/time`. Failures are recorded and do not stop the suite;
`--resume` skips successes and retries missing trials. Generated results,
parameters, binaries, and reference data are untracked.

Generate the completed-benchmark weak-scaling grids and their averaged CSV:

```bash
python3 plot_weak_scaling.py
```

Generate one end-to-end runtime-decomposition grid per DPU count:

```bash
python3 plot_runtime_decomposition.py
```

The PDFs are written under `results/runtime-decomposition/`. Each stacked bar
averages the completed trials. Solid segments contain setup and
measured-iteration work, hatched segments contain warm-up work, and the
unmeasured segment is the residual between all instrumented stages and
`/usr/bin/time` wall time. Error bars show the wall-time standard deviation.

The UPMEM SDK is loaded from `$UPMEM_ENV` or `/usr/upmem_env.sh`. Set `JULIA` to
select the Julia executable.
