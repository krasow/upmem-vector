# Focused LOC analysis

How many lines each programming model needs to express the same benchmark, as
a measure of programmer effort.

```bash
python3 analyze_model_loc.py                       # all 7 benchmarks
python3 analyze_model_loc.py --benchmarks red knn  # a subset
```

Results are written to `../results/loc-analysis/`. The scc counts below need
`scripts/install_scc.sh` (also run by `scripts/install_dependencies.sh`).

## Results

| Benchmark | Julia | PolymerPIM | SimplePIM | Baseline |
| --- | ---: | ---: | ---: | ---: |
| elementwise | 7 | 10 | 12 | 62 |
| hist | 6 | 7 | 20 | 51 |
| red | 5 | 5 | 20 | 45 |
| linreg | 14 | 23 | 44 | 58 |
| knn | 8 | 9 | 37 | 48 |
| kmeans | 27 | 31 | 53 | 75 |
| vector_search | 7 | 8 | 42 | 73 |
| **Total** | **74** | **93** | **228** | **412** |
| *Mean reduction vs SimplePIM* | *66.5%* | *57.5%* | -- | -- |
| *Mean reduction vs baseline* | *82.8%* | *78.3%* | *45.1%* | -- |

## Method

`refs/<benchmark>/<variant>.<ext>.ref` holds a hand-curated excerpt of each
implementation, tracking the sources under `../main-benchmarks`. Logical LOC
excludes blank lines and comments. An excerpt is everything the programmer must
write for that model:

| Model | Covered by the excerpt |
| --- | --- |
| `baseline` | host transfers, launch, host-side merge, **and the DPU kernel** |
| `simplepim` | host orchestration **and** every callback in the benchmark's `*_funcs/` folder (`start` / `map_to_val` or `map` / `init` / `combine`) |
| `polymerpim` | host-level expression only |
| `julia` | host-level expression only |

**Excluded:** what every model pays equally -- parameter plumbing, timing, DPU
allocation and teardown, input synthesis, reference loading, verification,
shared headers -- plus harness workarounds like SimplePIM's
`rescatter_to_existing`, which exists only because our round trip re-uploads
every iteration.

**Counted:** signatures and closing delimiters (`}`, `end`); an exact callback
signature is part of what a model demands. Cleanup that one language does
implicitly and another explicitly (a C++ destructor versus Julia's `release!`)
counts for nobody.

## scc cross-check

Every run also counts with [boyter/scc](https://github.com/boyter/scc), found
at `opt/scc/bin/scc` or on `PATH`, so the headline numbers do not rest on a
counter shipped in this repo.

scc's SLOC matches the built-in count on all 28 files -- any divergence is
printed -- and it adds three metrics a line count misses:

| Total, all 7 benchmarks | Julia | PolymerPIM | SimplePIM | Baseline |
| --- | ---: | ---: | ---: | ---: |
| Cyclomatic complexity | 11 | 11 | 24 | 47 |
| Cognitive complexity | 25 | 25 | 50 | 130 |
| ULOC (distinct lines) | 71 | 88 | 202 | 375 |

Mean ULOC reduction is 63.1% (Julia) and 54.9% (PolymerPIM) vs SimplePIM, 81.7%
and 77.4% vs baseline, so the advantage is not just repeated boilerplate.
Per-file counts: `report.md` and the `scc_*` columns of `summary.csv`.
