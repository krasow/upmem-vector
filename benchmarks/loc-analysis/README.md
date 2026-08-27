# Focused LOC analysis

How many lines each programming model needs to express the same benchmark, as
a measure of programmer effort.

```bash
python3 analyze_model_loc.py                       # all 8 benchmarks
python3 analyze_model_loc.py --benchmarks red knn  # a subset
```

Results are written to `../results/loc-analysis/`. The scc counts below need
`scripts/install_scc.sh` (also run by `scripts/install_dependencies.sh`).

## Results

| Benchmark | Julia | PolymerPIM | SimplePIM | Baseline |
| --- | ---: | ---: | ---: | ---: |
| elementwise | 7 | 10 | 27 | 62 |
| hist | 6 | 7 | 28 | 51 |
| red | 5 | 5 | 27 | 45 |
| linreg | 14 | 23 | 38 | 58 |
| knn | 8 | 9 | 31 | 48 |
| kmeans | 27 | 31 | 47 | 75 |
| vector_search | 7 | 8 | 42 | 73 |
| multitask_classifier | 33 | 44 | 94 | 136 |
| **Total** | **107** | **137** | **334** | **548** |
| *Mean reduction vs SimplePIM* | *70.3%* | *62.3%* | -- | -- |
| *Mean reduction vs baseline* | *81.9%* | *77.0%* | *40.3%* | -- |

## Method

`refs/<benchmark>/<variant>.<ext>.ref` holds a hand-curated excerpt of each
implementation, tracking the sources under `../main-benchmarks`. Logical LOC
excludes blank lines and comments. An excerpt is everything the programmer must
write for that model:

| Model | Covered by the excerpt |
| --- | --- |
| `baseline` | host transfers, launch, host-side merge, **and the DPU kernel** |
| `simplepim` | host orchestration **and** the `map_to_val` / `combine` / `init` / `start` callbacks |
| `polymerpim` | host-level expression only |
| `julia` | host-level expression only |

Excluded is what every model pays equally: parameter plumbing, timing, DPU
allocation and teardown, input synthesis, reference loading, verification, and
shared headers. Function signatures and closing delimiters (`}`, `end`) count --
for SimplePIM and the baseline, a callback with an exact signature is part of
what the model demands -- while cleanup one language does implicitly and another
explicitly (a C++ destructor versus Julia's `release!`) counts for nobody.

## scc cross-check

Every run also counts with [boyter/scc](https://github.com/boyter/scc), found
at `opt/scc/bin/scc` or on `PATH`, so the headline numbers do not rest on a
counter shipped in this repo.

scc's SLOC matches the built-in count on all 32 files -- any divergence is
printed -- and it adds three metrics a line count misses:

| Total, all 8 benchmarks | Julia | PolymerPIM | SimplePIM | Baseline |
| --- | ---: | ---: | ---: | ---: |
| Cyclomatic complexity | 20 | 21 | 40 | 70 |
| Cognitive complexity | 47 | 49 | 99 | 215 |
| ULOC (distinct lines) | 98 | 121 | 298 | 491 |

Mean ULOC reduction is 68.6% (Julia) and 61.4% (PolymerPIM) vs SimplePIM, 81.1%
and 76.7% vs baseline, so the advantage is not just repeated boilerplate.
Per-file counts: `report.md` and the `scc_*` columns of `summary.csv`.
