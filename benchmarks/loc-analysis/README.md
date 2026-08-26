# Focused LOC analysis

How much code each programming model needs per benchmark. Migrated from
`benchmark-upmem/analyze_model_loc.py`, with Julia added as a fourth model and
`vector_search` / `multitask_classifier` added to its original six.

```bash
python3 analyze_model_loc.py                       # all 8 benchmarks
python3 analyze_model_loc.py --benchmarks red knn  # a subset
```

Writes `summary.csv`, `file_breakdown.csv`, `overall_metrics.csv`,
`overall_metrics_legacy_six.csv`, and `report.md` to `../results/loc-analysis/`.

## Results

Logical LOC, lower is better. Regenerate with the command above.

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

Restricted to the original six benchmarks: Julia 69.0% / 81.5% and PolymerPIM
60.7% / 76.5% against SimplePIM / baseline.

## What is counted

`refs/<benchmark>/<variant>.<ext>.ref` is a hand-curated excerpt of that
variant's implementation, kept in sync with `../main-benchmarks`. Logical LOC
excludes blank lines and comments (`//` and `/* */` for C/C++; `#`, nestable
`#= =#`, and triple-quoted docstrings for Julia).

An excerpt keeps everything the programmer must write for that model:

| Model | Covered by the excerpt |
| --- | --- |
| `baseline` | host transfers, launch, host-side merge, **and the DPU kernel** |
| `simplepim` | host orchestration **and** the `map_to_val` / `combine` / `init` / `start` callbacks |
| `polymerpim` | host-level expression only |
| `julia` | host-level expression only |

Excluded because all four pay it equally: parameter plumbing, timing and stage
instrumentation, DPU set allocation and teardown, host input synthesis and
reference loading, and result verification. Headers shared verbatim across
variants (`vector_search_common.h`, `multitask_classifier_common.h`) are not
charged to anyone.

## Curation rules

- **Function signatures count.** Defining `map_to_val_func` or `dpu_main` with
  an exact signature is the burden SimplePIM and the baseline impose; dropping
  signatures would erase what the comparison measures.
- **Closing delimiters count.** Excluding bare `}` / `end` removes 14--21% of
  every variant's lines and moves each reduction by at most 1.6 points, leaving
  the ordering unchanged.
- **Scope-exit cleanup is excluded.** C++ frees a `DPUVector` by destructor;
  Julia calls `release!` explicitly because its GC cannot see DPU memory
  pressure. Dropped to match the implicit C++ side; counting them adds about
  two lines each to Julia's `red`, `elementwise`, and `multitask_classifier`.
- **Excerpts track current sources.** The `polymerpim` excerpts were rewritten
  against today's API; the originals predated the `DPUVector` rename and still
  used `dpu_jit_foreach` lambdas where the current API has
  `local_hist[buckets] += 1` and `argmin(distances)`.

## Earlier figures

Prose accompanying the original analysis quoted 41.2% vs SimplePIM and 69.7% vs
baseline; that script against its own checked-in refs gives 52.8% / 71.7%. Both
predate the API work above. `overall_metrics_legacy_six.csv` restricts the
current excerpts to the same six benchmarks for a like-for-like comparison.
