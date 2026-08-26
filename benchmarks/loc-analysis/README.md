# Focused LOC analysis

```bash
python3 analyze_model_loc.py                       # all 8 benchmarks
python3 analyze_model_loc.py --benchmarks red knn  # a subset
```

Writes `summary.csv`, `file_breakdown.csv`, `overall_metrics.csv`,
`overall_metrics_legacy_six.csv`, and `report.md` to `../results/loc-analysis/`.

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

Over the original six benchmarks only: Julia 69.0% / 81.5%, PolymerPIM
60.7% / 76.5% against SimplePIM / baseline.

## What is counted

`refs/<benchmark>/<variant>.<ext>.ref` is a hand-curated excerpt of that
variant's implementation, kept in sync with `../main-benchmarks`. Logical LOC
excludes blank lines and comments.

| Model | Covered by the excerpt |
| --- | --- |
| `baseline` | host transfers, launch, host-side merge, **and the DPU kernel** |
| `simplepim` | host orchestration **and** the `map_to_val` / `combine` / `init` / `start` callbacks |
| `polymerpim` | host-level expression only |
| `julia` | host-level expression only |

Excluded because all four pay it equally: parameter plumbing, timing
instrumentation, DPU set allocation and teardown, host input synthesis and
reference loading, result verification, and headers shared verbatim across
variants.

## Curation rules

- **Function signatures count.** Defining `map_to_val_func` or `dpu_main` with
  an exact signature is the burden SimplePIM and the baseline impose.
- **Closing delimiters count.** Excluding bare `}` / `end` removes 14--21% of
  every variant's lines and moves each reduction by at most 1.6 points.
- **Scope-exit cleanup is excluded.** C++ frees a `DPUVector` by destructor;
  Julia's explicit `release!` is dropped to match.
- **Excerpts track current sources.** The `polymerpim` excerpts were rewritten
  against today's API, which the originals predated.
