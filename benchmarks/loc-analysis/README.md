# Focused LOC analysis

How many lines each programming model needs to express the same benchmark, as
a measure of programmer effort.

```bash
python3 analyze_model_loc.py                       # all 8 benchmarks
python3 analyze_model_loc.py --benchmarks red knn  # a subset
```

Results are written to `../results/loc-analysis/`.

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
excludes blank lines and comments.

An excerpt contains everything the programmer must write for that model:

| Model | Covered by the excerpt |
| --- | --- |
| `baseline` | host transfers, launch, host-side merge, **and the DPU kernel** |
| `simplepim` | host orchestration **and** the `map_to_val` / `combine` / `init` / `start` callbacks |
| `polymerpim` | host-level expression only |
| `julia` | host-level expression only |

It excludes what every model pays equally: parameter plumbing, timing
instrumentation, DPU allocation and teardown, input synthesis and reference
loading, result verification, and headers shared verbatim across variants.

Function signatures and closing delimiters (`}`, `end`) are counted; for
SimplePIM and the baseline, having to define a callback with an exact signature
is part of what the model demands. Cleanup that one language does implicitly
and another explicitly -- a C++ destructor versus Julia's `release!` -- is
counted for neither.
