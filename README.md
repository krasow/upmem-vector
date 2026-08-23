# PolymerPIM

please run to commit stuff

```bash
bash format.sh
```

to run the test suite, enable the upmem env
```
make test
```

`make` installs the repository-local dependencies on first use.

The self-contained benchmark suite is documented in
[`benchmarks/README.md`](benchmarks/README.md).

## C++ API

Include `<polymerpim.h>`. Expressions are lazy, and host scalars become runtime
parameters automatically:

```cpp
using namespace polymerpim;

DPUVector<int32_t> x(host_x);
auto distance = sqr(x - centroid);
auto nearest = minimum(distance);
sync();
```

Local reductions use indexed updates. They remain pending until `sync()` or
`to_cpu()`:

```cpp
DPULocalVector<int32_t> bins(256);
bins[index] += value;
auto result = bins.to_cpu();
```

The C++ API is installed under `include/polymerpim`.
