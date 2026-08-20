# UpmemVector.jl

Julia bindings for vectordpu: a 1-D `Int32` vector living in UPMEM DPU memory.

## Build

```sh
source /usr/upmem_env.sh

# 1. Install the C++ library (BACKEND=simulator if the DPU hardware is down)
cd .. && make install PIPELINE=1 JIT=1 BACKEND=hw

# 2. Build the wrapper and run the tests
cd julia
make test VECTORDPU_DIR=$(cd ../../vectordpu && pwd)
```

`VECTORDPU_DIR` must point at the install prefix (the main Makefile's `DESTDIR`,
`../vectordpu` by default). `make` here needs `julia` on `PATH`.

## Usage

```julia
using UpmemVector

a = DpuVector(Int32[1, 2, 3, 4])
b = DpuVector(fill(Int32(10), 4))

Array(a + b)          # elementwise; also - * div, and vector/scalar forms
Array(a >> 1)         # arithmetic shift by a scalar
Array(abs(-a))
Array(a < b)          # comparison mask (1/0)
Array(a == 2)         # equality against a scalar
Array(select_op(a < b, a, b))   # elementwise ifelse

sum(a)                # reductions: sum, prod, minimum, maximum

a .+ b                # broadcasting works and forwards to the same kernels
abs.(.-a)

add!(acc, b)          # in-place: add! sub! mul! div! shr!, no intermediate
```

Operations are queued, not executed on the spot. `Array(v)` reads a result back
and blocks; `fence(v)` waits without transferring; `sync()` drains everything.

## Reductions and fusion

The runtime merges independent work into one DPU kernel pass. Reading a
reduction immediately prevents that, so reducing many vectors has a lazy form:

```julia
totals = sums(vectors)                # queues all, then reads: 1 kernel pass

f = lazy_sum(a)                       # or hold futures yourself
g = lazy_sum(b)
get(f), get(g)
```

Measured over 8 vectors of 1024 elements:

| | kernel passes | horizontal fusions |
| --- | --- | --- |
| `[sum(v) for v in vs]` | 8 | 0 |
| `sums(vs)` | **1** | 7 |

`lazy_prod`, `lazy_minimum` and `lazy_maximum` exist too. The runtime counters
behind that table are exposed for tests: `UpmemVector.stat_compute_launches()`,
`stat_horizontal_fusions()`, `stat_vertical_fusions()`.

## Adding an operation

`lib/wrapper/wrapper.cpp` holds one table per op category, each entry a call to
the public C++ operator. Adding an op means one line in the table plus one enum
entry in `src/operations.jl` — the indices must stay in step.
