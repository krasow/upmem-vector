# PolymerPIM.jl

Julia bindings for vectordpu: a 1-D `Int32` vector living in UPMEM DPU memory.

Requires the C++ library built with `PIPELINE=1 JIT=1`; the wrapper refuses to
compile against anything else.

## Build

`make install` in the source tree builds the wrapper too, against the prefix it
just installed:

```sh
source /usr/upmem_env.sh
cd .. && make install PIPELINE=1 JIT=1 BACKEND=hw   # BACKEND=simulator if HW is down
cd julia && make test
```

It is skipped, with a message, under any other configuration or without `julia`
on `PATH`. To build against a prefix elsewhere, pass `VECTORDPU_DIR` (defaults to
the source tree's `../vectordpu`).

`lib/wrapper/` holds the wrapper source and its cmake tree; the finished library
is installed to `lib/wrapper/PolymerPIM/`, next to the two stamps recording what
it is:

```sh
make config          # both stamps
```

| file | contents |
| --- | --- |
| `build.config` | the flags `libvectordpu` was compiled with, copied from the prefix |
| `install.config` | where that prefix came from (git rev, date, host) and how the wrapper was configured against it |

`build.config` is not just a record. The wrapper reaches its `libvectordpu`
through `RUNPATH`, so a later `make install` with different flags would swap the
library out from under it; the package compares this stamp against what the
loaded library reports for itself and refuses a mismatch instead of silently
running the wrong configuration. From Julia:

```julia
PolymerPIM.versioninfo()   # all of it, printed
installinfo()              # provenance, as recorded at wrapper build time
configuration()            # what the loaded libvectordpu says it is -- ground truth
ndpus(), ntasklets()
```

`versioninfo` is not exported, since `InteractiveUtils` exports one too; call it
qualified.

DPUs are claimed on the first allocation, so `NR_DPUS` has to be set before it
(default 8); `ndpus()` reports the count either way. Tasklets per DPU are fixed
at library build time by `NR_TASKLETS`.

```sh
NR_DPUS=32 julia --project=. yourscript.jl
```

## Usage

```julia
using PolymerPIM

a = DpuVector(Int32[1, 2, 3, 4])
b = DpuVector(fill(Int32(10), 4))

Array(a + b)          # elementwise; also - * div, and vector/scalar forms
Array(a >> 1)         # arithmetic shift by a scalar
Array(abs(-a))
Array(a < b)          # comparisons give a 1/0 mask; also > >= <= ==
Array(select_op(a < b, a, b))

sum(a)                # reductions: sum, prod, minimum, maximum
add!(acc, b)          # in-place: add! sub! mul! div! shr!
```

Operations are queued, not run on the spot. `Array(v)` reads back and blocks,
`fence(v)` waits without transferring, `sync()` drains everything.

## Broadcasting

The expression tree stays lazy and lowers to a single RPN program, so a whole
broadcast is one kernel pass:

```julia
r = a .+ b .* c
d .= abs.(a .- b) .+ 1          # writes through d's buffer
m = ifelse.(a .> b, a, b)       # per-lane select
```

Supported: `+ - * div >> == < > <= >=`, `-`, `abs`, and three-argument
`ifelse`. Anything else (`sqrt`, `sin`, …) raises rather than falling back to a
host loop.

Only within one expression — `d = d .+ 1` in a loop is still a pass per
statement.

## Inspecting the generated kernel

A broadcast becomes one RPN program and one JIT-compiled C kernel.
`@code_jitted` shows that kernel without compiling or launching anything:

```julia
julia> @code_jitted a .+ b .* c
JIT kernel k_9a0cf153d43a1c54 -- 5 opcodes, 2 operands, 16 elements
  build/jit/k_9a0cf153d43a1c54.c (not compiled yet)

#include <barrier.h>
...
int k_9a0cf153d43a1c54(void) {
...
```

The source is generated on the spot, so nothing exists on disk yet -- the path is
where the kernel *will* land, and `iscompiled(code)` says whether it has. The
first launch of a program writes `build/jit/k_<hash>.c` and `.o`, plus one
`main_<n>.c` per compiled batch holding the launch args, tasklet barrier and WRAM
workspace its kernels share. `<hash>` is the cache key, so the same expression
later reuses that object instead of regenerating it. Once written, the file is
byte-identical to what `@code_jitted` printed.

The result is a `JittedCode` -- `.source`, `.ops`, `.hash`, and `.path` are all
readable. `a + b` and `sum(a)` run on statically compiled kernels and have no
generated source.

## Reductions

A reduction returns a future, not a number. Reading one forces it; leave several
unread and they share one kernel pass:

```julia
f = sum(a); g = sum(b)          # queued, nothing read
f[], g[]                        # one pass for both

totals = [sum(v) for v in vectors]   # 8 vectors, 1 pass
[t[] for t in totals]
```

`f[]`, `get(f)` and `fetch(f)` are the same read; `prod`, `minimum` and
`maximum` behave the same.

Reducing a lazy expression folds the terminal into the same pass:

```julia
sum(abs, a)                     # 1 pass -- traced into the program
mapreduce(abs, +, a)            # same, op in (+, *, min, max)
sum(abs.(a))                    # 1 pass
sum(a .* b)                     # 1 pass, two vectors
```

`sum.(a .+ b)` is not a reduction at all -- `sum.` broadcasts `sum` over each
element, which is identity on integers -- so it raises rather than lowering.

Integer scalars in a lazy broadcast are launch parameters automatically. Their
values are captured when the expression is written but stay out of the opcode
stream, so changing a scalar reuses the compiled kernel:

```julia
distance = abs2.(col .- centroid)   # no wrapper required
```

The raw RPN builders and launch helpers live in `PolymerPIM.Internal`. They are
implementation details rather than part of the supported Julia API.

## K-ary

`argmin.(zip(v1, v2, v3))` / `argmax.(...)` give the winning vector per element,
1-based as Julia's are, in one pass. `findmin_lanes(vectors)` /
`findmax_lanes(vectors)` add the winning value in the same pass, returning the
two columns unzipped -- Julia's `findmin.(zip(...))` would be a vector of tuples,
which a DPU cannot hold. `argmax(v::DpuVector)` is the other axis, over one
vector's elements: two passes, both on the DPUs, and only scalars come back.
`min_squared_distance(cols, query)` is the minimum over rows of the squared
distance to `query`.

## Tests

`test/runtests.jl` drives one file per concern in `test/suites/`: `core`,
`elementwise`, `reductions`, `inplace`, `kary`, `broadcast`, and `internal`.

```sh
julia --project=. test/runtests.jl                       # everything
julia --project=. test/runtests.jl broadcast internal  # substring filter
```

Suite files rely on the driver for `using` and `N`, so run them through it.

## Adding an operation

Ops are named by their opcode. `src/internal/opcodes.jl` is generated by
`tools/generate.py` alongside `common/opcodes.h` and `lib/wrapper/wrapper.cpp`
switches on the same value — one numbering, nothing to keep in step. Do not edit
`src/internal/opcodes.jl`.

If a C++ `dpu_vector` operator already implements it: one `case` in the relevant
`apply_*_op` switch plus a `Base` overload in `src/operations.jl`. If it is
expressible as RPN, prefer `src/expr.jl` — no C++ change at all.
