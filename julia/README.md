# UpmemVector.jl

Julia bindings for vectordpu: a 1-D `Int32` vector living in UPMEM DPU memory.

## Requirements

The library must be built with `PIPELINE=1 JIT=1`. The `PIPELINE=0` and `JIT=0`
configurations exist so the C++ side can measure the alternatives; this package
binds the op set that only exists in the full one. The wrapper refuses to
compile otherwise, and the module re-checks on every load.

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

a .+ b                # broadcasts fuse into one kernel; see Broadcasting
abs.(.-a)

add!(acc, b)          # in-place: add! sub! mul! div! shr!, no intermediate
```

Operations are queued, not executed on the spot. `Array(v)` reads a result back
and blocks; `fence(v)` waits without transferring; `sync()` drains everything.

## Broadcasting

A broadcast is kept lazy and the whole expression tree is lowered to a single
RPN program at materialise time, so it is one kernel pass by construction — no
host-side intermediates, and nothing depending on the fusion pass or its
lookahead window.

```julia
r = a .+ b .* c                 # one program, one pass
d .= abs.(a .- b) .+ 1          # writes through d's buffer, one pass
m = ifelse.(a .> b, a, b)       # ifelse lowers to a per-lane select
```

Measured over 512 elements, against the eager operator spelling of the same
expression:

| | kernel passes |
| --- | --- |
| `a + b * c` / `a .+ b .* c` | 1 / **1** |
| `abs(a - b) + 1` / `abs.(a .- b) .+ 1` | 3 / **1** |
| eight-way `+` / eight-way `.+` | 7 / **1** |

The lazy path records zero fusions in every case: there is nothing to fuse,
because only one event is ever submitted.

Reachable inside a broadcast: `+ - * div >> == < > <= >=`, `-` and `abs`, and
three-argument `ifelse`. An integer operand becomes an immediate rather than a
pushed value. Anything else (`sqrt`, `sin`, …) raises rather than silently
falling back to a host loop.

`dest .= expr` writes through `dest`'s existing buffer. That matters because
`DpuVector` is a handle type — rebinding it would leave other handles looking at
the old buffer. The destination may appear in its own expression
(`c .= c .+ 100`).

Fusion across *statements* is a different matter: `d = d .+ 1` in a loop is one
pass per statement, since each assignment materialises. Write the chain as one
expression to get one pass.

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

## Expressions

Anything beyond a single operator is built as an RPN program and submitted in
one fused kernel. `transform` returns a vector, `reduce_expr` a future. The
callback receives a `Vector{DpuExpr}`: `x[1]` is the vector it was launched on,
`x[2:end]` the extra operands.

```julia
d = transform(a, b) do x            # elementwise, one pass
    abs(x[1] - x[2])
end

dot = reduce_expr(a, b) do x        # dot product, one pass
    sum(x[1] * x[2])
end
get(dot)
```

Available inside a program: `+ - * div >> == < > <= >=` (against another
expression or an integer), `- abs sqr dup select`, the leaves `input()`,
`operand(i)`, `constant(v)`, `scalar_var(i)`, `lane_index()`, the terminals
`sum prod minimum maximum`, and `argmin_lanes` / `argmax_lanes`.

`scalar_var(i)` reads `scalars[i]` at launch instead of baking the value into
the program, so changing it reuses the same compiled kernel:

```julia
transform(a; scalars = Int32[10]) do x
    add_var(x[1], 1)                # a .+ 10, same kernel as any other scalar
end
```

`dpu_pipeline` and `dpu_pipeline_reduce` take a program directly if you would
rather build one yourself. (They are not called `pipeline` because that name is
taken by `Base.pipeline`.)

These go through `pipeline()` rather than the C++ `transform()`/`reduce()`
templates, which cannot take a Julia closure.

Reductions built this way still fuse. Five independent dot products over 512
elements, left unread: **1 kernel pass, 4 horizontal fusions**.

## Comparisons

`>`, `>=`, `<=` and elementwise `==` are routed through RPN, since the opcodes
exist but no C++ `dpu_vector` operator wraps them. They yield 1/0 per lane.

## argmin / argmax

`argmin_of(vectors)` / `argmax_of(vectors)` give the 0-based index of the
winning vector per element, in one pass. `argmin_lanes` / `argmax_lanes` are the
same thing spelled inside a larger expression.

`min_squared_distance(cols, query)` is the minimum over rows of the squared
distance to `query`. `vectordpu.h` declared this but never defined it, so it is
built from the expression API here.

## Build limits

`MAX_VFUSE_INPUTS` and `MAX_PIPELINE_SCALARS` are read from the library rather
than hardcoded, so they track the configuration it was compiled with.

## Adding an operation

Ops are named by their opcode. `src/opcodes.jl` is generated by
`tools/generate.py` alongside `common/opcodes.h`, and `lib/wrapper/wrapper.cpp`
switches on the same value, so there is one numbering and nothing to keep in
step by hand. Do not edit `src/opcodes.jl`.

Adding an op that a C++ `dpu_vector` operator already implements means one
`case` in the relevant `apply_*_op` switch plus one `Base` overload in
`src/operations.jl`. For anything expressible as RPN, prefer `src/expr.jl`
instead — no C++ change at all.
