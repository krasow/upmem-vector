# vectordpu test suite

One binary, `test/vectordpu_test`, holding every suite. Select subsets at run
time.

```sh
source /usr/upmem_env.sh
make PIPELINE=1 JIT=1 test
make PIPELINE=1 JIT=1 test TEST_ARGS="--filter=hfuse --stats"
make PIPELINE=1 JIT=1 list-tests
```

Fusion parameters must be passed to `make` directly — `build.config` is
regenerated on every build, so editing it does nothing.

Formatting is handled by `.pre-commit-config.yaml` (`pre-commit install` once
per clone; `./format.sh` for the clang-format pass alone). pre-commit 4.x needs
git >= 2.31 for `--all-files`; the SDK box ships git 2.20, so pin
`pre-commit==3.2.0` there.

## Options

| option | meaning |
| --- | --- |
| `--list` / `--filter=SUBSTR` / `--exact=suite.name` | select tests |
| `--isolate` | one process per test (**default** for `make test`) |
| `--elements=N` / `--dpus=N` / `--seed=N` | defaults: safe-for-N-DPUs, 64, 12345 |
| `--stats` | print the counter delta for every test |
| `--timeout=SEC` | abort a wedged test (default 300, 0 disables) |
| `--run-known-fatal` | also run tests that crash or hang |
| `--fail-fast` / `-v` | |

`--isolate` re-executes the binary once per test. It costs a DPU allocation per
test but survives a crashing or deadlocking test and — more importantly — stops
a failing test from changing later results. Several open bugs do exactly that:
**the hfuse suite fails in a shared process and passes entirely in isolation**,
so non-isolated numbers are not trustworthy while they are open.

## Suites

| suite | covers |
| --- | --- |
| `elementwise` | every operator vs a CPU reference; aliasing; compound assignment |
| `reductions` | sum/product/min/max, lazy futures, identity and size edge cases |
| `vfuse` | vertical fusion: chain depth, operand budget, fusion barriers |
| `hfuse` | horizontal fusion: chains per pass, reduction fan-out, linreg/histogram shapes |
| `pipeline` | hand-written RPN, `pipeline_reduce`, `dpu_pipeline_expr` |
| `jit` | `jit()`, kernel caching, signature collisions, `transform`/`reduce`, argmin/argmax |
| `sharding` | per-DPU shard layout and host readback |
| `lifecycle` | copy/move, allocator reuse, memory pressure, shutdown ordering |

All tests use `int32_t`: `opinfo.h` is generated with an `OpInfo<int32_t>`
specialisation only and there are no float kernels, so `dpu_vector<float>` does
not link.

## Asserting on fusion

A fusion rule that stops firing still returns the right answer, just with more
kernel passes — so the fusion suites assert on pass count too. `host/stats.h`
adds always-on runtime counters; `tf::measure` fences, runs a region, fences
again, and returns the delta:

```cpp
StatsSnapshot k = tf::measure([&] {
  dpu_vector<T> res = ((a + b) - c) * d;
  actual = res.to_cpu();
});
CHECK_VEC_EQ(actual, expected);
CHECK_KERNELS_EQ(k, 1);                          // 4 operators, 1 kernel pass
```

Expectations are computed from the build parameters, so they survive a sweep:

```cpp
size_t per_pass = tf::max_reduction_chains();
CHECK_KERNELS_EQ(k, tf::ceil_div(count, per_pass));
```

Also: `tf::max_hfuse_chains()`, `tf::max_combined_inputs()`,
`tf::max_vfuse_ops()`, `tf::fusion_lookahead()`. With `PIPELINE=0` the
`CHECK_KERNELS_*` macros become no-ops so the value checks still run.

## Markers

```cpp
TEST(suite, name)                             { ... }
TEST_XFAIL(suite, name, "why it fails")       { ... }   // known-broken runtime
TEST_KNOWN_FATAL(suite, name, "how it dies")  { ... }   // crashes or hangs
```

`TEST_XFAIL` is a *correct* test the runtime currently fails: it reports `XFAIL`
and does not fail the run, but reports `XPASS` and **does** fail once it starts
passing, so markers cannot rot. `TEST_KNOWN_FATAL` is skipped unless isolated.
The `*_IF_FUSED` variants apply only when `PIPELINE=1`.

## Known bugs found by this suite

All open as of writing; each has a test that documents it.

1. **A shared intermediate with 2+ consumers returns zeros to all but the
   first.** `expand_absorbed_inputs` inlines the producer into the first
   consumer and erases it, so MRAM is never written; later consumers were not
   yet submitted when the `other_consumers` scan ran, and the `external_holder`
   guard (`use_count() > internal_refs + 5`) is too slack to fire.
   `shared = a*b; left = shared+c; right = shared-d` → `right` reads 0. With 8
   consumers (histogram) it deadlocks instead. A fence after the producer avoids
   it. — `vfuse.diamond_dependency_is_correct`,
   `elementwise.shared_intermediate_two_consumers`,
   `hfuse.histogram_shape_counts_are_correct`

2. **`to_cpu()` corrupts data — and sometimes the heap — unless every per-DPU
   shard is 8-byte aligned.** `vec_xfer_from_dpu` advances the host pointer by
   `size_bytes` per DPU but transfers `align8(size_bytes)` to every DPU. For
   `int32_t`, `n` must be a multiple of `2 * num_dpus`; at 8 DPUs n=4099 loses
   2561/4099 lanes and n=15/17 abort in glibc. Fix: transfer each DPU's own
   `allocated_bytes` and pad the host buffer per shard. `tf::safe_elements()`
   exists to dodge this and should go once it is fixed. — `sharding` suite

3. **A binary op over two fresh intermediates loses one operand.**
   `(a+b)*c - (d-a)` returns `d-a`. — `vfuse.binary_op_over_two_intermediates`

4. **Chained in-place compound assignment double-applies earlier ops.** For an
   in-place op the recorded `absorbed_rpn` is self-referential, so the consumer
   re-reads a buffer the producer already overwrote: `a=40; a+=10; a-=3` → 57.
   Five in a row deadlock. Fix: in `EventQueue::submit`, skip the `absorbed_rpn`
   registration when `e->output` aliases an input. —
   `elementwise.compound_chain_*`

5. **`to_cpu()` can return more elements than the vector holds** — a vector of
   exactly `num_dpus` int32 reads back with `2 * num_dpus`, which also makes
   `max()` of a 1-element vector return 0. —
   `elementwise.to_cpu_size_matches_vector_size`, `reductions.single_element_min_max`

6. **Destroying a `dpu_vector` after `shutdown()` segfaults** — `~VectorDesc`
   uses the logger and allocator that `shutdown()` already reset. —
   `lifecycle.destruct_after_shutdown`

7. **Absorbed inlining ignores `MAX_VFUSE_OPS`.** `try_vfuse`/`try_hfuse` bail
   over budget; `expand_absorbed_inputs` does not. Harmless under `JIT=1`, but
   the interpreter path clamps with `num_ops = min(ops.size(), MAX_VFUSE_OPS)`
   and silently drops the tail (16 of 80 `+1` steps at `MAX_VFUSE_OPS=128`). —
   `vfuse.chain_far_beyond_max_ops_is_correct`

8. **`dpu_vector`'s copy is shallow** — `a = b` aliases the same MRAM. May be
   intentional; if so, document it and consider `share()`/`clone()`. —
   `lifecycle.copy_is_independent`

## Adding a test

Drop a file in `test/` — the Makefile globs `test/*.cc` and tests self-register.

```cpp
#include "framework.h"

TEST(mysuite, does_the_thing) {
  const size_t n = tf::elements();
  std::vector<int32_t> a = tf::random_vector<int32_t>(n, -50, 50);
  dpu_vector<int32_t> da = dpu_vector<int32_t>::from_cpu(a);

  StatsSnapshot k = tf::measure([&] { /* ... */ });
  CHECK_VEC_EQ(actual, expected);
  CHECK_KERNELS_EQ(k, 1);
}
```

Checks: `CHECK`, `CHECK_EQ`/`NE`/`LT`/`LE`/`GT`/`GE`, `CHECK_NEAR`,
`CHECK_VEC_EQ`, `CHECK_VEC_NEAR`, `CHECK_KERNELS_EQ`/`LE`/`GE`/`LT`,
`CHECK_FUSIONS_GE`, `SKIP("reason")`.

A marker can be config-dependent when the runtime is only broken one way:

```cpp
#if JIT
TEST(vfuse, some_case) {
#else
TEST_XFAIL(vfuse, some_case, "why the non-JIT path fails") {
#endif
```

Verified configurations: `PIPELINE=1 JIT=1`, `PIPELINE=1 JIT=0`, `PIPELINE=0`.
