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

1. ~~**A shared intermediate with 2+ consumers returns zeros to all but the
   first.**~~ **Fixed.** The erase decision was made at submit time, when later
   consumers did not exist yet, and guessed at live callers from
   `shared_ptr::use_count()` with a `+ 5` fudge — unreliable because
   `absorbed_inputs` on *other* descriptors also hold references. Now:
   * `VectorDesc::handle_count` counts live `dpu_vector` handles exactly, so
     "no caller can build another op on this vector" is a fact, not a guess;
   * `expand_absorbed_inputs` only *marks* the producer
     (`Event::output_was_inlined`) instead of erasing it, and
     `EventQueue::output_still_needed` decides at dispatch — by which point
     every consumer in the batch is queued and temporaries have died.

   Cost: a producer is only dropped once the vector's last handle has gone, so
   reading a result inside the same full-expression that built it keeps the
   temporary alive and costs one extra pass. `sum(a + b).get()` is 2 passes;
   `auto f = sum(a + b); f.get();` is 1. See
   `vfuse.reduction_terminates_the_chain` and
   `vfuse.reduction_of_expression_is_one_kernel`, which pin both.


2. ~~**`to_cpu()` corrupts data — and sometimes the heap — unless every per-DPU
   shard is 8-byte aligned.**~~ **Fixed**, and it was three faults at once:
   * `vec_xfer_from_dpu` advanced the host pointer by `size_bytes` per DPU but
     pushed `align8(size_bytes)` to every DPU. A transfer is one
     `dpu_push_xfer`, which applies a single size *and* a single MRAM offset to
     the whole set, so ragged shards both over-ran their host slot and let the
     per-DPU addresses drift apart. `allocated_bytes` is now uniform across
     DPUs, and `to_cpu` reads into a padded staging buffer and compacts.
   * the eager and lazy allocation paths computed the layout independently, so a
     vector and a result of the same shape could disagree.
   * a one-element shard was silently widened to two, which is what made
     `to_cpu` return an oversized vector (bug 5).

   Verified across 1…65537 elements at 8 and 64 DPUs. `tf::safe_elements()` is
   gone and the suite's default element count is now a ragged prime (4099).
   — `sharding` suite


3. ~~**A binary op over two fresh intermediates loses one operand.**~~
   **Fixed.** `try_vfuse` recorded its whole merged program as the "recipe" for
   one output. Once the event is horizontally fused that program spans *every*
   chain, separated by `OP_NEXT_CHAIN`, so splicing it into a consumer yields
   nonsense — `(a+b)*c - (d-a)` returned `d-a` because the subtraction inlined a
   two-chain program as if it were a single value. The recipe is now recorded
   only for single-chain events. —
   `vfuse.binary_op_over_two_intermediates`,
   `vfuse.fused_matches_fenced_evaluation`


4. ~~**Chained in-place compound assignment double-applies earlier ops.**~~
   **Fixed**, and it was two bugs behind one symptom:
   * the recorded `absorbed_rpn` was self-referential for an in-place op, so a
     consumer inlining it re-read a buffer the producer had overwritten
     (`a=40; a+=10; a-=3` → 57). `EventQueue::enqueue` now skips the
     registration when the output aliases an input.
   * a fused event could take a dependency on an event it had just absorbed,
     which never completes — that was the five-in-a-row deadlock (event 2
     waiting on event 3, which had been merged into event 2).
     `detail::adopt_fused_event` now drops dependencies inside
     `[last->id, last->max_id]`, the range the fused event stands for.
   — `elementwise.compound_chain_*`, all passing

5. ~~**`to_cpu()` can return more elements than the vector holds.**~~ **Fixed**
   with bug 2 — the result length now comes from the sum of the shard payloads,
   and the one-element-shard widening that caused it is gone. —
   `elementwise.to_cpu_size_matches_vector_size`,
   `reductions.single_element_min_max`


6. ~~**Destroying a `dpu_vector` after `shutdown()` segfaults.**~~ **Fixed:**
   `~VectorDesc` now returns early when the runtime is down, since the logger
   and allocator it would use are gone and the DPU set is already freed. This
   crashed the Julia bindings on every exit, because CxxWrap finalizers run
   after `atexit`. Covered by the Julia suite;
   `lifecycle.destruct_after_shutdown` records why the C++ runner cannot host
   it (it drains the queue after every test).

7. ~~**Absorbed inlining ignores `MAX_VFUSE_OPS`.**~~ **Fixed.** That limit is
   the size of the interpreter's `args.pipeline.ops` buffer, so it only binds
   when there is no JIT — a generated kernel carries its program in C and can be
   any length. `expand_absorbed_inputs` now stops inlining before overflowing it
   under `JIT=0`, and `internal_launch_universal_pipeline` throws instead of
   silently truncating. — `vfuse.chain_far_beyond_max_ops_is_correct`


8. ~~**`dpu_vector`'s copy is shallow.**~~ **By design** — it is a handle type,
   confirmed with the author. Copying MRAM on every assignment would be a
   silent, expensive device transfer; `to_cpu()` is how you take a snapshot.
   Documented on the class and pinned by `lifecycle.copy_shares_the_buffer`.

9. ~~**Every explicit `pipeline()` call ran kernel 0 (binary add) under
   `JIT=0`.**~~ **Fixed.** `launch_compute` dispatches `e->pipeline_kid` on the
   interpreter path, but `launch_universal_pipeline` only set `e->kid`, leaving
   `pipeline_kid` at its default 0 — kernel id 0 is `binary_int32_t_add`. The
   interpreter never ran, so results were whatever stale MRAM had been recycled
   into the result buffer. `launch_binary` passes `pipeline_kid` explicitly,
   which is why operator-built expressions worked and only the explicit RPN API
   broke. Fixed by setting both ids in `launch_universal_pipeline`. — the
   `pipeline.rpn_*`, `expr_builder_*` and `reduce_*` tests under `JIT=0`

10. ~~**Ragged shards lost their last lane in every static DPU kernel.**~~
    **Fixed.** MRAM DMA only honours transfer lengths that are multiples of 8
    bytes and rounds a ragged tail *down*, so a shard holding an odd number of
    4-byte elements silently dropped its final element — both on the read and
    the write. `binary.inl`, `unary.inl`, `reduce.inl` and `pipeline.inl` all
    passed `block_elems * sizeof(TYPE)` straight to `mram_read`/`mram_write`.
    The JIT codegen already rounded up via `b_b_aligned`, which is why `JIT=1`
    was unaffected. Shards are `align8` padded by the allocator, so rounding the
    transfer up stays inside the allocation. — `sharding.unaligned_sizes_are_correct`,
    `sharding.sizes_around_one_block_are_correct`, `reductions.sum_at_all_sizes`

Bugs 9 and 10 were found by running the suite under `PIPELINE=1 JIT=0`, which
had never been exercised end-to-end — the tests that catch them already existed
and passed under `JIT=1`. **Run all three configurations**; a green `JIT=1` run
says nothing about the interpreter path.

Note that `MAX_PIPELINE_STACK_DEPTH` is 2 and cannot practically be raised —
`.data.stacks` overflows WRAM at 8 — so the interpreter genuinely cannot run
deep RPN programs. That is a resource limit, not a bug: `JIT=1` bakes the
program into C and has no stack buffer.


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
