# vectordpu test suite

All suites build into `test/vectordpu_test` and are selected at runtime.

```sh
source /usr/upmem_env.sh
make PIPELINE=1 JIT=1 test
make PIPELINE=1 JIT=1 test TEST_ARGS="--filter=hfuse --stats"
make PIPELINE=1 JIT=1 list-tests
```

Pass fusion settings to `make`; `build.config` is regenerated and should not be
edited. Run all three configurations: `PIPELINE=1 JIT=1`,
`PIPELINE=1 JIT=0`, and `PIPELINE=0`.

## Runner options

| Option | Purpose |
| --- | --- |
| `--list`, `--filter=TEXT`, `--exact=SUITE.NAME` | Select tests |
| `--isolate` | Run each test in a fresh process; default for `make test` |
| `--elements=N`, `--dpus=N`, `--seed=N` | Defaults: 4099, 64, 12345 |
| `--stats` | Print runtime counter deltas |
| `--timeout=SEC` | Per-test timeout; default 300, 0 disables |
| `--run-known-fatal` | Include tests known to crash or hang |
| `--fail-fast`, `-v` | Stop on failure or enable verbose output |

Isolation costs one DPU allocation per test but contains crashes, deadlocks,
and leaked runtime state.

## Suites

| Suite | Coverage |
| --- | --- |
| `elementwise` | Operators, aliasing, compound assignment |
| `reductions` | Reductions, lazy futures, edge cases |
| `vfuse`, `hfuse` | Vertical and horizontal fusion |
| `pipeline` | RPN, `pipeline_reduce`, expression builder |
| `jit` | Code generation, caching, signatures, indirect operations |
| `sharding` | Per-DPU layout and host readback |
| `lifecycle` | Handles, allocation, shutdown |

Tests use `int32_t`; generated operation metadata does not support float tests.

## Writing tests

Files matching `test/*.cc` are discovered automatically. Include `framework.h`
and register tests with one of these markers:

```cpp
TEST(suite, name) { /* expected to pass */ }
TEST_XFAIL(suite, name, "reason") { /* known wrong result */ }
TEST_KNOWN_FATAL(suite, name, "reason") { /* crash or hang */ }
```

`XFAIL` becomes a failing `XPASS` after the bug is fixed. Known-fatal tests run
only with `--run-known-fatal`; the `*_IF_FUSED` variants apply only when
`PIPELINE=1`.

Use `tf::measure` and assert both values and launch counts for fusion tests:

```cpp
StatsSnapshot stats = tf::measure([&] {
  dpu_vector<T> result = ((a + b) - c) * d;
  actual = result.to_cpu();
});
CHECK_VEC_EQ(actual, expected);
CHECK_KERNELS_EQ(stats, 1);
```

Build-dependent expectations can use `tf::max_reduction_chains()`,
`tf::max_hfuse_chains()`, `tf::max_combined_inputs()`,
`tf::max_vfuse_ops()`, and `tf::fusion_lookahead()`. Kernel-count checks are
no-ops when `PIPELINE=0`.

Common assertions include `CHECK`, `CHECK_EQ` and its comparison variants,
`CHECK_NEAR`, `CHECK_VEC_EQ`, `CHECK_VEC_NEAR`, `CHECK_KERNELS_*`,
`CHECK_FUSIONS_GE`, and `SKIP`.

## Limits and formatting

`MAX_PIPELINE_STACK_DEPTH` is 2. Raising it to 8 overflows WRAM; deep RPN
programs require JIT code generation.

Run `./format.sh` before committing. The SDK's older Git requires
`pre-commit==3.2.0` if running hooks over all files.
