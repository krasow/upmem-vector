// Per-DPU sharding and host readback.
//
// The allocator shards a vector as `size_bytes = elems_per_dpu * sizeof(T)`
// with `allocated_bytes = align8(size_bytes)`, but detail::vec_xfer_from_dpu
// advances the host pointer by `size_bytes` per DPU while transferring
// `desc[0].allocated_bytes` to *every* DPU:
//
//   DPU_FOREACH(...)  dpu_prepare_xfer(dpu, &cpu[element]);
//                     element += desc[idx].size_bytes - reserved_bytes;
//   xfer_size = desc[0].allocated_bytes - reserved_bytes;
//
// KNOWN BUG 2 (README): when a shard is not a whole number of 8-byte words each
// DPU overruns its host slot, clobbering the next lane and, on the last DPU,
// the end of the vector.  For a 4-byte element the readback is only correct
// when n is a multiple of 2 * num_dpus.  Measured at 8 DPUs (a + b, lane by
// lane):
//
//   n=16 ok | n=24 -> 1 wrong | n=33 -> 28 wrong | n=100 -> 49 wrong
//   n=1000 -> 1 wrong | n=4099 -> 2561 wrong | n=9,10,17 -> glibc abort
//   n=15 -> layout-dependent: wrong, aborting, or accidentally correct
//
// Fix: transfer each DPU's own allocated_bytes and pad the host buffer per
// shard, or advance by allocated_bytes and compact afterwards.

#include <iostream>

#include "framework.h"

namespace {

using T = int32_t;

// a + b at an exact element count; returns the number of wrong lanes.
size_t wrong_lanes_for_add(size_t n) {
  std::vector<T> a(n), b(n);
  for (size_t i = 0; i < n; ++i) {
    a[i] = (T)(i + 1);
    b[i] = (T)(i + 100);
  }
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  std::vector<T> got = (da + db).to_cpu();

  size_t wrong = 0;
  for (size_t i = 0; i < n && i < got.size(); ++i)
    if (got[i] != a[i] + b[i]) wrong++;
  return wrong;
}

}  // namespace

// --------------------------------------------------------------------------
// aligned sizes: what the runtime handles correctly today
// --------------------------------------------------------------------------

// Multiples of 2 * num_dpus across three orders of magnitude.
TEST(sharding, aligned_sizes_are_correct) {
  const uint32_t num_dpus = DpuRuntime::get().num_dpus();
  const size_t granularity = (size_t)num_dpus * 2;

  for (size_t multiplier :
       {(size_t)1, (size_t)2, (size_t)3, (size_t)8, (size_t)16, (size_t)64}) {
    const size_t n = granularity * multiplier;
    const size_t wrong = wrong_lanes_for_add(n);
    if (wrong != 0) {
      tf::fail("n=" + tf::str(n) + " (shard " + tf::str(n / num_dpus) +
               " elements): " + tf::str(wrong) + " lanes wrong");
      return;
    }
  }
}

// The rest of the suite relies on this rounding.
TEST(sharding, safe_elements_rounds_up_to_a_correct_size) {
  const uint32_t num_dpus = DpuRuntime::get().num_dpus();
  const size_t granularity = (size_t)num_dpus * 2;

  for (size_t hint : {(size_t)1, (size_t)7, granularity - 1, granularity,
                      granularity + 1, (size_t)4099}) {
    const size_t n = tf::safe_elements(hint);
    CHECK_GE(n, hint);
    CHECK_EQ(n % granularity, (size_t)0);
    CHECK(tf::is_safe_element_count(n));
  }
}

// Special-cased in the allocator (`elems_per_dpu * size_type == 4` bumps to 2),
// so it works despite the nominal 4-byte shard.
TEST(sharding, one_element_per_dpu) {
  const size_t n = DpuRuntime::get().num_dpus();
  CHECK_EQ(wrong_lanes_for_add(n), (size_t)0);
}

TEST(sharding, single_element_vector) {
  CHECK_EQ(wrong_lanes_for_add(1), (size_t)0);
}

// --------------------------------------------------------------------------
// unaligned sizes: documented corruption
// --------------------------------------------------------------------------

// Odd shard, no remainder: only a few lanes wrong, so the easiest variant to
// mistake for a fusion bug.
TEST_XFAIL(sharding, odd_shard_size_corrupts_readback,
           "shard of 3 int32 = 12 bytes, transferred as align8(12)=16") {
  const uint32_t num_dpus = DpuRuntime::get().num_dpus();
  const size_t n = (size_t)num_dpus * 3;  // 3 elements per DPU, no remainder
  const size_t wrong = wrong_lanes_for_add(n);
  if (wrong != 0)
    tf::fail("n=" + tf::str(n) + ": " + tf::str(wrong) + " lanes wrong");
}

// A remainder shard of different parity: offsets diverge part way through, so
// most lanes end up wrong.
TEST_KNOWN_FATAL(sharding, remainder_shard_corrupts_readback,
                 "shards of 4 and 5 int32 mix 16- and 24-byte transfers; the "
                 "overrun trips glibc's heap check at exit") {
  const uint32_t num_dpus = DpuRuntime::get().num_dpus();
  const size_t n = (size_t)num_dpus * 4 + 1;
  const size_t wrong = wrong_lanes_for_add(n);
  if (wrong != 0)
    tf::fail("n=" + tf::str(n) + ": " + tf::str(wrong) + " of " + tf::str(n) +
             " lanes wrong");
}

// A deliberately ragged size -- and the default this suite first shipped with.
TEST_XFAIL(sharding, ragged_prime_size_corrupts_readback,
           "n=4099 loses roughly two thirds of its lanes") {
  const size_t wrong = wrong_lanes_for_add(4099);
  if (wrong != 0)
    tf::fail("n=4099: " + tf::str(wrong) + " of 4099 lanes wrong");
}

// Past one WRAM block the overrun runs off the end of the host allocation
// rather than into a neighbouring lane, so glibc catches it.
//
// BLOCK_SIZE-1 is deliberately not tested: at 8 DPUs it lands on shards of
// 2x7+1 and whether the overrun corrupts a visible lane, aborts, or goes
// unnoticed depends on the surrounding heap layout, so it is flaky rather than
// informative.  The deterministic corruption cases are above.
TEST_KNOWN_FATAL(sharding, size_just_over_one_block_corrupts_the_heap,
                 "n=BLOCK_SIZE+1 aborts in glibc") {
  const size_t wrong = wrong_lanes_for_add(BLOCK_SIZE + 1);
  if (wrong != 0) tf::fail("n=" + tf::str(BLOCK_SIZE + 1) + ": corrupted");
}

// Not an assertion: prints wrong-lane counts across a sweep so the shape of the
// bug is visible in one place.  Sizes known to abort are skipped.
TEST(sharding, report_size_sweep) {
  if (!tf::verbose()) SKIP("run with -v to print the sweep");

  const uint32_t num_dpus = DpuRuntime::get().num_dpus();
  std::cout << "         size sweep at " << num_dpus << " DPUs:\n";
  for (size_t n = num_dpus; n <= num_dpus * 8; n += num_dpus) {
    const size_t shard = n / num_dpus;
    const bool aligned = (shard * sizeof(T)) % 8 == 0;
    if (!aligned && shard > 1 && shard % 2 == 1 && n < 32) continue;
    const size_t wrong = wrong_lanes_for_add(n);
    std::cout << "           n=" << n << " shard=" << shard
              << " aligned=" << (aligned ? "yes" : "no ") << " wrong=" << wrong
              << "\n";
  }
}
