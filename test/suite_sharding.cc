// Per-DPU sharding and host readback.
//
// A vector is split across DPUs by allocator::materialize_descriptor_layout.
// Each shard's `size_bytes` is its own payload, but `allocated_bytes` is the
// SAME on every DPU, because a host transfer is one dpu_push_xfer and that
// applies a single size *and* a single MRAM offset to the whole set.  Ragged
// payloads are read into a padded staging buffer and compacted by
// dpu_vector::to_cpu.
//
// This used to be wrong three ways at once -- the transfer pushed
// align8(shard) into unpadded host slots, the eager and lazy allocation paths
// disagreed about the layout, and a one-element shard was silently widened to
// two -- which corrupted data, and sometimes the heap, for any size that was
// not a multiple of 2 * num_dpus.  These tests pin the fix.

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

  if (got.size() != n) {
    tf::fail("n=" + tf::str(n) + ": to_cpu returned " + tf::str(got.size()) +
             " elements");
    return 0;
  }
  size_t wrong = 0;
  for (size_t i = 0; i < n; ++i)
    if (got[i] != a[i] + b[i]) wrong++;
  return wrong;
}

// Checks a list of sizes, reporting the first that comes back wrong.
void check_sizes(const std::vector<size_t>& sizes) {
  for (size_t n : sizes) {
    const size_t wrong = wrong_lanes_for_add(n);
    if (wrong != 0) {
      tf::fail("n=" + tf::str(n) + ": " + tf::str(wrong) + " lanes wrong");
      return;
    }
  }
}

}  // namespace

// --------------------------------------------------------------------------
// shard shapes
// --------------------------------------------------------------------------

// Shards that divide evenly and are already 8-byte aligned: the fast path,
// where the transfer goes straight into the caller's buffer.
TEST(sharding, aligned_sizes_are_correct) {
  const size_t granularity = (size_t)DpuRuntime::get().num_dpus() * 2;
  check_sizes({granularity, granularity * 2, granularity * 3, granularity * 8,
               granularity * 64});
}

// Uniform shards with an odd element count, remainder shards of a different
// parity from the rest, and ragged primes -- every case that needs staging.
TEST(sharding, unaligned_sizes_are_correct) {
  const size_t dpus = (size_t)DpuRuntime::get().num_dpus();
  check_sizes({dpus * 3, dpus * 4 + 1, dpus * 12 + 4, 4099, 9973});
}

// Around one WRAM block the old overrun ran off the end of the host allocation
// rather than into a neighbouring lane, so glibc caught it.
TEST(sharding, sizes_around_one_block_are_correct) {
  check_sizes({BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1});
}

// Fewer elements than DPUs, so some shards are empty.
TEST(sharding, fewer_elements_than_dpus) {
  const size_t dpus = (size_t)DpuRuntime::get().num_dpus();
  check_sizes({1, 2, dpus / 2, dpus - 1});
}

// One element per DPU is the narrowest shard that still gets its own slot; it
// used to be widened to two elements behind the caller's back.
TEST(sharding, one_element_per_dpu) {
  check_sizes({(size_t)DpuRuntime::get().num_dpus()});
}

// --------------------------------------------------------------------------
// diagnostic
// --------------------------------------------------------------------------

// Not an assertion: prints the wrong-lane count across a size sweep.
TEST(sharding, report_size_sweep) {
  if (!tf::verbose()) SKIP("run with -v to print the sweep");

  const uint32_t num_dpus = DpuRuntime::get().num_dpus();
  std::cout << "         size sweep at " << num_dpus << " DPUs:\n";
  for (size_t n = 1; n <= num_dpus * 8; n += std::max(1u, num_dpus / 4)) {
    const size_t shard = n / num_dpus;
    std::cout << "           n=" << n << " shard=" << shard
              << " wrong=" << wrong_lanes_for_add(n) << "\n";
  }
}
