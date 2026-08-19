// The parts of the runtime that are not arithmetic: dpu_vector copy/move, the
// allocator's reuse of freed MRAM, memory pressure, and shutdown ordering.

#include <iostream>

#include "framework.h"

namespace {

using T = int32_t;

}  // namespace

// --------------------------------------------------------------------------
// copy / move semantics
// --------------------------------------------------------------------------

// KNOWN BUG 8 (README): the copy constructor and copy assignment just share the
// VectorDescRef, so `copy = original` is a second handle on one buffer and the
// `copied` flag is set but never consulted -- there is no copy-on-write.  This
// asserts the value semantics the interface implies; if handle semantics are
// deliberate, delete this test and document it on the class instead.
TEST_XFAIL(lifecycle, copy_is_independent,
           "copy ctor shares the VectorDescRef, so writes through the copy are "
           "visible through the original") {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -50, 50);

  dpu_vector<T> original = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> copy = original;
  copy += (T)1000;

  std::vector<T> original_host = original.to_cpu();
  std::vector<T> copy_host = copy.to_cpu();

  std::vector<T> expected_copy(n);
  for (size_t i = 0; i < n; ++i) expected_copy[i] = a[i] + 1000;

  CHECK_VEC_EQ(original_host, a);
  CHECK_VEC_EQ(copy_host, expected_copy);
}

TEST(lifecycle, move_preserves_contents) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -50, 50);

  dpu_vector<T> original = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> moved = std::move(original);

  CHECK_VEC_EQ(moved.to_cpu(), a);
  CHECK_EQ(moved.size(), n);
}

// Descriptors are reference counted, so a result outlives its operands.
TEST(lifecycle, result_outlives_its_operands) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -50, 50);
  std::vector<T> b = tf::random_vector<T>(n, -50, 50);

  dpu_vector<T> result;
  {
    dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
    dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
    result = da + db;
  }

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] + b[i];
  CHECK_VEC_EQ(result.to_cpu(), expected);
}

// Same, but with the op still queued: the event holds the only references left.
TEST(lifecycle, pending_result_after_operands_destroyed) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 3);
  std::vector<T> b = tf::constant_vector<T>(n, 4);

  dpu_vector<T> result;
  {
    dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
    dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
    result = da * db;
    // Deliberately no fence: the multiply is still queued here.
  }
  CHECK_VEC_EQ(result.to_cpu(), tf::constant_vector<T>(n, 12));
}

// --------------------------------------------------------------------------
// allocator reuse
// --------------------------------------------------------------------------

// With a leak this runs out of memory long before the last round.
TEST(lifecycle, repeated_alloc_free_reuses_memory) {
  const size_t n = 256 * 1024;
  std::vector<T> host = tf::constant_vector<T>(n, 2);

  for (size_t round = 0; round < 12; ++round) {
    dpu_vector<T> a = dpu_vector<T>::from_cpu(host);
    dpu_vector<T> b = dpu_vector<T>::from_cpu(host);
    std::vector<T> result = (a + b).to_cpu();
    if (result[0] != 4 || result[n - 1] != 4) {
      tf::fail("round " + tf::str(round) + ": got " + tf::str(result[0]) +
               " expected 4");
      return;
    }
  }
}

// Many small vectors at once: the free list rather than the bump allocator.
TEST(lifecycle, many_small_vectors) {
  const size_t count = 64;
  const size_t n = 1024;
  std::vector<T> host = tf::constant_vector<T>(n, 1);

  std::vector<dpu_vector<T>> vecs;
  vecs.reserve(count);
  for (size_t i = 0; i < count; ++i)
    vecs.push_back(dpu_vector<T>::from_cpu(host));
  tf::drain();

  // Free every other one, then allocate into the holes.
  for (size_t i = 0; i < count; i += 2) vecs[i] = dpu_vector<T>();
  std::vector<dpu_vector<T>> more;
  for (size_t i = 0; i < count / 2; ++i)
    more.push_back(dpu_vector<T>::from_cpu(host));
  tf::drain();

  CHECK_VEC_EQ(more.back().to_cpu(), host);
  CHECK_VEC_EQ(vecs[1].to_cpu(), host);
}

// --------------------------------------------------------------------------
// memory pressure
// --------------------------------------------------------------------------

// `a + b + c + a + b` allocates an intermediate per operator without fusion;
// with fusion it must fit in far less memory, which is the pipeline's original
// motivation.  Sized from the DPU count so it scales with --dpus.
TEST(lifecycle, chained_ops_under_memory_pressure) {
  const size_t num_dpus = DpuRuntime::get().num_dpus();
  // Per-DPU MRAM is 64 MB; stay inside it while still needing the intermediates
  // elided.
  const size_t mb_per_dpu = tf::pipeline_enabled() ? 8 : 3;
  const size_t n = (mb_per_dpu * 1024 * 1024 * num_dpus) / sizeof(T);

  std::vector<T> host = tf::constant_vector<T>(n, 1);
  try {
    dpu_vector<T> a = dpu_vector<T>::from_cpu(host, "a");
    dpu_vector<T> b = dpu_vector<T>::from_cpu(host, "b");
    dpu_vector<T> c = dpu_vector<T>::from_cpu(host, "c");

    dpu_vector<T> result = a + b + c + a + b;
    std::vector<T> got = result.to_cpu();
    CHECK_EQ(got[0], (T)5);
    CHECK_EQ(got[n - 1], (T)5);
  } catch (const std::exception& error) {
    tf::fail(std::string("threw: ") + error.what());
  }
}

// --------------------------------------------------------------------------
// runtime shutdown
// --------------------------------------------------------------------------

// KNOWN BUG 6 (README): shutdown() resets logger_ and allocator_, but
// ~VectorDesc still calls both, so any dpu_vector still in scope at shutdown --
// the natural shape for main() -- crashes the process after all the work
// succeeded.  Only meaningful as the last thing a process does, so it is not
// run here; reproduce with:
//
//   DpuRuntime::get().init(8);
//   auto v = dpu_vector<int>::from_cpu(host);
//   DpuRuntime::get().shutdown();
//   return 0;                      // <-- segfault in ~VectorDesc
//
// Fix: no-op ~VectorDesc once the runtime is down, or keep those two alive.
TEST_KNOWN_FATAL(
    lifecycle, destruct_after_shutdown,
    "~VectorDesc uses the logger and allocator that shutdown() has "
    "already destroyed; see the comment above this test") {
  SKIP("would tear down the test process");
}
