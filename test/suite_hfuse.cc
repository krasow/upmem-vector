// Horizontal fusion: independent chains over equal-length vectors share one
// kernel pass, each writing its own output slot.
//
// How many fit is a function of the build parameters, so expectations are
// computed from the macros rather than hard-coded, and survive a sweep:
//
//   chains per pass  = MAX_HFUSE_CHAINS                  (primary + N-1 extras)
//   reduction chains = MAX_SAFE_HFUSED_REDUCTION_CHAINS
//   distinct inputs  = MAX_COMBINED_INPUTS

#include "framework.h"

namespace {

using T = int32_t;

// Uploads `count` vectors and drains, so a later probe measures compute only.
struct Operands {
  std::vector<std::vector<T>> host;
  std::vector<dpu_vector<T>> dpu;
};

Operands make_operands(size_t count, size_t n, T lo = 0, T hi = 20) {
  Operands out;
  out.host.reserve(count);
  out.dpu.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.host.push_back(tf::random_vector<T>(n, lo, hi));
    out.dpu.push_back(dpu_vector<T>::from_cpu(out.host.back()));
  }
  tf::drain();
  return out;
}

int64_t host_sum(const std::vector<T>& v) {
  int64_t total = 0;
  for (T x : v) total += x;
  return total;
}

// Reduction chains per pass; each also consumes one operand slot.
size_t reduction_chains_per_pass() {
  size_t limit = tf::max_reduction_chains();
  if (tf::max_hfuse_chains() < limit) limit = tf::max_hfuse_chains();
  if (tf::max_combined_inputs() < limit) limit = tf::max_combined_inputs();
  return limit;
}

}  // namespace

// --------------------------------------------------------------------------
// independent elementwise chains
// --------------------------------------------------------------------------

TEST(hfuse, two_independent_adds_share_one_kernel) {
  const size_t n = tf::elements();
  Operands ops = make_operands(4, n);

  std::vector<T> left, right;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> l = ops.dpu[0] + ops.dpu[1];
    dpu_vector<T> r = ops.dpu[2] + ops.dpu[3];
    left = l.to_cpu();
    right = r.to_cpu();
  });

  std::vector<T> expected_left(n), expected_right(n);
  for (size_t i = 0; i < n; ++i) {
    expected_left[i] = ops.host[0][i] + ops.host[1][i];
    expected_right[i] = ops.host[2][i] + ops.host[3][i];
  }
  CHECK_VEC_EQ(left, expected_left);
  CHECK_VEC_EQ(right, expected_right);

  CHECK_KERNELS_EQ(k, 1);
  CHECK_FUSIONS_GE(k, 1u);
}

// One kernel iterates one element count, so different lengths cannot share.
TEST(hfuse, different_lengths_do_not_fuse) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, 0, 20);
  std::vector<T> b = tf::random_vector<T>(n, 0, 20);
  std::vector<T> c = tf::random_vector<T>(n / 2, 0, 20);
  std::vector<T> d = tf::random_vector<T>(n / 2, 0, 20);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  dpu_vector<T> dc = dpu_vector<T>::from_cpu(c);
  dpu_vector<T> dd = dpu_vector<T>::from_cpu(d);
  tf::drain();

  std::vector<T> big, small;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> l = da + db;
    dpu_vector<T> r = dc + dd;
    big = l.to_cpu();
    small = r.to_cpu();
  });

  std::vector<T> expected_big(n), expected_small(n / 2);
  for (size_t i = 0; i < n; ++i) expected_big[i] = a[i] + b[i];
  for (size_t i = 0; i < n / 2; ++i) expected_small[i] = c[i] + d[i];
  CHECK_VEC_EQ(big, expected_big);
  CHECK_VEC_EQ(small, expected_small);
  CHECK_KERNELS_EQ(k, 2);
}

// --------------------------------------------------------------------------
// reduction fan-out: the linear-regression / histogram shape
// --------------------------------------------------------------------------

// Exactly as many reductions as fit in one pass.
TEST(hfuse, reductions_at_chain_limit_share_one_kernel) {
  const size_t count = reduction_chains_per_pass();
  const size_t n = tf::elements();
  Operands ops = make_operands(count, n, 0, 10);

  std::vector<int64_t> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_future_vector<T> futures;
    for (size_t i = 0; i < count; ++i) futures.push_back(sum(ops.dpu[i]));
    for (auto value : futures.get()) actual.push_back((int64_t)value);
  });

  CHECK_EQ(actual.size(), count);
  for (size_t i = 0; i < count && i < actual.size(); ++i) {
    if (actual[i] != host_sum(ops.host[i]))
      tf::fail("reduction " + tf::str(i) + ": got " + tf::str(actual[i]) +
               " want " + tf::str(host_sum(ops.host[i])));
  }
  CHECK_KERNELS_EQ(k, 1);
}

// One past the limit spills into a second pass -- and no further.
TEST(hfuse, reductions_past_chain_limit_spill_to_two_kernels) {
  const size_t per_pass = reduction_chains_per_pass();
  const size_t count = per_pass + 1;
  const size_t n = tf::elements();
  Operands ops = make_operands(count, n, 0, 10);

  std::vector<int64_t> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_future_vector<T> futures;
    for (size_t i = 0; i < count; ++i) futures.push_back(sum(ops.dpu[i]));
    for (auto value : futures.get()) actual.push_back((int64_t)value);
  });

  CHECK_EQ(actual.size(), count);
  for (size_t i = 0; i < count && i < actual.size(); ++i) {
    if (actual[i] != host_sum(ops.host[i]))
      tf::fail("reduction " + tf::str(i) + ": got " + tf::str(actual[i]) +
               " want " + tf::str(host_sum(ops.host[i])));
  }
  CHECK_KERNELS_EQ(k, tf::ceil_div(count, per_pass));
}

// The general property a fusion regression breaks: K reductions cost
// ceil(K / per_pass) passes across a range of K.
TEST(hfuse, reduction_kernel_count_scales_with_chain_limit) {
  const size_t per_pass = reduction_chains_per_pass();
  const size_t n = 1024;

  for (size_t count : {(size_t)1, (size_t)2, per_pass, per_pass + 1,
                       per_pass * 2, per_pass * 2 + 1}) {
    Operands ops = make_operands(count, n, 0, 10);

    std::vector<int64_t> actual;
    StatsSnapshot k = tf::measure([&] {
      dpu_future_vector<T> futures;
      for (size_t i = 0; i < count; ++i) futures.push_back(sum(ops.dpu[i]));
      for (auto value : futures.get()) actual.push_back((int64_t)value);
    });

    bool values_ok = actual.size() == count;
    for (size_t i = 0; values_ok && i < count; ++i)
      values_ok = actual[i] == host_sum(ops.host[i]);
    if (!values_ok) {
      tf::fail("K=" + tf::str(count) + ": wrong reduction values");
      return;
    }

#if PIPELINE
    // Hand-rolled so the message names the failing K.  Nothing to fuse at
    // PIPELINE=0, hence the guard.
    const size_t expected = tf::ceil_div(count, per_pass);
    if (k.compute_launches != expected) {
      tf::fail("K=" + tf::str(count) + ": expected " + tf::str(expected) +
               " kernel pass(es), got " + tf::str(k.compute_launches) + "  [" +
               k.to_string() + "]");
      return;
    }
#else
    (void)k;
    (void)per_pass;
#endif
  }
}

// Different reduction kinds over one input: distinct chains, one operand slot.
TEST(hfuse, mixed_reduction_kinds_over_one_input) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, 1, 20);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  tf::drain();

  int64_t got_sum = 0, got_min = 0, got_max = 0;
  StatsSnapshot k = tf::measure([&] {
    auto s = sum(da);
    auto lo = min(da);
    auto hi = max(da);
    got_sum = (int64_t)s.get();
    got_min = (int64_t)lo.get();
    got_max = (int64_t)hi.get();
  });

  int64_t want_sum = host_sum(a);
  T want_min = a[0], want_max = a[0];
  for (T x : a) {
    if (x < want_min) want_min = x;
    if (x > want_max) want_max = x;
  }
  CHECK_EQ(got_sum, want_sum);
  CHECK_EQ(got_min, (int64_t)want_min);
  CHECK_EQ(got_max, (int64_t)want_max);

  CHECK_KERNELS_LE(k, 3);
}

// --------------------------------------------------------------------------
// horizontal + vertical together
// --------------------------------------------------------------------------

// The linreg gradient shape: K chains that each multiply before reducing, so
// both fusion directions must apply at once.
TEST(hfuse, weighted_sums_fuse_vertically_and_horizontally) {
  // Each chain needs its own column plus the shared error vector, so the
  // operand budget is usually the binding constraint.
  size_t count = reduction_chains_per_pass();
  if (count + 1 > tf::max_combined_inputs())
    count = tf::max_combined_inputs() - 1;

  const size_t n = 2048;
  Operands cols = make_operands(count, n, 0, 8);
  std::vector<T> err = tf::random_vector<T>(n, 0, 4);
  dpu_vector<T> derr = dpu_vector<T>::from_cpu(err);
  tf::drain();

  std::vector<int64_t> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_future_vector<T> futures;
    for (size_t i = 0; i < count; ++i)
      futures.push_back(sum(cols.dpu[i] * derr));
    for (auto value : futures.get()) actual.push_back((int64_t)value);
  });

  CHECK_EQ(actual.size(), count);
  for (size_t i = 0; i < count && i < actual.size(); ++i) {
    int64_t want = 0;
    for (size_t j = 0; j < n; ++j) want += (int64_t)cols.host[i][j] * err[j];
    if (actual[i] != want)
      tf::fail("gradient " + tf::str(i) + ": got " + tf::str(actual[i]) +
               " want " + tf::str(want));
  }
  CHECK_KERNELS_EQ(k, 1);
  CHECK_FUSIONS_GE(k, count - 1);
}

// The histogram shape: one derived vector feeding N masked counts.
//
// KNOWN BUG 1 (README), many-consumer form: the first sum absorbs the bucket
// vector and erases its producer, and the remaining seven wait forever on an
// event id that will never complete.
TEST_KNOWN_FATAL_IF_FUSED(
    hfuse, histogram_shape_counts_are_correct,
    "deadlocks in EventQueue::process_next: 8 consumers of one "
    "absorbed intermediate") {
  const size_t n = 4096;
  const T bins = 8;
  const T depth = 3;

  std::vector<T> a = tf::iota_vector<T>(n, 0, 1);
  for (T& value : a) value = value % 4096;
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  da.add_fence();

  std::vector<int64_t> counts;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> buckets = (da * bins) >> depth;
    dpu_future_vector<T> futures;
    for (T bin = 0; bin < bins; ++bin) futures.push_back(sum(buckets == bin));
    for (auto value : futures.get()) counts.push_back((int64_t)value);
  });

  std::vector<int64_t> expected((size_t)bins, 0);
  for (size_t i = 0; i < n; ++i) {
    T bucket = (a[i] * bins) >> depth;
    if (bucket >= 0 && bucket < bins) expected[(size_t)bucket]++;
  }
  CHECK_EQ(counts.size(), (size_t)bins);
  for (size_t i = 0; i < counts.size() && i < expected.size(); ++i) {
    if (counts[i] != expected[i])
      tf::fail("bin " + tf::str(i) + ": got " + tf::str(counts[i]) + " want " +
               tf::str(expected[i]));
  }

  // Bucket vector plus 8 masked counts: well inside one or two passes.
  CHECK_KERNELS_LE(k, 3);
}

// --------------------------------------------------------------------------
// horizontal fusion must not change results
// --------------------------------------------------------------------------

// Fused reductions must equal the same reductions forced apart by fences.
TEST(hfuse, fused_reductions_match_serialised_reductions) {
  const size_t count = reduction_chains_per_pass() + 2;
  const size_t n = 1024;
  Operands ops = make_operands(count, n, -10, 10);

  std::vector<int64_t> fused;
  StatsSnapshot k = tf::measure([&] {
    dpu_future_vector<T> futures;
    for (size_t i = 0; i < count; ++i) futures.push_back(sum(ops.dpu[i]));
    for (auto value : futures.get()) fused.push_back((int64_t)value);
  });

  std::vector<int64_t> serial;
  for (size_t i = 0; i < count; ++i) {
    auto future = sum(ops.dpu[i]);
    serial.push_back((int64_t)future.get());
    tf::drain();
  }

  CHECK_EQ(fused.size(), serial.size());
  for (size_t i = 0; i < fused.size() && i < serial.size(); ++i) {
    if (fused[i] != serial[i])
      tf::fail("reduction " + tf::str(i) + ": fused " + tf::str(fused[i]) +
               " != serialised " + tf::str(serial[i]));
  }

  CHECK_KERNELS_LT(k, count);
}
