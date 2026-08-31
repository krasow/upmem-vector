#include <polymerpim.h>

#include "framework.h"

using namespace polymerpim;
using T = int32_t;

TEST(public_api, lazy_expression_captures_scalars) {
#if !PIPELINE
  SKIP("requires PIPELINE=1");
#else
  const size_t n = tf::elements();
  std::vector<T> host = tf::random_vector<T>(n, -20, 20);
  DPUVector<T> input(host);

  T offset = 7;
  auto actual = (sqr(input - offset) + offset).to_cpu();
  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i)
    expected[i] = (host[i] - offset) * (host[i] - offset) + offset;
  CHECK_VEC_EQ(actual, expected);
#endif
}

TEST(public_api, scalar_values_reuse_one_jit_program) {
#if !JIT
  SKIP("requires JIT=1");
#else
  std::vector<T> host = tf::random_vector<T>(1024, -20, 20);
  DPUVector<T> input(host);

  auto run = [&](T offset) {
    auto result = maximum(sqr(input - offset) + (T)37);
    sync();
    return result.get();
  };

  (void)run(3);
  RuntimeStatistics before = statistics();
  (void)run(11);
  RuntimeStatistics second = statistics() - before;
  CHECK_EQ(second.jit_kernel_compiles, 0u);
#endif
}

TEST(public_api, arg_reductions_return_value_and_first_index_in_one_pass) {
#if !PIPELINE
  SKIP("requires PIPELINE=1");
#else
  std::vector<T> host = {-4, 9, 2, 9, 1};
  DPUVector<T> input(host);

  sync();
  RuntimeStatistics before = statistics();
  ArgResult maximum = argmax(input + (T)3).get();
  sync();
  CHECK_EQ(statistics().compute_launches - before.compute_launches, 1u);
  CHECK_EQ(maximum.value, 12);
  CHECK_EQ(maximum.index, 1u);

  before = statistics();
  ArgResult minimum = argmin(input - (T)2).get();
  sync();
  CHECK_EQ(statistics().compute_launches - before.compute_launches, 1u);
  CHECK_EQ(minimum.value, -6);
  CHECK_EQ(minimum.index, 0u);
#endif
}

TEST(public_api, interpreter_falls_back_for_deep_tree) {
#if !PIPELINE || JIT
  SKIP("requires PIPELINE=1 JIT=0");
#else
  const size_t n = tf::elements();
  std::vector<T> host = tf::random_vector<T>(n, -20, 20);
  DPUVector<T> input(host);

  auto actual = (sqr(input - (T)3) + sqr(input - (T)7)).to_cpu();
  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i)
    expected[i] = (host[i] - 3) * (host[i] - 3) + (host[i] - 7) * (host[i] - 7);
  CHECK_VEC_EQ(actual, expected);
#endif
}

TEST(public_api, local_updates_flush_on_read) {
#if !JIT
  SKIP("requires JIT=1");
#else
  const size_t n = tf::elements();
  std::vector<T> host(n);
  for (size_t i = 0; i < n; ++i) host[i] = (T)(i % 4);
  DPUVector<T> input(host);
  DPULocalVector<T> bins(4);

  bins[input] += (T)1;
  auto actual = bins.to_cpu();
  std::vector<T> expected(4, (T)(n / 4));
  for (size_t i = 0; i < n % 4; ++i) ++expected[i];
  CHECK_VEC_EQ(actual, expected);
#endif
}

TEST(public_api, local_updates_flush_on_sync) {
#if !JIT
  SKIP("requires JIT=1");
#else
  const size_t n = tf::elements();
  std::vector<T> host(n);
  for (size_t i = 0; i < n; ++i) host[i] = (T)(i % 4);
  DPUVector<T> input(host);
  DPULocalVector<T> bins(4);

  bins[input] += (T)1;
  RuntimeStatistics before = statistics();
  sync();
  RuntimeStatistics flushed = statistics() - before;
  CHECK_GE(flushed.compute_launches, 1u);

  auto actual = bins.to_cpu();
  std::vector<T> expected(4, (T)(n / 4));
  for (size_t i = 0; i < n % 4; ++i) ++expected[i];
  CHECK_VEC_EQ(actual, expected);
#endif
}

TEST(public_api, sized_constructor_is_a_free_zero_fill) {
  const size_t n = tf::elements();
  RuntimeStatistics before = statistics();
  DPUVector<T> zeros(n, "zeros");
  DPUVector<T> sevens(n, (T)7, "sevens");
  RuntimeStatistics allocated = statistics() - before;

  // A fill is pending contents, not storage: no transfer, no kernel.
  CHECK_EQ(zeros.size(), n);
  CHECK_EQ(sevens.size(), n);
  CHECK_EQ(allocated.host_transfers, 0u);
  CHECK_EQ(allocated.compute_launches, 0u);

  // Reading one anyway still yields the promised contents.
  CHECK_VEC_EQ(zeros.to_cpu(), std::vector<T>(n, (T)0));
  CHECK_VEC_EQ(sevens.to_cpu(), std::vector<T>(n, (T)7));
}

// Ragged tails exercise the kernel's block loop and its 8-byte DMA rounding.
TEST(public_api, device_fill_covers_ragged_lengths) {
  const uint32_t dpus = tf::dpus();
  const size_t lengths[] = {
      1,
      2,
      7,
      8,
      9,
      15,
      16,
      17,
      63,
      64,
      65,
      (size_t)dpus - 1,
      (size_t)dpus,
      (size_t)dpus + 1,
      (size_t)dpus * 16 + 3,
      4099,
  };
  for (size_t n : lengths) {
    if (n == 0) continue;
    DPUVector<T> filled(n, (T)-3, "ragged");
    CHECK_VEC_EQ(filled.to_cpu(), std::vector<T>(n, (T)-3));
  }
}

// The value round-trips through a uint32_t argument slot.
TEST(public_api, device_fill_preserves_the_value) {
  const size_t n = tf::elements();
  const T values[] = {0, 1, -1, 7, -3, 2147483647, -2147483648};
  for (T v : values) {
    DPUVector<T> filled(n, v, "value");
    CHECK_VEC_EQ(filled.to_cpu(), std::vector<T>(n, v));
  }
}

// A fill has no input, so fusing it would build a program around one that
// isn't there and yield the wrong constant.
TEST(public_api, device_fill_is_not_fused_into_its_consumer) {
  const size_t n = tf::elements();
  std::vector<T> host = tf::random_vector<T>(n, -20, 20);
  DPUVector<T> input(host);
  sync();

  DPUVector<T> filled(n, (T)5, "five");
  auto actual = (filled + input).to_cpu();
  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = (T)(host[i] + 5);
  CHECK_VEC_EQ(actual, expected);
}

TEST(public_api, zero_fill_folds_out_of_its_first_use) {
#if !PIPELINE
  SKIP("requires PIPELINE=1");
#else
  const size_t n = tf::elements();
  const uint32_t dim = 4;
  std::vector<std::vector<T> > host(dim);
  std::vector<DPUVector<T> > columns;
  std::vector<T> query(dim);
  for (uint32_t d = 0; d < dim; ++d) {
    host[d] = tf::random_vector<T>(n, -20, 20);
    columns.emplace_back(host[d]);
    sync();
    query[d] = (T)(d + 1);
  }

  // Starting at d=0 still runs as one kernel: the fill folds away and the
  // assignments stay unevaluated until argmax.
  RuntimeStatistics before = statistics();
  DPUVector<T> score(n);
  for (uint32_t d = 0; d < dim; ++d) score = score + columns[d] + query[d];
  auto pending = argmax(score);
  sync();
  ArgResult actual = pending.get();
  RuntimeStatistics fused = statistics() - before;
  CHECK_EQ(fused.compute_launches, 1u);

  T best = 0;
  uint32_t best_index = 0;
  for (size_t i = 0; i < n; ++i) {
    T sum = 0;
    for (uint32_t d = 0; d < dim; ++d) sum = (T)(sum + host[d][i] + query[d]);
    if (i == 0 || sum > best) {
      best = sum;
      best_index = (uint32_t)i;
    }
  }
  CHECK_EQ(actual.value, best);
  CHECK_EQ(actual.index, best_index);
#endif
}

TEST(public_api, assignment_defers_until_something_needs_storage) {
#if !PIPELINE
  SKIP("requires PIPELINE=1");
#else
  const size_t n = tf::elements();
  std::vector<T> host = tf::random_vector<T>(n, -20, 20);
  DPUVector<T> input(host);
  sync();

  RuntimeStatistics before = statistics();
  DPUVector<T> doubled(n);
  doubled = input * (T)2;
  sync();
  RuntimeStatistics assigned = statistics() - before;
  CHECK_EQ(assigned.compute_launches, 0u);
  CHECK_EQ(doubled.size(), n);

  // Construction is the other half of the rule: naming a value runs the
  // expression, so several consumers read one result instead of re-fusing it
  // into each of them (linreg's error vector, read once per dimension).
  before = statistics();
  DPUVector<T> named = input * (T)3;
  sync();
  RuntimeStatistics constructed = statistics() - before;
  CHECK_EQ(constructed.compute_launches, 1u);

  // First consumer fuses; a second materializes once instead of re-evaluating.
  before = statistics();
  auto first = sum(doubled);
  auto second = sum(doubled + (T)1);
  sync();
  (void)first.get();
  (void)second.get();
  RuntimeStatistics consumed = statistics() - before;
  CHECK_LE(consumed.compute_launches, 3u);

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = (T)(host[i] * 2);
  CHECK_VEC_EQ(doubled.to_cpu(), expected);
#endif
}

// A derived vector read by more than one reduction: folding it into a
// multi-chain consumer used to drop the later chains' pushes, which aborted the
// JIT on a stack underflow and faulted the DPUs under the interpreter.
TEST(public_api, intermediate_feeding_several_reductions) {
#if !PIPELINE
  SKIP("requires PIPELINE=1");
#else
  const size_t n = tf::elements();
  std::vector<T> a_host = tf::random_vector<T>(n, -20, 20);
  std::vector<T> b_host = tf::random_vector<T>(n, -20, 20);
  DPUVector<T> a(a_host);
  sync();
  DPUVector<T> b(b_host);
  sync();

  DPUVector<T> factor = ((a == (T)1) * (T)2 - (T)1) * (T)-1;
  std::vector<DpuFuture<T> > sums;
  sums.push_back(sum(sqr(factor)));
  for (int lane = 0; lane < 3; ++lane) sums.push_back(sum(factor * b));
  sync();

  int64_t actual = 0;
  for (auto& pending : sums) actual += (int64_t)pending.get();

  int64_t expected = 0;
  for (size_t i = 0; i < n; ++i) {
    T factor_i = (T)((((a_host[i] == 1) ? 1 : 0) * 2 - 1) * -1);
    expected += (int64_t)factor_i * factor_i;
    expected += 3 * ((int64_t)factor_i * b_host[i]);
  }
  CHECK_EQ(actual, expected);
#endif
}

// A fill is only free where it folds away.  Anywhere else -- the wrong side of
// a multiply, a constructor, a reduction -- it has to behave like the zeros it
// promised rather than like the uninitialised memory it used to be.
TEST(public_api, fill_behaves_as_its_value_outside_the_fold) {
  const size_t n = tf::elements();
  std::vector<T> host = tf::random_vector<T>(n, -20, 20);
  DPUVector<T> input(host);
  sync();

  DPUVector<T> zeros(n);
  zeros = zeros * input;
  CHECK_VEC_EQ(zeros.to_cpu(), std::vector<T>(n, (T)0));

  DPUVector<T> reversed(n);
  reversed = input * reversed;
  CHECK_VEC_EQ(reversed.to_cpu(), std::vector<T>(n, (T)0));

  DPUVector<T> scaled(n, (T)7);
  scaled = scaled * input;
  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = (T)(host[i] * 7);
  CHECK_VEC_EQ(scaled.to_cpu(), expected);

  // A fill assigned an expression that folds to another scalar stays a fill of
  // that value, sized by the destination.
  DPUVector<T> shifted(n);
  shifted = shifted + (T)5;
  CHECK_EQ(shifted.size(), n);
  CHECK_VEC_EQ(shifted.to_cpu(), std::vector<T>(n, (T)5));
}
