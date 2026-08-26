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
