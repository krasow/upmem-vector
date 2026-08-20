// Every host-side vector operator, checked against a CPU reference.  No kernel
// counts here: these establish that each opcode is correct in isolation, which
// is what the fusion suites build on.

#include "framework.h"

namespace {

using T = int32_t;

// Runs `dpu_op` on random inputs, compares against `cpu_op` lane by lane.
template <typename DpuOp, typename CpuOp>
void check_binary(DpuOp dpu_op, CpuOp cpu_op, T lo = -100, T hi = 100,
                  bool avoid_zero_rhs = false) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, lo, hi);
  std::vector<T> b = tf::random_vector<T>(n, lo, hi);
  if (avoid_zero_rhs)
    for (T& value : b)
      if (value == 0) value = 1;

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  dpu_vector<T> res = dpu_op(da, db);
  std::vector<T> actual = res.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = cpu_op(a[i], b[i]);
  CHECK_VEC_EQ(actual, expected);
}

template <typename DpuOp, typename CpuOp>
void check_unary(DpuOp dpu_op, CpuOp cpu_op, T lo = -100, T hi = 100) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, lo, hi);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> res = dpu_op(da);
  std::vector<T> actual = res.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = cpu_op(a[i]);
  CHECK_VEC_EQ(actual, expected);
}

}  // namespace

// --------------------------------------------------------------------------
// vector OP vector
// --------------------------------------------------------------------------

TEST(elementwise, add) {
  check_binary(
      [](const dpu_vector<T>& a, const dpu_vector<T>& b) { return a + b; },
      [](T x, T y) { return x + y; });
}

TEST(elementwise, sub) {
  check_binary(
      [](const dpu_vector<T>& a, const dpu_vector<T>& b) { return a - b; },
      [](T x, T y) { return x - y; });
}

TEST(elementwise, mul) {
  // Bounded so the product cannot overflow int32.
  check_binary(
      [](const dpu_vector<T>& a, const dpu_vector<T>& b) { return a * b; },
      [](T x, T y) { return x * y; }, -1000, 1000);
}

TEST(elementwise, div) {
  check_binary(
      [](const dpu_vector<T>& a, const dpu_vector<T>& b) { return a / b; },
      [](T x, T y) { return x / y; }, -1000, 1000,
      /*avoid_zero_rhs=*/true);
}

TEST(elementwise, less_than) {
  check_binary(
      [](const dpu_vector<T>& a, const dpu_vector<T>& b) { return a < b; },
      [](T x, T y) { return (T)(x < y ? 1 : 0); }, -20, 20);
}

// OP_SELECT only exists on the pipeline/JIT path (vectordpu.inl
// static_asserts).
#if PIPELINE || JIT
TEST(elementwise, select) {
  const size_t n = tf::elements();
  std::vector<T> cond = tf::random_vector<T>(n, 0, 1);
  std::vector<T> lhs = tf::random_vector<T>(n);
  std::vector<T> rhs = tf::random_vector<T>(n);

  dpu_vector<T> dcond = dpu_vector<T>::from_cpu(cond);
  dpu_vector<T> dlhs = dpu_vector<T>::from_cpu(lhs);
  dpu_vector<T> drhs = dpu_vector<T>::from_cpu(rhs);
  std::vector<T> actual = select(dcond, dlhs, drhs).to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = cond[i] != 0 ? lhs[i] : rhs[i];
  CHECK_VEC_EQ(actual, expected);
}
#endif  // PIPELINE || JIT

// --------------------------------------------------------------------------
// unary
// --------------------------------------------------------------------------

TEST(elementwise, negate) {
  check_unary([](const dpu_vector<T>& a) { return -a; },
              [](T x) { return -x; });
}

TEST(elementwise, absolute) {
  check_unary([](const dpu_vector<T>& a) { return abs(a); },
              [](T x) { return x < 0 ? -x : x; });
}

// --------------------------------------------------------------------------
// vector OP scalar
// --------------------------------------------------------------------------

TEST(elementwise, add_scalar) {
  check_unary([](const dpu_vector<T>& a) { return a + (T)7; },
              [](T x) { return x + 7; });
}

TEST(elementwise, scalar_add_vector) {
  check_unary([](const dpu_vector<T>& a) { return (T)7 + a; },
              [](T x) { return 7 + x; });
}

TEST(elementwise, sub_scalar) {
  check_unary([](const dpu_vector<T>& a) { return a - (T)7; },
              [](T x) { return x - 7; });
}

TEST(elementwise, mul_scalar) {
  check_unary([](const dpu_vector<T>& a) { return a * (T)3; },
              [](T x) { return x * 3; });
}

TEST(elementwise, scalar_mul_vector) {
  check_unary([](const dpu_vector<T>& a) { return (T)3 * a; },
              [](T x) { return 3 * x; });
}

TEST(elementwise, div_scalar) {
  check_unary([](const dpu_vector<T>& a) { return a / (T)3; },
              [](T x) { return x / 3; });
}

TEST(elementwise, shift_right_scalar) {
  // Arithmetic shift: the reference must also be a signed shift.
  check_unary([](const dpu_vector<T>& a) { return a >> (T)2; },
              [](T x) { return x >> 2; });
}

TEST(elementwise, eq_scalar) {
  check_unary([](const dpu_vector<T>& a) { return a == (T)5; },
              [](T x) { return (T)(x == 5 ? 1 : 0); }, 0, 9);
}

// --------------------------------------------------------------------------
// compound assignment (in-place)
//
// Chained in-place ops had two separate bugs, both fixed:
//   * the recorded absorbed_rpn was self-referential, so a consumer inlining it
//     re-read a buffer the producer had already overwritten
//     (EventQueue::enqueue now skips that registration when the output aliases
//     an input); and
//   * a fused event could take a dependency on an event it had just absorbed,
//     which never completes (detail::adopt_fused_event now filters those).
// --------------------------------------------------------------------------

TEST(elementwise, compound_single_op_scalar) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -50, 50);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  da += (T)10;
  std::vector<T> actual = da.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] + 10;
  CHECK_VEC_EQ(actual, expected);
}

TEST(elementwise, compound_single_op_vector) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -50, 50);
  std::vector<T> b = tf::random_vector<T>(n, -50, 50);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  da += db;
  std::vector<T> actual = da.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] + b[i];
  CHECK_VEC_EQ(actual, expected);
}

// A fence forces each producer to materialise: the configuration that works.
TEST(elementwise, compound_chain_fenced) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -50, 50);
  std::vector<T> b = tf::random_vector<T>(n, 1, 50);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

  da += db;
  dpu_fence();
  da -= db;
  dpu_fence();
  da *= db;
  dpu_fence();
  da /= db;
  dpu_fence();
  std::vector<T> actual = da.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) {
    T v = a[i];
    v += b[i];
    v -= b[i];
    v *= b[i];
    v /= b[i];
    expected[i] = v;
  }
  CHECK_VEC_EQ(actual, expected);
}

TEST(elementwise, compound_chain_two_scalar_ops) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 40);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  da += (T)10;
  da -= (T)3;
  std::vector<T> actual = da.to_cpu();

  CHECK_VEC_EQ(actual, tf::constant_vector<T>(n, 47));
}

TEST(elementwise, compound_chain_two_shifts) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 40);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  da >>= (T)1;
  da >>= (T)1;
  std::vector<T> actual = da.to_cpu();

  CHECK_VEC_EQ(actual, tf::constant_vector<T>(n, 10));
}

TEST(elementwise, compound_chain_vector_ops) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -50, 50);
  std::vector<T> b = tf::random_vector<T>(n, 1, 50);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

  da += db;
  da -= db;
  da *= db;
  da /= db;
  std::vector<T> actual = da.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) {
    T v = a[i];
    v += b[i];
    v -= b[i];
    v *= b[i];
    v /= b[i];
    expected[i] = v;
  }
  CHECK_VEC_EQ(actual, expected);
}

TEST(elementwise, compound_chain_five_scalar_ops) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 40);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  da += (T)10;
  da -= (T)3;
  da *= (T)4;
  da /= (T)2;
  da >>= (T)1;
  std::vector<T> actual = da.to_cpu();

  CHECK_VEC_EQ(actual, tf::constant_vector<T>(n, 47));
}

// --------------------------------------------------------------------------
// aliasing and vector lifetime
// --------------------------------------------------------------------------

// Both operands the same vector -- the shape that broke KNN fusion.
TEST(elementwise, self_aliased_binary) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -100, 100);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> res = da * da;
  std::vector<T> actual = res.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] * a[i];
  CHECK_VEC_EQ(actual, expected);
}

TEST(elementwise, self_aliased_compound) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -100, 100);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  da *= da;
  std::vector<T> actual = da.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] * a[i];
  CHECK_VEC_EQ(actual, expected);
}

// The second read must not re-run an already-consumed fused chain.
TEST(elementwise, repeated_readback_is_stable) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n);
  std::vector<T> b = tf::random_vector<T>(n);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  dpu_vector<T> res = da + db;

  std::vector<T> first = res.to_cpu();
  std::vector<T> second = res.to_cpu();
  CHECK_VEC_EQ(second, first);
}

// KNOWN BUG 1 (README): an intermediate with two readers is absorbed into the
// first, so the second reads unwritten MRAM as zeros.
TEST(elementwise, shared_intermediate_two_consumers) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -50, 50);
  std::vector<T> b = tf::random_vector<T>(n, -50, 50);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

  dpu_vector<T> shared = da + db;
  dpu_vector<T> left = shared * (T)2;
  dpu_vector<T> right = shared - (T)5;

  std::vector<T> actual_left = left.to_cpu();
  std::vector<T> actual_right = right.to_cpu();

  std::vector<T> expected_left(n), expected_right(n);
  for (size_t i = 0; i < n; ++i) {
    expected_left[i] = (a[i] + b[i]) * 2;
    expected_right[i] = (a[i] + b[i]) - 5;
  }
  CHECK_VEC_EQ(actual_left, expected_left);
  CHECK_VEC_EQ(actual_right, expected_right);
}

// --------------------------------------------------------------------------
// sizes and shapes
// --------------------------------------------------------------------------

namespace {
// Same computation at an explicit element count.  Only the first n lanes are
// compared: to_cpu can return a longer vector (known bug 5).
void check_add_at_size(size_t n) {
  std::vector<T> a = tf::random_vector<T>(n);
  std::vector<T> b = tf::random_vector<T>(n);

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  std::vector<T> actual = (da + db).to_cpu();

  if (actual.size() < n) {
    tf::fail("n=" + tf::str(n) + ": to_cpu returned only " +
             tf::str(actual.size()) + " elements");
    return;
  }
  actual.resize(n);

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] + b[i];
  std::string message;
  if (!tf::vec_equal(actual, expected, message))
    tf::fail("n=" + tf::str(n) + ": " + message);
}
}  // namespace

// Rounded to what the readback path handles; see suite_sharding.cc for the
// rest.
TEST(elementwise, size_one_block) { check_add_at_size(BLOCK_SIZE); }

TEST(elementwise, size_two_blocks) { check_add_at_size(BLOCK_SIZE * 2); }

TEST(elementwise, size_large) { check_add_at_size(65537); }

TEST(elementwise, size_one_element_per_dpu) {
  check_add_at_size(DpuRuntime::get().num_dpus());
}

TEST(elementwise, size_single_element) { check_add_at_size(1); }

// to_cpu returns exactly as many elements as the vector holds.  It used to pad
// the result to 8 bytes per DPU when a shard held one element.
TEST(elementwise, to_cpu_size_matches_vector_size) {
  const size_t n = DpuRuntime::get().num_dpus();
  std::vector<T> a = tf::constant_vector<T>(n, 3);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  std::vector<T> host = da.to_cpu();
  CHECK_EQ(host.size(), n);
}
