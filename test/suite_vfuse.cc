// Vertical fusion: a chain of dependent ops must collapse into one kernel pass.
//
// Each test asserts both halves of the contract -- the values, and the number
// of kernel passes.  The second is the half that regresses silently: a rule
// that stops firing still produces correct output, just slower.

#include "framework.h"

namespace {

using T = int32_t;

struct Inputs {
  std::vector<T> a, b, c, d;
  dpu_vector<T> da, db, dc, dd;
};

// Uploads four operands and drains, so a later probe sees only compute events.
Inputs make_inputs(size_t n, T lo = -50, T hi = 50) {
  Inputs in;
  in.a = tf::random_vector<T>(n, lo, hi);
  in.b = tf::random_vector<T>(n, lo, hi);
  in.c = tf::random_vector<T>(n, lo, hi);
  in.d = tf::random_vector<T>(n, lo, hi);
  in.da = dpu_vector<T>::from_cpu(in.a, "a");
  in.db = dpu_vector<T>::from_cpu(in.b, "b");
  in.dc = dpu_vector<T>::from_cpu(in.c, "c");
  in.dd = dpu_vector<T>::from_cpu(in.d, "d");
  tf::drain();
  return in;
}

}  // namespace

// --------------------------------------------------------------------------
// chain depth
// --------------------------------------------------------------------------

// The second op consumes the first on-stack, so the intermediate never reaches
// MRAM and one kernel does both.
TEST(vfuse, two_op_chain_is_one_kernel) {
  const size_t n = tf::elements();
  Inputs in = make_inputs(n);

  std::vector<T> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> res = (in.da + in.db) - in.dc;
    actual = res.to_cpu();
  });

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = (in.a[i] + in.b[i]) - in.c[i];
  CHECK_VEC_EQ(actual, expected);

  CHECK_KERNELS_EQ(k, 1);
}

TEST(vfuse, four_op_chain_is_one_kernel) {
  const size_t n = tf::elements();
  Inputs in = make_inputs(n);

  std::vector<T> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> res = ((in.da + in.db) - in.dc) * in.dd;
    actual = res.to_cpu();
  });

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i)
    expected[i] = ((in.a[i] + in.b[i]) - in.c[i]) * in.d[i];
  CHECK_VEC_EQ(actual, expected);

  CHECK_KERNELS_EQ(k, 1);
}

// The RPN builder has a separate path per opcode class; exercise all three.
TEST(vfuse, mixed_op_chain_is_one_kernel) {
  const size_t n = tf::elements();
  Inputs in = make_inputs(n, -30, 30);

  std::vector<T> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> res = abs(-((in.da + in.db) * (T)3 - in.dc)) >> (T)1;
    actual = res.to_cpu();
  });

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) {
    T v = (in.a[i] + in.b[i]) * 3 - in.c[i];
    v = -v;
    if (v < 0) v = -v;
    expected[i] = v >> 1;
  }
  CHECK_VEC_EQ(actual, expected);

  CHECK_KERNELS_EQ(k, 1);
}

// Bounded by MAX_VFUSE_OPS, not by the number of source-level operators: a long
// scalar chain stays one kernel as long as its RPN fits.
TEST(vfuse, long_scalar_chain_stays_one_kernel) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -1000, 1000);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  tf::drain();

  // Each `+ 1` costs 5 RPN bytes (opcode + 4-byte immediate), so keep the
  // chain comfortably inside MAX_VFUSE_OPS.
  const size_t steps = tf::max_vfuse_ops() / 8;
  if (steps < 2) SKIP("MAX_VFUSE_OPS too small for this test");

  std::vector<T> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> res = da + (T)1;
    for (size_t i = 1; i < steps; ++i) res = res + (T)1;
    actual = res.to_cpu();
  });

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] + (T)steps;
  CHECK_VEC_EQ(actual, expected);

  CHECK_KERNELS_EQ(k, 1);
}

// A chain far longer than MAX_VFUSE_OPS.
//
// That limit is the size of the generic interpreter's args.pipeline.ops buffer,
// so it only binds when there is no JIT -- a generated kernel carries its
// program in C and can be any length.  With JIT=0 the inliner stops before
// overflowing (the chain then costs more passes); with JIT=1 it collapses to
// one.  Either way the values are right; the interpreter used to truncate the
// tail silently.
TEST(vfuse, chain_far_beyond_max_ops_is_correct) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -1000, 1000);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  tf::drain();

  // 5 RPN bytes per step, so ~3x the MAX_VFUSE_OPS budget.
  const size_t steps = (tf::max_vfuse_ops() * 3) / 5 + 4;

  std::vector<T> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> res = da + (T)1;
    for (size_t i = 1; i < steps; ++i) res = res + (T)1;
    actual = res.to_cpu();
  });

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] + (T)steps;
  CHECK_VEC_EQ(actual, expected);
  if (tf::verbose())
    std::cout << "         " << steps << " ops -> " << k.compute_launches
              << " kernel pass(es)\n";
}

// --------------------------------------------------------------------------
// operand budget
// --------------------------------------------------------------------------

// Limited by MAX_COMBINED_INPUTS (primary + MAX_VFUSE_INPUTS slots), not by
// chain length.
TEST(vfuse, chain_within_input_budget_is_one_kernel) {
  const size_t n = tf::elements();
  const size_t count = tf::max_combined_inputs();
  std::vector<std::vector<T>> host;
  std::vector<dpu_vector<T>> vecs;
  host.reserve(count);
  vecs.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    host.push_back(tf::random_vector<T>(n, -20, 20));
    vecs.push_back(dpu_vector<T>::from_cpu(host.back()));
  }
  tf::drain();

  std::vector<T> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> res = vecs[0] + vecs[1];
    for (size_t i = 2; i < count; ++i) res = res + vecs[i];
    actual = res.to_cpu();
  });

  std::vector<T> expected(n, 0);
  for (size_t i = 0; i < n; ++i) {
    T sum = 0;
    for (size_t v = 0; v < count; ++v) sum += host[v][i];
    expected[i] = sum;
  }
  CHECK_VEC_EQ(actual, expected);
  CHECK_KERNELS_EQ(k, 1);
}

// One vector past the budget: exactly two passes, since the extra input cannot
// fit the operand table.
TEST(vfuse, chain_over_input_budget_splits) {
  const size_t n = tf::elements();
  const size_t count = tf::max_combined_inputs() + 1;
  std::vector<std::vector<T>> host;
  std::vector<dpu_vector<T>> vecs;
  host.reserve(count);
  vecs.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    host.push_back(tf::random_vector<T>(n, -20, 20));
    vecs.push_back(dpu_vector<T>::from_cpu(host.back()));
  }
  tf::drain();

  std::vector<T> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> res = vecs[0] + vecs[1];
    for (size_t i = 2; i < count; ++i) res = res + vecs[i];
    actual = res.to_cpu();
  });

  std::vector<T> expected(n, 0);
  for (size_t i = 0; i < n; ++i) {
    T sum = 0;
    for (size_t v = 0; v < count; ++v) sum += host[v][i];
    expected[i] = sum;
  }
  CHECK_VEC_EQ(actual, expected);
  CHECK_KERNELS_GE(k, 2);
  CHECK_KERNELS_LE(k, 2);
}

// Re-use costs one operand slot, not one per use.
TEST(vfuse, repeated_operand_reuses_one_slot) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -4, 4);
  std::vector<T> b = tf::random_vector<T>(n, -4, 4);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  tf::drain();

  const size_t steps = 10;
  std::vector<T> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> res = da + db;
    for (size_t i = 1; i < steps; ++i) res = res + (i % 2 ? da : db);
    actual = res.to_cpu();
  });

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) {
    T v = a[i] + b[i];
    for (size_t s = 1; s < steps; ++s) v += (s % 2 ? a[i] : b[i]);
    expected[i] = v;
  }
  CHECK_VEC_EQ(actual, expected);
  CHECK_KERNELS_EQ(k, 1);
}

// --------------------------------------------------------------------------
// fusion barriers
// --------------------------------------------------------------------------

// Nothing fuses onto a finished reduction, but the work feeding it does.
//
// Two variants, because when the result is read matters.  A producer is only
// dropped once the vector's last handle has gone (see
// EventQueue::output_still_needed), and reading inside the same full-expression
// that built it keeps the temporary alive, costing one extra pass.
TEST(vfuse, reduction_terminates_the_chain) {
  const size_t n = tf::elements();
  Inputs in = make_inputs(n, -20, 20);

  int64_t actual = 0;
  StatsSnapshot k = tf::measure([&] {
    // (a + b) fuses into the sum, giving one kernel for the whole expression.
    actual = (int64_t)sum(in.da + in.db).get();
  });

  int64_t expected = 0;
  for (size_t i = 0; i < n; ++i) expected += (int64_t)in.a[i] + in.b[i];
  CHECK_EQ(actual, expected);
  // `sum(a + b).get()` reads inside the expression, so the `a + b` temporary is
  // still alive when the fence runs and its producer cannot be dropped.
  CHECK_KERNELS_EQ(k, 2);
}

// Holding the future first lets the temporary die, so the producer is dropped
// and the whole expression collapses into one pass.
TEST(vfuse, reduction_of_expression_is_one_kernel) {
  const size_t n = tf::elements();
  Inputs in = make_inputs(n, -20, 20);

  int64_t actual = 0;
  StatsSnapshot k = tf::measure([&] {
    auto future = sum(in.da + in.db);
    actual = (int64_t)future.get();
  });

  int64_t expected = 0;
  for (size_t i = 0; i < n; ++i) expected += (int64_t)in.a[i] + in.b[i];
  CHECK_EQ(actual, expected);
  CHECK_KERNELS_EQ(k, 1);
}

// An explicit fence is a hard barrier: the ops on either side must not merge.
TEST(vfuse, explicit_fence_prevents_fusion) {
  const size_t n = tf::elements();
  Inputs in = make_inputs(n);

  std::vector<T> actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> mid = in.da + in.db;
    mid.add_fence();
    dpu_vector<T> res = mid - in.dc;
    actual = res.to_cpu();
  });

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = (in.a[i] + in.b[i]) - in.c[i];
  CHECK_VEC_EQ(actual, expected);
  CHECK_KERNELS_EQ(k, 2);
}

// A host readback forces the intermediate to materialise, splitting the chain.
TEST(vfuse, host_readback_of_intermediate_splits_chain) {
  const size_t n = tf::elements();
  Inputs in = make_inputs(n);

  std::vector<T> mid_host, actual;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> mid = in.da + in.db;
    mid_host = mid.to_cpu();
    dpu_vector<T> res = mid - in.dc;
    actual = res.to_cpu();
  });

  std::vector<T> expected_mid(n), expected(n);
  for (size_t i = 0; i < n; ++i) {
    expected_mid[i] = in.a[i] + in.b[i];
    expected[i] = expected_mid[i] - in.c[i];
  }
  CHECK_VEC_EQ(mid_host, expected_mid);
  CHECK_VEC_EQ(actual, expected);
  CHECK_KERNELS_EQ(k, 2);
}

// KNOWN BUG 1 (README): one producer, two consumers -- only the first is right.
// The first consumer absorbs the producer and expand_absorbed_inputs erases it,
// so the shared vector never reaches MRAM and the second consumer reads zeros:
//
//   shared = a * b;  left = shared + c  -> 16 (ok);  right = shared - d -> -100
//
// A fence after the producer avoids it.
TEST(vfuse, diamond_dependency_is_correct) {
  const size_t n = tf::elements();
  Inputs in = make_inputs(n, -30, 30);

  std::vector<T> left, right;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> shared = in.da * in.db;
    dpu_vector<T> l = shared + in.dc;
    dpu_vector<T> r = shared - in.dd;
    left = l.to_cpu();
    right = r.to_cpu();
  });

  std::vector<T> expected_left(n), expected_right(n);
  for (size_t i = 0; i < n; ++i) {
    T shared = in.a[i] * in.b[i];
    expected_left[i] = shared + in.c[i];
    expected_right[i] = shared - in.d[i];
  }
  CHECK_VEC_EQ(left, expected_left);
  CHECK_VEC_EQ(right, expected_right);

  // Either the shared term materialises (3 passes) or it is recomputed inside
  // each consumer (2 passes); both are legal, more than 3 is not.
  CHECK_KERNELS_LE(k, 3);
}

// --------------------------------------------------------------------------
// fusion must not change results
// --------------------------------------------------------------------------

// KNOWN BUG 3 (README): a binary op over two fresh intermediates drops one.
// With a=2 b=3 c=4 d=10, `(a+b)*c - (d-a)` returns 8 -- exactly `d-a`, so the
// left intermediate was absorbed and read back as 0.  The fenced arm below is
// correct, which is what makes this a fusion bug rather than an arithmetic one.
TEST(vfuse, fused_matches_fenced_evaluation) {
  const size_t n = tf::elements();
  Inputs in = make_inputs(n, -25, 25);

  std::vector<T> fused;
  StatsSnapshot k = tf::measure([&] {
    dpu_vector<T> res = abs((in.da + in.db) * in.dc - (in.dd - in.da)) + (T)7;
    fused = res.to_cpu();
  });

  std::vector<T> fenced;
  {
    dpu_vector<T> t1 = in.da + in.db;
    t1.add_fence();
    dpu_vector<T> t2 = t1 * in.dc;
    t2.add_fence();
    dpu_vector<T> t3 = in.dd - in.da;
    t3.add_fence();
    dpu_vector<T> t4 = t2 - t3;
    t4.add_fence();
    dpu_vector<T> t5 = abs(t4);
    t5.add_fence();
    dpu_vector<T> t6 = t5 + (T)7;
    t6.add_fence();
    fenced = t6.to_cpu();
  }

  // Third opinion, so a failure names which arm is wrong.
  std::vector<T> host_ref(n);
  for (size_t i = 0; i < n; ++i) {
    T v = (in.a[i] + in.b[i]) * in.c[i] - (in.d[i] - in.a[i]);
    if (v < 0) v = -v;
    host_ref[i] = v + 7;
  }
  CHECK_VEC_EQ(fenced, host_ref);
  CHECK_VEC_EQ(fused, host_ref);
  CHECK_KERNELS_LE(k, 2);
  (void)k;
}

// Minimal form of the bug above.
TEST(vfuse, binary_op_over_two_intermediates) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 2);
  std::vector<T> b = tf::constant_vector<T>(n, 3);
  std::vector<T> c = tf::constant_vector<T>(n, 4);
  std::vector<T> d = tf::constant_vector<T>(n, 10);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  dpu_vector<T> dc = dpu_vector<T>::from_cpu(c);
  dpu_vector<T> dd = dpu_vector<T>::from_cpu(d);
  tf::drain();

  // (2 + 3) * 4 - (10 - 2) = 20 - 8 = 12
  std::vector<T> actual = ((da + db) * dc - (dd - da)).to_cpu();
  CHECK_VEC_EQ(actual, tf::constant_vector<T>(n, 12));
}
