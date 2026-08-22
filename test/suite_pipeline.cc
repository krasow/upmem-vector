// The explicit pipeline / JIT API.  These bypass the operator overloads, so
// they pin down the RPN contract the fusion passes rewrite: operand slot
// numbering, scalar immediates vs variables, stack discipline, and chain
// separation.

#include "framework.h"
#if JIT_PIPELINE_FALLBACK
#include <detail/rpn.h>
#endif

namespace {

using T = int32_t;

}  // namespace

#if PIPELINE

namespace {
using expr = dpu_pipeline_expr<T>;
}  // namespace

// --------------------------------------------------------------------------
// hand-written RPN
// --------------------------------------------------------------------------

TEST(pipeline, rpn_single_input_unary) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -100, 100);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  // -a
  std::vector<T> actual = da.pipeline({OP_PUSH_INPUT, OP_NEGATE}).vec.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = -a[i];
  CHECK_VEC_EQ(actual, expected);
}

TEST(pipeline, rpn_two_inputs_binary) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -100, 100);
  std::vector<T> b = tf::random_vector<T>(n, -100, 100);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

  // a + b, with b in operand slot 0
  std::vector<T> actual =
      da.pipeline({OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_ADD}, {db})
          .vec.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] + b[i];
  CHECK_VEC_EQ(actual, expected);
}

// abs(-((a + b) - a)) as one RPN program.
TEST(pipeline, rpn_deep_chain) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -100, 100);
  std::vector<T> b = tf::random_vector<T>(n, -100, 100);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

  std::vector<uint8_t> ops = {
      OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_ADD, OP_PUSH_INPUT,
      OP_SUB,        OP_NEGATE,         OP_ABS};
  std::vector<T> actual = da.pipeline(ops, {db}).vec.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) {
    T v = -((a[i] + b[i]) - a[i]);
    expected[i] = v < 0 ? -v : v;
  }
  CHECK_VEC_EQ(actual, expected);
}

// OP_DUP must duplicate the top of stack, not re-read the input.
TEST(pipeline, rpn_dup_squares_the_top_of_stack) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -40, 40);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  // (a + 1) squared, via DUP
  std::vector<uint8_t> ops = {OP_PUSH_INPUT, OP_ADD_SCALAR, 1, 0, 0, 0,
                              OP_DUP,        OP_MUL};
  std::vector<T> actual = da.pipeline(ops).vec.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = (a[i] + 1) * (a[i] + 1);
  CHECK_VEC_EQ(actual, expected);
}

// Scalar variables index the scalars table and the fusion passes renumber them;
// an off-by-one shows up as the wrong constant.
TEST(pipeline, rpn_scalar_variables) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -40, 40);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  // (a * s0) + s1  with s0 = 3, s1 = 100
  std::vector<uint8_t> ops = {OP_PUSH_INPUT, OP_MUL_SCALAR_VAR, 0,
                              OP_ADD_SCALAR_VAR, 1};
  std::vector<T> actual = da.pipeline(ops, {}, {3u, 100u}).vec.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] * 3 + 100;
  CHECK_VEC_EQ(actual, expected);
}

TEST(pipeline, rpn_select) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -40, 40);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  // a < 0 ? -a : a  (i.e. abs), built from SELECT
  std::vector<uint8_t> ops = {
      OP_PUSH_INPUT, OP_LT_SCALAR,  0,        0, 0, 0, OP_PUSH_INPUT,
      OP_NEGATE,     OP_PUSH_INPUT, OP_SELECT};
  std::vector<T> actual = da.pipeline(ops).vec.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] < 0 ? -a[i] : a[i];
  CHECK_VEC_EQ(actual, expected);
}

// --------------------------------------------------------------------------
// pipeline_reduce
// --------------------------------------------------------------------------

TEST(pipeline, reduce_sum_of_product) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -20, 20);
  std::vector<T> b = tf::random_vector<T>(n, -20, 20);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

  auto future = da.pipeline_reduce(
      {OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_MUL, OP_SUM}, {db});
  int64_t actual = (int64_t)future.get();

  int64_t expected = 0;
  for (size_t i = 0; i < n; ++i) expected += (int64_t)a[i] * b[i];
  CHECK_EQ(actual, expected);
}

TEST(pipeline, reduce_max_of_absolute) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -500, 500);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  auto future = da.pipeline_reduce({OP_PUSH_INPUT, OP_ABS, OP_MAX});
  int64_t actual = (int64_t)future.get();

  T expected = a[0] < 0 ? -a[0] : a[0];
  for (T x : a) {
    T v = x < 0 ? -x : x;
    if (v > expected) expected = v;
  }
  CHECK_EQ(actual, (int64_t)expected);
}

// Two chains over different primaries sharing one operand: the linreg gradient
// shape, submitted so they can share a pass.
TEST(pipeline, two_reduce_chains_agree_with_host) {
  const size_t n = 2048;
  std::vector<T> x = tf::random_vector<T>(n, 0, 9);
  std::vector<T> y = tf::random_vector<T>(n, 0, 9);
  std::vector<T> e = tf::constant_vector<T>(n, 1);
  dpu_vector<T> dx = dpu_vector<T>::from_cpu(x);
  dpu_vector<T> dy = dpu_vector<T>::from_cpu(y);
  dpu_vector<T> de = dpu_vector<T>::from_cpu(e);
  tf::drain();

  std::vector<uint8_t> ops = {OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_MUL, OP_SUM};
  StatsSnapshot k = tf::measure([&] {
    auto gx = dx.pipeline_reduce(ops, {de});
    auto gy = dy.pipeline_reduce(ops, {de});

    int64_t got_x = (int64_t)gx.get();
    int64_t got_y = (int64_t)gy.get();

    int64_t want_x = 0, want_y = 0;
    for (size_t i = 0; i < n; ++i) {
      want_x += (int64_t)x[i] * e[i];
      want_y += (int64_t)y[i] * e[i];
    }
    CHECK_EQ(got_x, want_x);
    CHECK_EQ(got_y, want_y);
  });
  CHECK_KERNELS_LE(k, 2);
}

// --------------------------------------------------------------------------
// dpu_pipeline_expr builder
// --------------------------------------------------------------------------

TEST(pipeline, expr_builder_matches_hand_written_rpn) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -20, 20);
  std::vector<T> b = tf::random_vector<T>(n, -20, 20);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  tf::drain();

  // sum((a - b) squared)
  expr diff = expr::input() - expr::operand(0);
  auto built = da.pipeline_reduce(diff.sqr().sum(), {db});
  int64_t from_builder = (int64_t)built.get();

  auto hand = da.pipeline_reduce(
      {OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_SUB, OP_DUP, OP_MUL, OP_SUM}, {db});
  int64_t from_rpn = (int64_t)hand.get();

  int64_t expected = 0;
  for (size_t i = 0; i < n; ++i) {
    int64_t d = (int64_t)a[i] - b[i];
    expected += d * d;
  }
  CHECK_EQ(from_builder, expected);
  CHECK_EQ(from_rpn, expected);
}

TEST(pipeline, expr_builder_scalar_ops) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -20, 20);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  // sum((a * 3) + 1)
  expr scaled = (expr::input() * (T)3) + (T)1;
  int64_t actual = (int64_t)da.pipeline_reduce(scaled.sum()).get();

  int64_t expected = 0;
  for (size_t i = 0; i < n; ++i) expected += (int64_t)a[i] * 3 + 1;
  CHECK_EQ(actual, expected);
}

TEST(pipeline, expr_builder_select) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -20, 20);
  std::vector<T> b = tf::random_vector<T>(n, -20, 20);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  tf::drain();

  // sum(min(a, b)) expressed as a select on a < b
  expr lhs = expr::input();
  expr rhs = expr::operand(0);
  expr smaller = (lhs < rhs).select(lhs, rhs);
  int64_t actual = (int64_t)da.pipeline_reduce(smaller.sum(), {db}).get();

  int64_t expected = 0;
  for (size_t i = 0; i < n; ++i) expected += a[i] < b[i] ? a[i] : b[i];
  CHECK_EQ(actual, expected);
}

#endif  // PIPELINE

// --------------------------------------------------------------------------
// JIT-only surface
// --------------------------------------------------------------------------

#if JIT

TEST(jit, explicit_jit_chain) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -100, 100);
  std::vector<T> b = tf::random_vector<T>(n, -100, 100);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

  std::vector<uint8_t> ops = {
      OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_ADD, OP_PUSH_INPUT,
      OP_SUB,        OP_NEGATE,         OP_ABS};
  std::vector<T> actual = da.jit(ops, {db}).vec.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) {
    T v = -((a[i] + b[i]) - a[i]);
    expected[i] = v < 0 ? -v : v;
  }
  CHECK_VEC_EQ(actual, expected);
}

// A repeated RPN signature must hit the kernel-object cache, not the compiler.
TEST(jit, identical_signature_hits_the_kernel_cache) {
  const size_t n = 1024;
  std::vector<T> a = tf::random_vector<T>(n, -100, 100);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  tf::drain();

  // Unlikely to have been compiled by an earlier test.
  std::vector<uint8_t> ops = {
      OP_PUSH_INPUT, OP_ADD_SCALAR, 77, 0, 0, 0, OP_MUL_SCALAR, 3, 0, 0, 0,
      OP_NEGATE};

  StatsSnapshot first = tf::measure([&] { (void)da.jit(ops).vec.to_cpu(); });
  StatsSnapshot second = tf::measure([&] { (void)da.jit(ops).vec.to_cpu(); });

  // Whether the first call compiles depends on what ran before; the second must
  // not compile anything new.
  CHECK_EQ(second.jit_kernel_compiles, 0u);
  (void)first;
}

TEST(jit, cached_kernel_returns_same_values) {
  const size_t n = 1024;
  std::vector<T> a = tf::random_vector<T>(n, -100, 100);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  tf::drain();

  std::vector<uint8_t> ops = {OP_PUSH_INPUT, OP_PUSH_INPUT, OP_ADD};
  std::vector<T> first = da.jit(ops).vec.to_cpu();
  std::vector<T> second = da.jit(ops).vec.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = a[i] + a[i];
  CHECK_VEC_EQ(first, expected);
  CHECK_VEC_EQ(second, expected);
}

#if JIT_PIPELINE_FALLBACK
TEST(jit, pipeline_fallback_validates_interpreter_contract) {
  CHECK(detail::pipeline_can_interpret(
      {OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_ADD}));
  CHECK(detail::pipeline_can_interpret({OP_PUSH_INPUT, OP_MAX}));
  CHECK(detail::pipeline_can_interpret(
      {OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_ARGMIN_K, 2}));

  CHECK(!detail::pipeline_can_interpret({OP_PUSH_INDEX}));
  CHECK(!detail::pipeline_can_interpret({OP_PUSH_INPUT, OP_MAX, OP_ABS}));
  CHECK(!detail::pipeline_can_interpret({OP_PUSH_INPUT, OP_ADD_SCALAR, 1}));
}

TEST(jit, pipeline_runs_while_kernel_compiles) {
  const size_t n = 1024;
  std::vector<T> input = tf::constant_vector<T>(n, 9);
  dpu_vector<T> values = dpu_vector<T>::from_cpu(input);
  tf::drain();

  StatsSnapshot before = RuntimeStats::get().snapshot();
  std::vector<T> actual = (-((values + (T)91) * (T)3)).to_cpu();
  T maximum_error = max(abs(values - (T)2)).get();
  StatsSnapshot delta = RuntimeStats::get().snapshot() - before;

  CHECK_VEC_EQ(actual, tf::constant_vector<T>(n, -300));
  CHECK_EQ(maximum_error, 7);
  CHECK_GT(delta.jit_pipeline_fallbacks, 0u);
}

TEST(jit, eager_runs_when_interpreter_cannot) {
  const size_t n = 1024;
  std::vector<T> host_input = tf::constant_vector<T>(n, 9);
  dpu_vector<T> input = dpu_vector<T>::from_cpu(host_input);
  dpu_vector<T> output(n, 0, true);
  output.data_desc_ref()->type_name = typeid(T).name();
  tf::drain();

  auto event = std::make_shared<Event>(
      Event::OperationType::COMPUTE,
      std::bind(detail::internal_launch_unary, output.data_desc_ref(),
                input.data_desc_ref(), OpInfo<T>::negate));
  event->inputs = {input.data_desc_ref()};
  event->output = output.data_desc_ref();
  event->rpn_ops = {OP_PUSH_INDEX};
  event->kid = OpInfo<T>::negate;
  event->pipeline_kid = OpInfo<T>::universal_pipeline;

  StatsSnapshot before = RuntimeStats::get().snapshot();
  DpuRuntime::get().get_event_queue().submit(event);
  std::vector<T> actual = output.to_cpu();
  StatsSnapshot delta = RuntimeStats::get().snapshot() - before;

  CHECK_VEC_EQ(actual, tf::constant_vector<T>(n, -9));
  CHECK_GT(delta.jit_eager_fallbacks, 0u);
}
#endif

// If the signature hash ignored part of the program, the second call would
// reuse the first kernel and return its answer.
TEST(jit, distinct_signatures_do_not_collide) {
  const size_t n = 1024;
  std::vector<T> a = tf::constant_vector<T>(n, 10);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(a);
  tf::drain();

  std::vector<T> plus =
      da.jit({OP_PUSH_INPUT, OP_ADD_SCALAR, 5, 0, 0, 0}).vec.to_cpu();
  std::vector<T> minus =
      db.jit({OP_PUSH_INPUT, OP_SUB_SCALAR, 5, 0, 0, 0}).vec.to_cpu();

  CHECK_VEC_EQ(plus, tf::constant_vector<T>(n, 15));
  CHECK_VEC_EQ(minus, tf::constant_vector<T>(n, 5));
}

// transform() builds RPN from a lambda over expression objects.
TEST(jit, transform_lambda) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -20, 20);
  std::vector<T> b = tf::random_vector<T>(n, -20, 20);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  tf::drain();

  std::vector<T> actual =
      da.transform(
            [](const std::vector<dpu_pipeline_expr<T>>& in) {
              return (in[0] + in[1]) * (T)2;
            },
            {db})
          .vec.to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) expected[i] = (a[i] + b[i]) * 2;
  CHECK_VEC_EQ(actual, expected);
}

TEST(jit, reduce_lambda) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -20, 20);
  std::vector<T> b = tf::random_vector<T>(n, -20, 20);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  tf::drain();

  auto future = da.reduce(
      [](const std::vector<dpu_pipeline_expr<T>>& in) {
        dpu_pipeline_expr<T> diff = in[0] - in[1];
        return diff.sqr().sum();
      },
      {db});
  int64_t actual = (int64_t)future.get();

  int64_t expected = 0;
  for (size_t i = 0; i < n; ++i) {
    int64_t d = (int64_t)a[i] - b[i];
    expected += d * d;
  }
  CHECK_EQ(actual, expected);
}

// argmin/argmax over K whole vectors: the per-element winning lane index.
TEST(jit, argmin_over_vectors) {
  const size_t n = 1024;
  const size_t lanes = 4;
  std::vector<std::vector<T>> host;
  std::vector<dpu_vector<T>> vecs;
  for (size_t i = 0; i < lanes; ++i) {
    host.push_back(tf::random_vector<T>(n, 0, 50));
    vecs.push_back(dpu_vector<T>::from_cpu(host.back()));
  }
  tf::drain();

  std::vector<T> actual = argmin(vecs).to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) {
    size_t best = 0;
    for (size_t l = 1; l < lanes; ++l)
      if (host[l][i] < host[best][i]) best = l;
    expected[i] = (T)best;
  }
  CHECK_VEC_EQ(actual, expected);
}

TEST(jit, argmax_over_vectors) {
  const size_t n = 1024;
  const size_t lanes = 4;
  std::vector<std::vector<T>> host;
  std::vector<dpu_vector<T>> vecs;
  for (size_t i = 0; i < lanes; ++i) {
    host.push_back(tf::random_vector<T>(n, 0, 50));
    vecs.push_back(dpu_vector<T>::from_cpu(host.back()));
  }
  tf::drain();

  std::vector<T> actual = argmax(vecs).to_cpu();

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) {
    size_t best = 0;
    for (size_t l = 1; l < lanes; ++l)
      if (host[l][i] > host[best][i]) best = l;
    expected[i] = (T)best;
  }
  CHECK_VEC_EQ(actual, expected);
}

// Both arg ops use strict comparison, so ties go to the lowest lane.
TEST(jit, argmin_ties_pick_lowest_lane) {
  const size_t n = 256;
  std::vector<std::vector<T>> host(3, tf::constant_vector<T>(n, 7));
  std::vector<dpu_vector<T>> vecs;
  for (size_t i = 0; i < host.size(); ++i)
    vecs.push_back(dpu_vector<T>::from_cpu(host[i]));
  tf::drain();

  CHECK_VEC_EQ(argmin(vecs).to_cpu(), tf::constant_vector<T>(n, 0));
}

#endif  // JIT

#if PIPELINE
// --------------------------------------------------------------------------
// absorbed-intermediate inlining vs. inline scalar immediates
// --------------------------------------------------------------------------

// A consumer whose RPN carries an inline immediate, reading an intermediate
// that fusion could absorb.
//
// Today this stays two kernel passes -- absorption declines the shape -- and
// the values are correct.  The test guards a latent hazard rather than a live
// bug: the rewrite loop in EventQueue::expand_absorbed_inputs handles
// OP_*_SCALAR_VAR (a one-byte slot index) but has no OP_INLINE_BYTES case, so
// if absorption ever starts firing here the four immediate bytes of an
// OP_*_SCALAR would be re-read as opcodes and mangle the program.
TEST(pipeline, absorbed_input_with_inline_scalar_immediate) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 10);
  std::vector<T> b = tf::constant_vector<T>(n, 4);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  tf::drain();

  // mid = a + b = 14, then res = mid + 7 = 21, with the +7 as an immediate.
  dpu_vector<T> mid = da + db;
  std::vector<T> actual =
      mid.pipeline({OP_PUSH_INPUT, OP_ADD_SCALAR, 7, 0, 0, 0}).vec.to_cpu();

  CHECK_VEC_EQ(actual, tf::constant_vector<T>(n, 21));
}

// The same shape with a scalar *variable*, which absorption does fuse (one
// kernel pass): the consumer's slot index must shift past the producer's
// scalar table, or the consumer reads the producer's constant instead.
TEST(pipeline, absorbed_input_with_scalar_variable) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 10);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  tf::drain();

  // mid = a * s0 (s0 = 3) = 30, then res = mid + s0 (s0 = 100) = 130.
  dpu_vector<T> mid =
      da.pipeline({OP_PUSH_INPUT, OP_MUL_SCALAR_VAR, 0}, {}, {3u}).vec;
  std::vector<T> actual =
      mid.pipeline({OP_PUSH_INPUT, OP_ADD_SCALAR_VAR, 0}, {}, {100u})
          .vec.to_cpu();

  CHECK_VEC_EQ(actual, tf::constant_vector<T>(n, 130));
}
#endif  // PIPELINE
