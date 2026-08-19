#pragma once

#include <common.h>
#include <config.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include "jit.h"
#include "kernelids.h"
#include "opinfo.h"
#include "runtime.h"
#include "timer.h"
#include "vectordesc.h"

#if __cplusplus < 202002L
// Fake source_location for pre-C++20
namespace std {
struct source_location {
  constexpr source_location(const char* file = "unknown", int line = 0,
                            int column = 0,
                            const char* function = "unknown")
      : file_(file), line_(line), column_(column), function_(function) {}
  static source_location current() { return {}; }
  constexpr const char* file_name() const { return file_; }
  constexpr int line() const { return line_; }
  constexpr int column() const { return column_; }
  constexpr const char* function_name() const { return function_; }

 private:
  const char* file_;
  int line_;
  int column_;
  const char* function_;
};
};  // namespace std
#define VECTORDPU_SOURCE_LOCATION \
  std::source_location(__FILE__, __LINE__, 0, __func__)
#else
#include <source_location>
#define VECTORDPU_SOURCE_LOCATION std::source_location::current()
#endif

using std::vector;

#define LOGGER_ARGS_WITH_DEFAULTS \
  std::string_view name = "",     \
                   std::source_location loc = std::source_location::current()

// Forward declarations
void dpu_fence();

template <typename T>
class dpu_vector;

template <typename T>
struct lazy_reduction_result;

template <typename T>
struct reduction_batch_result_state;

template <typename T>
class dpu_reduction_batch;

template <typename T>
class dpu_local_vector;

template <typename T>
class dpu_pipeline_context;

#if PIPELINE
template <typename T>
class dpu_pipeline_expr;

template <typename T>
using dpu_expr = dpu_pipeline_expr<T>;
#endif

namespace detail {
struct jit_recorder {
  std::vector<uint8_t> rpn;
  std::vector<detail::VectorDescRef> operands;
  std::vector<detail::VectorDescRef> locals;

  uint8_t add_operand(detail::VectorDescRef d) {
    for (size_t i = 0; i < operands.size(); i++)
      if (operands[i] == d) return (uint8_t)i;
    operands.push_back(d);
    return (uint8_t)(operands.size() - 1);
  }
  uint8_t add_local(detail::VectorDescRef d) {
    for (size_t i = 0; i < locals.size(); i++)
      if (locals[i] == d) return (uint8_t)i;
    locals.push_back(d);
    return (uint8_t)(locals.size() - 1);
  }
};

struct jit_index_expr {
  jit_recorder& rec;
};
template <typename T>
struct jit_indirect_load_expr {
  const dpu_vector<T>& vec;
  jit_recorder& rec;

  jit_indirect_load_expr operator*(T rhs) const;
  jit_indirect_load_expr operator+(T rhs) const;
  jit_indirect_load_expr operator-(T rhs) const;
};
template <typename T>
struct jit_indirect_ref_expr {
  dpu_local_vector<T>& vec;
  jit_recorder& rec;

  void operator++(int);
  void apply(T value);
  void apply(const jit_indirect_load_expr<T>& value);
};
}  // namespace detail

inline detail::jit_index_expr dpu_index(detail::jit_recorder& rec) {
  return {rec};
}

#if PIPELINE
template <typename T>
struct pipeline_result;
#endif

// ============================
// DPU Vector
// ============================

template <typename T>
struct reduction_result {
  using type = T;
};

#if ENABLE_PROMOTION_REDUCTIONS
template <>
struct reduction_result<int32_t> {
  using type = int64_t;
};
#endif

template <typename T>
class dpu_vector {
 public:
  dpu_vector() noexcept;
  dpu_vector(size_t n, uint32_t reserved = 0, bool lazy = false,
             LOGGER_ARGS_WITH_DEFAULTS);

  ~dpu_vector();

  dpu_vector(const dpu_vector& other);                 // copy constructor
  dpu_vector(dpu_vector&& other) noexcept;             // move constructor
  dpu_vector& operator=(const dpu_vector& other);      // copy assignment
  dpu_vector& operator=(dpu_vector&& other) noexcept;  // move assignment

  vector<T> to_cpu();

  static dpu_vector<T> from_cpu(std::vector<T>& cpu_vec,
                                LOGGER_ARGS_WITH_DEFAULTS);
  void add_fence();
  dpu_vector<T>& operator+=(const dpu_vector<T>& other);
  dpu_vector<T>& operator-=(const dpu_vector<T>& other);
  dpu_vector<T>& operator*=(const dpu_vector<T>& other);
  dpu_vector<T>& operator/=(const dpu_vector<T>& other);

  dpu_vector<T>& operator+=(T scalar);
  dpu_vector<T>& operator-=(T scalar);
  dpu_vector<T>& operator*=(T scalar);
  dpu_vector<T>& operator/=(T scalar);
  dpu_vector<T>& operator>>=(T scalar);

  dpu_vector<T> operator-() const;
  dpu_vector<T> operator==(T scalar) const;

#if PIPELINE
  dpu_vector<T>& operator=(const pipeline_result<T>& other);
#endif

  const detail::VectorDesc& data_desc() const { return *data_; }
  detail::VectorDescRef data_desc_ref() const { return data_; }

  size_t size() const { return size_; }
  uint32_t reserved() const { return reserved_; }

 private:
  detail::VectorDescRef data_;
  size_t size_;
  uint32_t reserved_ = 0;
  const char* debug_name = nullptr;

  const char* debug_file = nullptr;
  int debug_line = -1;
  mutable bool copied = false;

  static std::vector<uint8_t> prepare_rpn(const std::vector<uint8_t>& ops);

 public:
  using reduction_result_t = typename reduction_result<T>::type;

#if PIPELINE
  pipeline_result<T> pipeline(const std::vector<uint8_t>& ops);
  pipeline_result<T> pipeline(const std::vector<uint8_t>& ops,
                              const std::vector<dpu_vector<T>>& operands,
                              const std::vector<uint32_t>& scalars = {});
  lazy_reduction_result<T> pipeline_reduce(
      const std::vector<uint8_t>& ops,
      const std::vector<dpu_vector<T>>& operands = {},
      const std::vector<uint32_t>& scalars = {});
  lazy_reduction_result<T> pipeline_reduce(
      const dpu_pipeline_expr<T>& expr,
      const std::vector<dpu_vector<T>>& operands = {},
      const std::vector<uint32_t>& scalars = {});
  dpu_reduction_batch<T> reduction_batch();
#endif
#if JIT
  pipeline_result<T> jit(const std::vector<uint8_t>& ops);
  pipeline_result<T> jit(const std::vector<uint8_t>& ops,
                         const std::vector<dpu_vector<T>>& operands,
                         const std::vector<uint32_t>& scalars = {});

  detail::jit_indirect_load_expr<T> operator[](
      detail::jit_index_expr idx) const;
  template <typename F>
  pipeline_result<T> transform(F&& build,
                               const std::vector<dpu_vector<T>>& operands = {},
                               const std::vector<uint32_t>& scalars = {});
  template <typename F>
  lazy_reduction_result<T> reduce(
      F&& build, const std::vector<dpu_vector<T>>& operands = {},
      const std::vector<uint32_t>& scalars = {});
#endif
};

template <typename T>
struct lazy_reduction_result {
  dpu_vector<T> vec;
  KernelID rid = 0;
  std::shared_ptr<reduction_batch_result_state<T>> batch_state;
  size_t batch_index = 0;
  lazy_reduction_result() noexcept = default;
  lazy_reduction_result(dpu_vector<T> v, KernelID r)
      : vec(std::move(v)), rid(r) {}
  lazy_reduction_result(dpu_vector<T> v, KernelID r,
                        std::shared_ptr<reduction_batch_result_state<T>> state,
                        size_t index)
      : vec(std::move(v)),
        rid(r),
        batch_state(std::move(state)),
        batch_index(index) {}
  typename dpu_vector<T>::reduction_result_t get();
  operator typename dpu_vector<T>::reduction_result_t() { return get(); }
#if ENABLE_PROMOTION_REDUCTIONS
  operator T() { return (T)get(); }
#endif
};

template <typename T>
using dpu_future = lazy_reduction_result<T>;

// A collection of lazy DPU results. Keeping the producers queued until get()
// lets independent computations become horizontal fusion chains.
template <typename T>
class dpu_future_vector {
 public:
  using future_type = dpu_future<T>;
  using result_type = typename dpu_vector<T>::reduction_result_t;

  void reserve(size_t count) { futures_.reserve(count); }
  void push_back(future_type future) {
    futures_.push_back(std::move(future));
  }

  size_t size() const { return futures_.size(); }
  bool empty() const { return futures_.empty(); }

  std::vector<result_type> get() {
    std::vector<result_type> results;
    results.reserve(futures_.size());
    if (!futures_.empty()) dpu_fence();
    for (auto& future : futures_) results.push_back(future.get());
    return results;
  }

 private:
  std::vector<future_type> futures_;
};

#if PIPELINE
template <typename T>
class dpu_reduction_batch {
 public:
  explicit dpu_reduction_batch(dpu_vector<T>& primary);
  dpu_reduction_batch(const dpu_reduction_batch&) = delete;
  dpu_reduction_batch& operator=(const dpu_reduction_batch&) = delete;
  dpu_reduction_batch(dpu_reduction_batch&&) noexcept = default;
  dpu_reduction_batch& operator=(dpu_reduction_batch&&) noexcept = default;

  template <typename F>
  lazy_reduction_result<T> add(
      F&& build, const std::vector<dpu_vector<T>>& operands = {},
      const std::vector<uint32_t>& scalars = {});
  void submit();

 private:
  struct Entry {
    std::vector<uint8_t> ops;
    std::vector<detail::VectorDescRef> operands;
    std::vector<uint32_t> scalars;
    detail::VectorDescRef output;
    KernelID rid = OpInfo<T>::sum;
  };

  dpu_vector<T>* primary_;
  std::vector<Entry> entries_;
  std::shared_ptr<reduction_batch_result_state<T>> result_state_;

  static KernelID reduction_id(const std::vector<uint8_t>& ops);
};
#endif

enum class dpu_local_reduce_op : uint8_t {
  sum,
  product,
  min,
  max,
};

#if PIPELINE
template <typename T>
class dpu_pipeline_expr {
 public:
  dpu_pipeline_expr() = default;
  explicit dpu_pipeline_expr(std::vector<uint8_t> ops) : ops_(std::move(ops)) {}

  static dpu_pipeline_expr input() {
    return dpu_pipeline_expr({(uint8_t)OP_PUSH_INPUT});
  }

  static dpu_pipeline_expr operand(uint8_t idx) {
    return dpu_pipeline_expr({(uint8_t)(OP_PUSH_OPERAND_0 + idx)});
  }

  static dpu_pipeline_expr scalar(T value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T) < 4 ? sizeof(T) : 4);
    return dpu_pipeline_expr({
        (uint8_t)OP_PUSH_SCALAR,
        (uint8_t)(bits & 0xFF),
        (uint8_t)((bits >> 8) & 0xFF),
        (uint8_t)((bits >> 16) & 0xFF),
        (uint8_t)((bits >> 24) & 0xFF),
    });
  }

  static dpu_pipeline_expr scalar_var(uint8_t idx) {
    return dpu_pipeline_expr({
        (uint8_t)OP_PUSH_SCALAR_VAR,
        idx,
    });
  }

  dpu_pipeline_expr dup() const { return append(OP_DUP); }
  dpu_pipeline_expr sqr() const { return dup().append(OP_MUL); }
  dpu_pipeline_expr min() const { return append(OP_MIN); }
  dpu_pipeline_expr max() const { return append(OP_MAX); }
  dpu_pipeline_expr sum() const { return append(OP_SUM); }
  dpu_pipeline_expr product() const { return append(OP_PRODUCT); }

  dpu_pipeline_expr operator+(T rhs) const {
    return append_scalar_op(OP_ADD_SCALAR, rhs);
  }
  dpu_pipeline_expr operator-(T rhs) const {
    return append_scalar_op(OP_SUB_SCALAR, rhs);
  }
  dpu_pipeline_expr operator*(T rhs) const {
    return append_scalar_op(OP_MUL_SCALAR, rhs);
  }
	  dpu_pipeline_expr operator/(T rhs) const {
	    return append_scalar_op(OP_DIV_SCALAR, rhs);
	  }
	  dpu_pipeline_expr operator==(T rhs) const {
	    return append_scalar_op(OP_EQ_SCALAR, rhs);
	  }

  const std::vector<uint8_t>& ops() const { return ops_; }

  dpu_pipeline_expr operator+(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_ADD);
  }
  dpu_pipeline_expr operator-(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_SUB);
  }
  dpu_pipeline_expr operator<(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_LT);
  }
  dpu_pipeline_expr operator*(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_MUL);
  }
  dpu_pipeline_expr operator/(const dpu_pipeline_expr& rhs) const {
    return combine(rhs, OP_DIV);
  }

  dpu_pipeline_expr select(const dpu_pipeline_expr& then_expr,
                           const dpu_pipeline_expr& else_expr) const {
    dpu_pipeline_expr out;
    out.ops_.reserve(ops_.size() + then_expr.ops_.size() +
                     else_expr.ops_.size() + 1);
    out.ops_.insert(out.ops_.end(), ops_.begin(), ops_.end());
    out.ops_.insert(out.ops_.end(), then_expr.ops_.begin(),
                    then_expr.ops_.end());
    out.ops_.insert(out.ops_.end(), else_expr.ops_.begin(),
                    else_expr.ops_.end());
    out.ops_.push_back(OP_SELECT);
    return out;
  }

 private:
  std::vector<uint8_t> ops_;

  dpu_pipeline_expr append(uint8_t op) const {
    auto out = *this;
    out.ops_.push_back(op);
    return out;
  }

  dpu_pipeline_expr append_scalar_op(uint8_t op, T value) const {
    auto out = *this;
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T) < 4 ? sizeof(T) : 4);
    out.ops_.push_back(op);
    out.ops_.push_back((uint8_t)(bits & 0xFF));
    out.ops_.push_back((uint8_t)((bits >> 8) & 0xFF));
    out.ops_.push_back((uint8_t)((bits >> 16) & 0xFF));
    out.ops_.push_back((uint8_t)((bits >> 24) & 0xFF));
    return out;
  }

  dpu_pipeline_expr combine(const dpu_pipeline_expr& rhs, uint8_t op) const {
    // Keep scalar variables as scalar operations instead of materializing
    // them as a third stack value. Besides shortening the RPN, this lets the
    // generic interpreter evaluate accumulator + column * scalar_var with its
    // two-vector stack.
    if (rhs.ops_.size() == 2 && rhs.ops_[0] == OP_PUSH_SCALAR_VAR &&
        op >= OP_ADD && op <= OP_LE) {
      auto out = *this;
      out.ops_.push_back(
          (uint8_t)(op + (OP_ADD_SCALAR_VAR - OP_ADD)));
      out.ops_.push_back(rhs.ops_[1]);
      return out;
    }

    dpu_pipeline_expr out;
    out.ops_.reserve(ops_.size() + rhs.ops_.size() + 1);
    out.ops_.insert(out.ops_.end(), ops_.begin(), ops_.end());
    out.ops_.insert(out.ops_.end(), rhs.ops_.begin(), rhs.ops_.end());
    out.ops_.push_back(op);
    return out;
  }
};

// Horizontal argmin/argmax over K lanes, evaluated per element. `.label` is
// the winning lane index (lowers to the variadic OP_ARGMIN_K/OP_ARGMAX_K op);
// `.value` is the winning value (an elementwise min/max select-chain, since no
// dedicated min-of-vectors op exists yet). Both use a strict comparison so
// ties keep the lowest lane index. kmeans reads `.label` to drive the
// centroid scatter-add; inertia/convergence reads `.value`.
template <typename T>
struct dpu_arg_expr {
  dpu_pipeline_expr<T> label;
  dpu_pipeline_expr<T> value;
};

template <typename T>
dpu_arg_expr<T> detail_arg_k_lanes(
    const std::vector<dpu_pipeline_expr<T>>& lanes, uint8_t arg_op) {
  std::vector<uint8_t> label_ops;
  for (const auto& lane : lanes) {
    const auto& o = lane.ops();
    label_ops.insert(label_ops.end(), o.begin(), o.end());
  }
  label_ops.push_back(arg_op);
  label_ops.push_back((uint8_t)lanes.size());

  const bool is_min = (arg_op == (uint8_t)OP_ARGMIN_K);
  dpu_pipeline_expr<T> best = lanes[0];
  for (size_t j = 1; j < lanes.size(); ++j) {
    dpu_pipeline_expr<T> take = is_min ? (lanes[j] < best) : (best < lanes[j]);
    best = take.select(lanes[j], best);
  }
  return {dpu_pipeline_expr<T>(std::move(label_ops)), best};
}

// (a) over K candidate EXPRESSIONS, per element (used inside transform /
//     reduce / dpu_jit_foreach lambdas).
template <typename T>
dpu_arg_expr<T> argmin(const std::vector<dpu_pipeline_expr<T>>& lanes) {
  return detail_arg_k_lanes(lanes, (uint8_t)OP_ARGMIN_K);
}
template <typename T>
dpu_arg_expr<T> argmax(const std::vector<dpu_pipeline_expr<T>>& lanes) {
  return detail_arg_k_lanes(lanes, (uint8_t)OP_ARGMAX_K);
}

// (b) over K whole VECTORS, by itself -- launches its own fused kernel and
//     returns the per-element winning-lane-index vector.
template <typename T>
dpu_vector<T> argmin(std::vector<dpu_vector<T>>& lanes) {
  std::vector<dpu_vector<T>> operands(lanes.begin() + 1, lanes.end());
  return lanes[0]
      .transform(
          [](const std::vector<dpu_pipeline_expr<T>>& x) {
            return argmin(std::vector<dpu_pipeline_expr<T>>(x.begin(), x.end()))
                .label;
          },
          operands)
      .vec;
}
template <typename T>
dpu_vector<T> argmax(std::vector<dpu_vector<T>>& lanes) {
  std::vector<dpu_vector<T>> operands(lanes.begin() + 1, lanes.end());
  return lanes[0]
      .transform(
          [](const std::vector<dpu_pipeline_expr<T>>& x) {
            return argmax(std::vector<dpu_pipeline_expr<T>>(x.begin(), x.end()))
                .label;
          },
          operands)
      .vec;
}

template <typename T>
lazy_reduction_result<T> min_squared_distance(std::vector<dpu_vector<T>>& cols,
                                              const std::vector<T>& query);
#endif

#if PIPELINE
template <typename T>
struct pipeline_result {
  dpu_vector<T> vec;
  pipeline_result(dpu_vector<T> v) : vec(std::move(v)) {}
  operator T();
  operator int64_t();
  operator dpu_vector<T>() { return std::move(vec); }
  dpu_vector<T>* operator->() { return &vec; }
  operator lazy_reduction_result<T>() {
    return lazy_reduction_result<T>(std::move(vec),
                                    vec.data_desc().reduction_rid);
  }
};
#endif

template <typename T>
lazy_reduction_result<T> sum(const dpu_vector<T>& a);
template <typename T>
lazy_reduction_result<T> product(const dpu_vector<T>& a);
template <typename T>
lazy_reduction_result<T> min(const dpu_vector<T>& a);
template <typename T>
lazy_reduction_result<T> max(const dpu_vector<T>& a);
template <typename T>
dpu_vector<T> operator<(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs);
template <typename T>
dpu_vector<T> select(const dpu_vector<T>& cond, const dpu_vector<T>& then_vec,
                     const dpu_vector<T>& else_vec);

template <typename T>
class dpu_local_vector {
 public:
  dpu_local_vector(uint32_t n, LOGGER_ARGS_WITH_DEFAULTS);
  dpu_local_vector(uint32_t n, dpu_local_reduce_op reduce_op,
                   LOGGER_ARGS_WITH_DEFAULTS);
  ~dpu_local_vector() = default;

  detail::jit_indirect_ref_expr<T> operator[](
      const detail::jit_indirect_load_expr<T>& idx);

  vector<T> to_cpu();
  dpu_local_reduce_op reduce_op() const { return reduce_op_; }

  const detail::VectorDesc& data_desc() const { return *data_; }
  detail::VectorDescRef data_desc_ref() const { return data_; }
  uint32_t size() const { return size_; }

 private:
  detail::VectorDescRef data_;
  uint32_t size_;
  dpu_local_reduce_op reduce_op_ = dpu_local_reduce_op::sum;
};

#if PIPELINE
template <typename T>
class dpu_pipeline_context {
 public:
  void local_reduce(dpu_local_vector<T>& local,
                    const dpu_pipeline_expr<T>& index,
                    const dpu_pipeline_expr<T>& value);
  void local_reduce(dpu_local_vector<T>& local,
                    const dpu_pipeline_expr<T>& index, T value) {
    local_reduce(local, index, dpu_pipeline_expr<T>::scalar(value));
  }
  void local_sum(dpu_local_vector<T>& local,
                 const dpu_pipeline_expr<T>& index,
                 const dpu_pipeline_expr<T>& value) {
    local_reduce(local, index, value);
  }
  void local_sum(dpu_local_vector<T>& local,
                 const dpu_pipeline_expr<T>& index, T value) {
    local_reduce(local, index, value);
  }

  std::vector<uint8_t> materialize_ops() const;
  const std::vector<detail::VectorDescRef>& locals() const { return locals_; }

 private:
  struct LocalReduce {
    uint8_t local_id;
    uint8_t reduce_op;
    std::vector<uint8_t> index_ops;
    std::vector<uint8_t> value_ops;
  };

  std::vector<uint8_t> ops_;
  std::vector<detail::VectorDescRef> locals_;
  std::vector<LocalReduce> local_reductions_;

  uint8_t local_id(dpu_local_vector<T>& local);
};
#endif

template <typename T, typename F>
void dpu_jit_foreach(size_t n, F f);

#if JIT
template <typename T, typename F>
void dpu_jit_foreach(dpu_vector<T>& primary,
                     const std::vector<dpu_vector<T>>& operands,
                     const std::vector<uint32_t>& scalars, F f);
#endif

namespace detail {
void vec_xfer_from_dpu_span(char* cpu, VectorDescRef base, size_t span_bytes);

void launch_binary(VectorDescRef res, VectorDescRef lhs, VectorDescRef rhs,
                   KernelID kernel_id, uint8_t opcode, KernelID pipeline_kid);
void launch_binary_scalar(VectorDescRef res, VectorDescRef lhs, uint32_t scalar,
                          KernelID kernel_id, uint8_t opcode,
                          KernelID pipeline_kid);
void launch_unary(VectorDescRef res, VectorDescRef rhs, KernelID kernel_id,
                  uint8_t opcode, KernelID pipeline_kid);
void launch_reduction(VectorDescRef buf, VectorDescRef rhs, KernelID kernel_id,
                      uint8_t opcode, KernelID pipeline_kid);

void internal_launch_binary(VectorDescRef res, VectorDescRef lhs,
                            VectorDescRef rhs, KernelID kernel_id);
void internal_launch_unary(VectorDescRef res, VectorDescRef lhs,
                           KernelID kernel_id);
void internal_launch_reduction(VectorDescRef res, VectorDescRef rhs,
                               KernelID kernel_id);

#if PIPELINE
void launch_universal_pipeline(VectorDescRef res, VectorDescRef init,
                               const std::vector<uint8_t>& ops,
                               const std::vector<VectorDescRef>& operands,
                               KernelID kernel_id,
                               const std::vector<uint32_t>& scalars = {},
                               const std::vector<uint32_t>& extra_scalars = {},
                               const std::vector<VectorDescRef>& extra_outputs = {});

void internal_launch_universal_pipeline(
    VectorDescRef res, VectorDescRef init, const std::vector<uint8_t>& ops,
    const std::vector<VectorDescRef>& operands, KernelID kernel_id,
    const std::vector<uint32_t>& scalars,
    const std::vector<uint32_t>& extra_scalars = {},
    const std::vector<VectorDescRef>& extra_outputs = {},
    std::string_view kernel_hash = {});

void internal_launch_jit(const std::string& binary_path, VectorDescRef output,
                         const std::vector<VectorDescRef>& inputs,
                         const std::vector<uint8_t>& rpn_ops,
                         const std::vector<uint32_t>& extra_scalars = {},
                         const std::vector<VectorDescRef>& extra_outputs = {});
#endif
}  // namespace detail

#include "vectordpu.inl"
