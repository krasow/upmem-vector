#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "config.h"

namespace polymerpim {

class OutOfMemory : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

template <typename T>
class DPUVector;
template <typename T>
class DpuLazy;
template <typename T>
class DpuFuture;
template <typename T>
class DPULocalVector;

struct ArgResult {
  int32_t value;
  uint32_t index;
};

template <>
class DpuLazy<int32_t> {
 public:
  using value_type = int32_t;
  struct Impl;

  DpuLazy() = default;
  size_t size() const;
  std::vector<int32_t> to_cpu() const;

 private:
  explicit DpuLazy(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;

  friend class DPUVector<int32_t>;
  friend class DPULocalVector<int32_t>;
  friend class DpuFuture<int32_t>;
  friend DpuLazy operator+(const DpuLazy&, const DpuLazy&);
  friend DpuLazy operator-(const DpuLazy&, const DpuLazy&);
  friend DpuLazy operator*(const DpuLazy&, const DpuLazy&);
  friend DpuLazy operator/(const DpuLazy&, const DpuLazy&);
  friend DpuLazy operator<(const DpuLazy&, const DpuLazy&);
  friend DpuLazy operator==(const DpuLazy&, const DpuLazy&);
  friend DpuLazy operator+(const DpuLazy&, int32_t);
  friend DpuLazy operator-(const DpuLazy&, int32_t);
  friend DpuLazy operator*(const DpuLazy&, int32_t);
  friend DpuLazy operator/(const DpuLazy&, int32_t);
  friend DpuLazy operator<(const DpuLazy&, int32_t);
  friend DpuLazy operator==(const DpuLazy&, int32_t);
  friend DpuLazy operator>>(const DpuLazy&, int32_t);
  friend DpuLazy operator-(const DpuLazy&);
  friend DpuLazy abs(const DpuLazy&);
  friend DpuLazy sqr(const DpuLazy&);
  friend DpuLazy select(const DpuLazy&, const DpuLazy&, const DpuLazy&);
  friend DpuLazy argmin(const std::vector<DpuLazy>&);
  friend DpuLazy argmax(const std::vector<DpuLazy>&);
  friend DpuFuture<ArgResult> argmin(const DpuLazy&);
  friend DpuFuture<ArgResult> argmax(const DpuLazy&);
  friend DpuFuture<int32_t> sum(const DpuLazy&);
  friend DpuFuture<int32_t> product(const DpuLazy&);
  friend DpuFuture<int32_t> minimum(const DpuLazy&);
  friend DpuFuture<int32_t> maximum(const DpuLazy&);
};

template <>
class DPUVector<int32_t> {
 public:
  using value_type = int32_t;
#if ENABLE_PROMOTION_REDUCTIONS
  using reduction_result_t = int64_t;
#else
  using reduction_result_t = int32_t;
#endif
  struct Impl;

  DPUVector() = default;
  // `count` zeros (or `value`s) as a pending fill: nothing is allocated or
  // transferred, and a zero fill drops out of the first expression using it.
  explicit DPUVector(size_t count, std::string_view name = "");
  DPUVector(size_t count, int32_t value, std::string_view name = "");
  explicit DPUVector(std::vector<int32_t>& values, std::string_view name = "");
  DPUVector(int32_t* values, size_t count, std::string_view name = "");
  DPUVector(const DpuLazy<int32_t>& expression);

  // Records the expression rather than running it, so a chain of assignments
  // still fuses into one kernel; errors surface at the first read, not here.
  DPUVector& operator=(const DpuLazy<int32_t>& expression);
  DPUVector& operator+=(const DpuLazy<int32_t>& rhs);
  DPUVector& operator-=(const DpuLazy<int32_t>& rhs);
  DPUVector& operator*=(const DpuLazy<int32_t>& rhs);
  DPUVector& operator+=(int32_t rhs);
  DPUVector& operator-=(int32_t rhs);
  DPUVector& operator*=(int32_t rhs);

  std::vector<int32_t> to_cpu();
  size_t to_cpu_into(int32_t* output, size_t capacity);
  size_t size() const;
  operator DpuLazy<int32_t>() const;

 private:
  explicit DPUVector(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;

  friend class DpuLazy<int32_t>;
  friend class DPULocalVector<int32_t>;
  friend void fence(DPUVector&);
};

using Int32Vector = DPUVector<int32_t>;
using Int32Expression = DpuLazy<int32_t>;

DpuLazy<int32_t> operator+(const DpuLazy<int32_t>& lhs,
                           const DpuLazy<int32_t>& rhs);
DpuLazy<int32_t> operator-(const DpuLazy<int32_t>& lhs,
                           const DpuLazy<int32_t>& rhs);
DpuLazy<int32_t> operator*(const DpuLazy<int32_t>& lhs,
                           const DpuLazy<int32_t>& rhs);
DpuLazy<int32_t> operator/(const DpuLazy<int32_t>& lhs,
                           const DpuLazy<int32_t>& rhs);
DpuLazy<int32_t> operator<(const DpuLazy<int32_t>& lhs,
                           const DpuLazy<int32_t>& rhs);
DpuLazy<int32_t> operator==(const DpuLazy<int32_t>& lhs,
                            const DpuLazy<int32_t>& rhs);

DpuLazy<int32_t> operator+(const DpuLazy<int32_t>& expression, int32_t scalar);
DpuLazy<int32_t> operator-(const DpuLazy<int32_t>& expression, int32_t scalar);
DpuLazy<int32_t> operator*(const DpuLazy<int32_t>& expression, int32_t scalar);
DpuLazy<int32_t> operator/(const DpuLazy<int32_t>& expression, int32_t scalar);
DpuLazy<int32_t> operator<(const DpuLazy<int32_t>& expression, int32_t scalar);
DpuLazy<int32_t> operator==(const DpuLazy<int32_t>& expression, int32_t scalar);
DpuLazy<int32_t> operator>>(const DpuLazy<int32_t>& expression, int32_t scalar);
DpuLazy<int32_t> operator-(const DpuLazy<int32_t>& expression);
DpuLazy<int32_t> abs(const DpuLazy<int32_t>& expression);
DpuLazy<int32_t> sqr(const DpuLazy<int32_t>& expression);
DpuLazy<int32_t> select(const DpuLazy<int32_t>& condition,
                        const DpuLazy<int32_t>& then_value,
                        const DpuLazy<int32_t>& else_value);
DpuLazy<int32_t> argmin(const std::vector<DpuLazy<int32_t>>& lanes);
DpuLazy<int32_t> argmax(const std::vector<DpuLazy<int32_t>>& lanes);
DpuFuture<ArgResult> argmin(const DpuLazy<int32_t>& expression);
DpuFuture<ArgResult> argmax(const DpuLazy<int32_t>& expression);

template <>
class DpuFuture<int32_t> {
 public:
  using result_type = DPUVector<int32_t>::reduction_result_t;
  struct Impl;

  DpuFuture() = default;
  result_type get();
  operator result_type() { return get(); }

 private:
  explicit DpuFuture(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;

  friend DpuFuture sum(const DpuLazy<int32_t>&);
  friend DpuFuture product(const DpuLazy<int32_t>&);
  friend DpuFuture minimum(const DpuLazy<int32_t>&);
  friend DpuFuture maximum(const DpuLazy<int32_t>&);
};

template <>
class DpuFuture<ArgResult> {
 public:
  using result_type = ArgResult;
  struct Impl;

  DpuFuture() = default;
  result_type get();
  operator result_type() { return get(); }

 private:
  explicit DpuFuture(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;

  friend DpuFuture argmin(const DpuLazy<int32_t>&);
  friend DpuFuture argmax(const DpuLazy<int32_t>&);
};

DpuFuture<int32_t> sum(const DpuLazy<int32_t>& expression);
DpuFuture<int32_t> product(const DpuLazy<int32_t>& expression);
DpuFuture<int32_t> minimum(const DpuLazy<int32_t>& expression);
DpuFuture<int32_t> maximum(const DpuLazy<int32_t>& expression);

template <>
class DPULocalVector<int32_t> {
 public:
  struct Impl;

  class Reference {
   public:
    void operator+=(int32_t value);
    void operator+=(const DpuLazy<int32_t>& value);

   private:
    Reference(DPULocalVector& owner, DpuLazy<int32_t> index);
    DPULocalVector* owner_;
    DpuLazy<int32_t> index_;
    friend class DPULocalVector;
  };

  explicit DPULocalVector(uint32_t size, std::string_view name = "");
  ~DPULocalVector();
  DPULocalVector(const DPULocalVector&) = delete;
  DPULocalVector& operator=(const DPULocalVector&) = delete;

  Reference operator[](const DpuLazy<int32_t>& index);
  std::vector<int32_t> to_cpu();

 private:
  std::unique_ptr<Impl> impl_;
  void add(const DpuLazy<int32_t>& index, int32_t value);
  void add(const DpuLazy<int32_t>& index, const DpuLazy<int32_t>& value);
  void flush();

  friend void sync();
};

struct RuntimeStatistics {
  size_t events_submitted = 0;
  size_t compute_launches = 0;
  size_t dpu_transfers = 0;
  size_t host_transfers = 0;
  size_t fences = 0;
  size_t vertical_fusions = 0;
  size_t horizontal_fusions = 0;
  size_t absorbed_producers = 0;
  size_t binary_switches = 0;
  size_t oom_retries = 0;
  size_t jit_kernel_compiles = 0;
  size_t jit_kernel_cache_hits = 0;
  size_t jit_batch_links = 0;
  size_t jit_batch_cache_hits = 0;
  size_t jit_pipeline_fallbacks = 0;
  size_t jit_eager_fallbacks = 0;

  size_t total_launches() const;
  size_t fused_away() const;
  std::string to_string() const;
};

RuntimeStatistics operator-(const RuntimeStatistics& lhs,
                            const RuntimeStatistics& rhs);
RuntimeStatistics statistics();

void init(uint32_t dpus);
uint32_t ndpus();
uint32_t ntasklets();
void sync();
// sync() drains queued work; it does not materialise a vector's pending
// expression.  `x = f(x)` in a loop therefore extends one chain, and each depth
// is a distinct JIT kernel -- fence(x) per iteration keeps it to one.
void fence(DPUVector<int32_t>& vector);
void shutdown();

}  // namespace polymerpim
