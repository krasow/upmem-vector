#pragma once

#include <atomic>
#include <cstddef>
#include <string>

#include "config.h"

// Runtime observability counters.
//
// These are always compiled in, independent of ENABLE_DPU_LOGGING, so that
// tests can assert on fusion/JIT behaviour ("this chain of 5 ops must collapse
// into 1 kernel") and so benchmarks can report kernel counts without parsing
// log output.  Counters are monotonic: take a snapshot before and after a
// region of interest and subtract.
//
// Increments are relaxed atomics on the submit/dispatch path only -- a handful
// of instructions per event, which is noise next to a DPU launch.

#if JIT_PIPELINE_FALLBACK
#define VECTORDPU_HYBRID_STAT(X) \
  X(jit_pipeline_fallbacks, "interpreter launches while JIT is compiling")
#else
#define VECTORDPU_HYBRID_STAT(X)
#endif

// X(field, description)
#define VECTORDPU_STAT_LIST(X)                                                 \
  X(events_submitted, "events accepted by EventQueue::submit")                 \
  X(compute_launches, "COMPUTE events dispatched to the DPUs (kernel passes)") \
  X(dpu_transfers, "host-to-DPU transfers dispatched")                         \
  X(host_transfers, "DPU-to-host transfers dispatched")                        \
  X(fences, "fence events dispatched")                                         \
  X(vertical_fusions, "successful try_vfuse merges")                           \
  X(horizontal_fusions, "successful try_hfuse merges")                         \
  X(absorbed_producers, "producer events inlined into a consumer and erased")  \
  X(binary_switches, "dpu_load calls (DPU binary swaps)")                      \
  X(oom_retries, "events requeued after a DPU OOM")                            \
  X(jit_kernel_compiles, "distinct RPN kernels compiled to a DPU object")      \
  X(jit_kernel_cache_hits, "kernel-object cache hits")                         \
  X(jit_batch_links, "JIT batches linked into a new DPU binary")               \
  X(jit_batch_cache_hits, "linked-binary cache hits")                          \
  VECTORDPU_HYBRID_STAT(X)

struct StatsSnapshot {
#define VECTORDPU_STAT_FIELD(name, desc) size_t name = 0;
  VECTORDPU_STAT_LIST(VECTORDPU_STAT_FIELD)
#undef VECTORDPU_STAT_FIELD

  // Total kernel passes plus transfers -- i.e. every DPU-visible operation.
  size_t total_launches() const {
    return compute_launches + dpu_transfers + host_transfers + fences;
  }

  // Events that never reached the DPU because fusion absorbed them.
  size_t fused_away() const {
    return vertical_fusions + horizontal_fusions + absorbed_producers;
  }

  std::string to_string() const;
};

inline StatsSnapshot operator-(const StatsSnapshot& a, const StatsSnapshot& b) {
  StatsSnapshot out;
#define VECTORDPU_STAT_SUB(name, desc) out.name = a.name - b.name;
  VECTORDPU_STAT_LIST(VECTORDPU_STAT_SUB)
#undef VECTORDPU_STAT_SUB
  return out;
}

class RuntimeStats {
 public:
  // Lives in libvectordpu.so so the library and its clients share one instance.
  static RuntimeStats& get();

  StatsSnapshot snapshot() const {
    StatsSnapshot out;
#define VECTORDPU_STAT_READ(name, desc) \
  out.name = name##_.load(std::memory_order_relaxed);
    VECTORDPU_STAT_LIST(VECTORDPU_STAT_READ)
#undef VECTORDPU_STAT_READ
    return out;
  }

  void reset() {
#define VECTORDPU_STAT_ZERO(name, desc) \
  name##_.store(0, std::memory_order_relaxed);
    VECTORDPU_STAT_LIST(VECTORDPU_STAT_ZERO)
#undef VECTORDPU_STAT_ZERO
  }

#define VECTORDPU_STAT_NOTE(name, desc)              \
  void note_##name(size_t n = 1) {                   \
    name##_.fetch_add(n, std::memory_order_relaxed); \
  }
  VECTORDPU_STAT_LIST(VECTORDPU_STAT_NOTE)
#undef VECTORDPU_STAT_NOTE

 private:
  RuntimeStats() = default;

#define VECTORDPU_STAT_MEMBER(name, desc) std::atomic<size_t> name##_{0};
  VECTORDPU_STAT_LIST(VECTORDPU_STAT_MEMBER)
#undef VECTORDPU_STAT_MEMBER
};

// Shorthand used on the hot paths.
#define VECTORDPU_NOTE(counter) RuntimeStats::get().note_##counter()
