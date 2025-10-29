#pragma once

#include <iostream>

#ifndef DPURT
#define DPURT
#include <dpu> // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif


#include "allocator.h"
#include "queue.h"

class DpuRuntime {
 private:
  DpuRuntime() : initialized_(false), allocator_(nullptr) {}

  bool initialized_;
  dpu_set_t dpu_set_;
  uint32_t num_dpus_;
  allocator* allocator_;
  EventQueue* event_queue_;

 public:
  // Delete copy/move
  DpuRuntime(const DpuRuntime&) = delete;
  DpuRuntime& operator=(const DpuRuntime&) = delete;

  // Get singleton instance
  static DpuRuntime& get() {
    static DpuRuntime instance;
    return instance;
  }

  void init(uint32_t num_dpus) {
    if (!initialized_) {
      num_dpus_ = num_dpus;

#if ENABLE_DPU_LOGGING == 1
      std::cout << "[runtime] Initializing DPU runtime with " << num_dpus_
                << " DPUs..." << std::endl;
#endif

      DPU_ASSERT(dpu_alloc(num_dpus_, "backend=simulator", &dpu_set_));
      DPU_ASSERT(dpu_load(dpu_set_, DPU_RUNTIUME, NULL));

#if ENABLE_DPU_LOGGING == 1
      std::cout << "[runtime] DPU runtime initialized." << std::endl;
#endif

      initialized_ = true;
      allocator_ = new allocator(0, 64 * 1024 * 1024, num_dpus_);
      event_queue_ = new EventQueue();
    }
  }

  ~DpuRuntime() { delete allocator_; }

  bool is_initialized() const { return initialized_; }

  dpu_set_t& dpu_set() { return dpu_set_; }
  uint32_t num_dpus() const { return num_dpus_; }
  uint32_t num_tasklets() const { return NR_TASKLETS; }
  allocator& get_allocator() { return *allocator_; }
  EventQueue& get_event_queue() { return *event_queue_; }
};
