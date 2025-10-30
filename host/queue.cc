#include "queue.h"

#include "runtime.h"

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

/*static*/ dpu_error_t upmem_start_callback(
    [[maybe_unused]] struct dpu_set_t stream, [[maybe_unused]] uint32_t rank_id,
    void* data) {
  Event* me = (Event*)data;
  me->mark_finished(/* true */);
  return DPU_OK;
}

void Event::add_completion_callback() {
  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_start_callback, (void*)this,
      (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                             DPU_CALLBACK_SINGLE_CALL)));
  assert(this->finished == false);
}

void EventQueue::add_fence(Event e) {
  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_start_callback, (void*)&e,
      (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                             DPU_CALLBACK_SINGLE_CALL)));
  assert(e.finished == false);
}

void EventQueue::process_next() {
  auto e = operations_.front();  // Peek

  switch (e.op) {
    case Event::OperationType::FENCE:
      EventQueue::add_fence(e);
      break;
    case Event::OperationType::COMPUTE:
      e.started = true;
      e.cb();
      e.add_completion_callback();
      break;
    case Event::OperationType::DPU_TRANSFER:
      e.started = true;
      e.cb();
      e.add_completion_callback();
      break;
    case Event::OperationType::HOST_TRANSFER:
      e.started = true;
      e.cb();
      e.add_completion_callback();
      break;
  }
  operations_.pop();  // Remove
}

void EventQueue::process_events() {
  while (!operations_.empty()) {
    this->process_next();
  }
}