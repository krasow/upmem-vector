#include "queue.h"

#include "logger.inl"
#include "runtime.h"

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

std::string operationtype_to_string(Event::OperationType op) {
  switch (op) {
    case Event::OperationType::COMPUTE:
      return "COMPUTE";
    case Event::OperationType::DPU_TRANSFER:
      return "DPU_TRANSFER";
    case Event::OperationType::HOST_TRANSFER:
      return "HOST_TRANSFER";
    case Event::OperationType::FENCE:
      return "FENCE";
    default:
      return "UNKNOWN";
  }
}

/*static*/ dpu_error_t upmem_start_callback(
    [[maybe_unused]] struct dpu_set_t stream, [[maybe_unused]] uint32_t rank_id,
    void* data) {
  Event* me = (Event*)data;
  me->mark_finished(/* true */);
#ifdef ENABLE_DPU_LOGGING
  std::cout << "[Event] Callback of event of type "
            << operationtype_to_string(me->op) << " finished." << std::endl;
#endif
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

#ifdef ENABLE_DPU_LOGGING
  std::cout << "[Event] Added completion callback." << std::endl;
#endif
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
  if (operations_.empty()) {
    return;
  }
  auto e = operations_.front();  // Peek

  debug_print_queue();

#ifdef ENABLE_DPU_LOGGING
  std::cout << "[EventQueue] Processing " << operationtype_to_string(e.op)
            << " event." << std::endl;
#endif

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
    default:
      assert(false && "Unknown event type");
  }
  operations_.pop();  // Remove
}

void EventQueue::process_events() {
  while (!operations_.empty()) {
    this->process_next();
  }
}

void EventQueue::debug_print_queue() {
#ifdef ENABLE_DPU_LOGGING
  std::cout << "[EventQueue] Current queue state:" << std::endl;
  std::queue<Event> temp_queue = operations_;
  while (!temp_queue.empty()) {
    Event e = temp_queue.front();
    std::cout << "  Event type: " << operationtype_to_string(e.op)
              << ", started: " << e.started << ", finished: " << e.finished
              << std::endl;
    temp_queue.pop();
  }
#endif
}