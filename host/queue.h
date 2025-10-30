#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <variant>

#include "vectordpu.h"

class Event {
 public:
  enum class OperationType { COMPUTE, DPU_TRANSFER, HOST_TRANSFER, FENCE };

  OperationType op;
  std::function<void()> cb;

  // Result of the event
  std::variant<std::monostate, dpu_vector<int>, dpu_vector<float>> res;

  Event(OperationType t) : op(t), res(std::monostate()) {}

  template <typename Callable>
  Event(OperationType t, Callable&& c)
      : op(t), cb(std::forward<Callable>(c)), res(std::monostate()) {}

  bool finished = false;
  bool started = false;

  void add_completion_callback();
  void mark_finished() { this->finished = true; }
};

class EventQueue {
 public:
  EventQueue() = default;
  ~EventQueue() = default;

  void submit(std::shared_ptr<Event> e) {
    operations_.push(e);
  }

  void add_fence(std::shared_ptr<Event> e);

  void wait();
  void process_next();
  void process_events();
  void debug_print_queue();

  bool has_pending() const { return !operations_.empty(); }
  std::size_t pending_count() const { return operations_.size(); }

 private:
  std::queue<std::shared_ptr<Event>> operations_;
};
