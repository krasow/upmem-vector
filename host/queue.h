#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
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
  Event(OperationType t, std::function<void()> c)
      : op(t), cb(c), res(std::monostate()) {}

  bool finished = false;
  bool started = false;

  void add_completion_callback();
  void mark_finished() { this->finished = true; }
};

class EventQueue {
 public:
  EventQueue() = default;
  ~EventQueue() = default;

  void submit(Event e) { operations_.push(std::move(e)); }

  void add_fence(Event e);

  void wait();
  void process_next();
  void process_events();

  bool has_pending() const { return !operations_.empty(); }
  std::size_t pending_count() const { return operations_.size(); }

 private:
  std::queue<Event> operations_;
};
