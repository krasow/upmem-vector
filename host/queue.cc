#include "queue.h"

#include <cassert>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <sstream>
#include <thread>

#include "fusion.h"
#include "jit.h"
#include "opinfo.h"
#include "perfetto/trace.h"
#include "perfetto/trace_internal.h"
#include "runtime.h"
#include "stats.h"
#include "vectordpu.h"

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

namespace {
namespace fs = std::filesystem;

[[maybe_unused]] const char* canonical_jit_type_name(
    const char* raw_type_name) {
  if (!raw_type_name) return "int32_t";
  std::string tn = raw_type_name;
  if (tn == "i" || tn == "int" || tn == "int32_t") return "int32_t";
  if (tn == "j" || tn == "uint32_t") return "uint32_t";
  if (tn == "f" || tn == "float") return "float";
  if (tn == "d" || tn == "double") return "double";
  return raw_type_name;
}

std::string normalize_binary_path(std::string path) {
  return fs::path(std::move(path)).lexically_normal().string();
}

std::string describe_binary_path(const std::string& path) {
  if (path.empty()) return "<none>";

  std::string default_path =
      normalize_binary_path(DpuRuntime::get().get_default_binary_path());
  std::string current_path = normalize_binary_path(path);
  if (current_path == default_path) return "default runtime.dpu";

  return current_path;
}

size_t jit_link_batch_limit() {
  constexpr size_t kIramSafeLinkBatch = 6;
  return JIT_BATCH_SIZE < kIramSafeLinkBatch ? JIT_BATCH_SIZE
                                             : kIramSafeLinkBatch;
}

std::string compact_rpn_name(const std::vector<uint8_t>& rpn_ops) {
  size_t op_count = 0;
  size_t chain_count = rpn_ops.empty() ? 0 : 1;
  size_t reduction_count = 0;
  for (size_t i = 0; i < rpn_ops.size(); ++i) {
    uint8_t op = rpn_ops[i];
    if (op == OP_NEXT_CHAIN) {
      chain_count++;
    } else {
      op_count++;
      if (IS_OP_REDUCTION(op)) reduction_count++;
    }
    if (OP_INLINE_BYTES(op) > 0) i += OP_INLINE_BYTES(op);
  }

  std::string name = "Fused Pipeline (ops=" + std::to_string(op_count) +
                     ", bytes=" + std::to_string(rpn_ops.size());
  if (chain_count > 1) name += ", chains=" + std::to_string(chain_count);
  if (reduction_count > 0)
    name += ", reductions=" + std::to_string(reduction_count);
  name += ")";
  return name;
}

std::string human_bytes(size_t bytes) {
  static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB"};
  double value = static_cast<double>(bytes);
  size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
    value /= 1024.0;
    unit++;
  }

  std::ostringstream out;
  if (unit == 0) {
    out << bytes << " " << units[unit];
  } else {
    out << std::fixed << std::setprecision(2) << value << " " << units[unit];
  }
  return out.str();
}

std::string describe_vector(detail::VectorDescRef vec,
                            size_t transfer_bytes = 0) {
  if (!vec) return "vec=<none>";

  std::ostringstream out;
  out << "vec#" << vec->vector_id << " dpu_vector<"
      << (vec->type_name != nullptr ? vec->type_name : "unknown") << ">"
      << " size=" << vec->num_elements;
  if (transfer_bytes > 0) out << " bytes=" << human_bytes(transfer_bytes);
  if (vec->needs_layout_materialization) out << "  layout=lazy";
  if (vec->debug_name != nullptr && vec->debug_name[0] != '\0') {
    out << "  name=\"" << vec->debug_name << "\"";
  }
  return out.str();
}

std::string describe_transfer_summary(const std::shared_ptr<Event>& e) {
  if (e->op == Event::OperationType::DPU_TRANSFER) {
    return "transfer=host_to_dpu";
  }
  if (e->op == Event::OperationType::HOST_TRANSFER) {
    return "transfer=dpu_to_host";
  }
  return "name=\"" + e->slice_name + "\"";
}

std::string describe_transfer_details(const std::shared_ptr<Event>& e) {
  std::ostringstream out;
  if (e->op == Event::OperationType::DPU_TRANSFER) {
    out << describe_vector(e->output, e->transfer_size);
    return out.str();
  }

  if (e->op == Event::OperationType::HOST_TRANSFER) {
    if (e->inputs.size() == 1) {
      out << describe_vector(e->inputs[0], e->transfer_size);
    } else if (!e->inputs.empty()) {
      out << "vectors=" << e->inputs.size()
          << " first=" << describe_vector(e->inputs[0], e->transfer_size);
    }
    return out.str();
  }
  return {};
}
}  // namespace

/*static*/ dpu_error_t upmem_callback([[maybe_unused]] struct dpu_set_t stream,
                                      [[maybe_unused]] uint32_t rank_id,
                                      void* data) {
  auto self_ptr = static_cast<std::shared_ptr<Event>*>(data);
  std::shared_ptr<Event> me = *self_ptr;

  auto& runtime = DpuRuntime::get();
  auto& queue = runtime.get_event_queue();
  std::recursive_mutex& mtx = queue.get_mutex();

  {
    std::lock_guard<std::recursive_mutex> lock(mtx);
    me->mark_finished();
  }

  static std::atomic<size_t> callback_count{0};
  size_t count = ++callback_count;
  if (count % 100 == 0) {
#if ENABLE_DPU_LOGGING >= 1
    Logger& logger = DpuRuntime::get().get_logger();
    logger.lock(logcat::QUEUE_HEARTBEAT, 2)
        << "callback fired (" << count << ") for id=" << me->id << std::endl;
#endif
  }

  delete self_ptr;
  queue.outstanding_callbacks_--;

  return DPU_OK;
}

void Event::add_completion_callback(std::shared_ptr<Event> self) {
  assert(this->finished == false);

  auto& runtime = DpuRuntime::get();
  auto& queue = runtime.get_event_queue();
  dpu_set_t& dpu_set = runtime.dpu_set();

  queue.outstanding_callbacks_++;
  auto wrapper = new std::shared_ptr<Event>(self);

  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_callback, (void*)wrapper,
      (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                             DPU_CALLBACK_SINGLE_CALL)));
}

void EventQueue::add_fence(std::shared_ptr<Event> e) {
  assert(e->finished == false);

  auto& runtime = DpuRuntime::get();
  auto& queue = runtime.get_event_queue();
  dpu_set_t& dpu_set = runtime.dpu_set();

  queue.outstanding_callbacks_++;
  auto wrapper = new std::shared_ptr<Event>(std::move(e));

  CHECK_UPMEM(dpu_callback(
      dpu_set, &upmem_callback, (void*)wrapper,
      (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING |
                             DPU_CALLBACK_SINGLE_CALL)));
}

void EventQueue::sync() {
  size_t last_id;
  {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    if (operations_.empty() && running_events_.empty()) return;
    last_id = counter_ - 1;
  }
  process_events(last_id);

  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  CHECK_UPMEM(dpu_sync(dpu_set));
  finalize_finished_events();
}

void EventQueue::finalize_finished_events() {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  while (!running_events_.empty() && running_events_.front()->finished) {
    auto e = running_events_.front();
    last_finished_id_.store(e->max_id);
    trace::execution_end();
    running_events_.pop_front();

    if (!running_events_.empty()) {
      auto next = running_events_.front();
      trace::execution_begin(next);
    }
  }
}

constexpr bool NO_PROGRESS = false;
constexpr bool YES_PROGRESS = true;

bool EventQueue::process_next() {
  std::shared_ptr<Event> e;
  {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    if (operations_.empty()) return NO_PROGRESS;
    e = operations_.front();
    operations_.pop_front();
  }

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
#endif

#if ENABLE_DPU_LOGGING >= 2
  logger.lock(logcat::QUEUE_NEXT, 2)
      << "id=" << e->id << " type=" << (int)e->op
      << " deps=" << e->dependencies.size() << " started=" << (int)e->started
      << " finished=" << (int)e->finished.load() << std::endl;
#endif

#if ENABLE_DPU_LOGGING >= 2
  logger.lock(logcat::EVENT, 2)
      << "id=" << e->id << " type=" << operationtype_to_string(e->op)
      << " phase=started" << std::endl;
#endif

  trace::inqueue_end(e);

  // Wait for dependencies
  if (!e->dependencies.empty()) {
    finalize_finished_events();
    size_t max_dep = 0;
    for (size_t dep : e->dependencies)
      if (dep > max_dep) max_dep = dep;
#if ENABLE_DPU_LOGGING >= 2
    if (this->get_last_finished_id() < max_dep)
      logger.lock(logcat::QUEUE_WAIT, 2)
          << "id=" << e->id << " waiting for max_dep=" << max_dep
          << " (current=" << this->get_last_finished_id() << ")" << std::endl;
    size_t loop_count = 0;
#endif
    while (this->get_last_finished_id() < max_dep) {
      finalize_finished_events();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
#if ENABLE_DPU_LOGGING >= 2
      if (++loop_count % 1000 == 0)
        logger.lock(logcat::QUEUE_HEARTBEAT, 2)
            << "id=" << e->id << " dependency block on " << max_dep
            << " (current=" << this->get_last_finished_id() << ")" << std::endl;
#endif
    }
  }

  // 1. Initial naming
  if (e->slice_name.empty()) e->slice_name = operationtype_to_string(e->op);

  // 2. Look-ahead fusion and JIT locking
  {
    std::lock_guard<std::recursive_mutex> lock(mtx_);

#if PIPELINE
    if (e->op == Event::OperationType::COMPUTE) {
      size_t lookahead = 0;
      while (lookahead < FUSION_LOOKAHEAD && !operations_.empty()) {
        auto next = operations_.front();
        if (next->op != Event::OperationType::COMPUTE) break;
        if (!try_fuse(e, next)) break;
        operations_.pop_front();
        lookahead++;
#if ENABLE_DPU_LOGGING >= 2
        logger.lock(logcat::FUSION, 2)
            << "lookahead: child #" << next->id << " -> top #" << e->id
            << "  slot " << lookahead << std::endl;
#endif
        if (operations_.empty())
          std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
#endif

#if JIT
    if (e->op == Event::OperationType::COMPUTE && JIT_BATCH_SIZE > 0) {
      if (!e->is_locked_for_jit) lock_for_jit(e);

      if (!e->jit_future.valid()) {
        if (operations_.empty())
          std::this_thread::sleep_for(std::chrono::microseconds(200));

        auto it = operations_.begin();
        while (pending_unique_kernels_.size() < jit_link_batch_limit() &&
               it != operations_.end()) {
          auto next = *it;
          if (next->op != Event::OperationType::COMPUTE) break;
          lock_for_jit(next);
          if (e->jit_future.valid()) break;
          ++it;
        }

        if (!e->jit_future.valid()) flush_jit_batch();
      }
    }
#endif
  }

  // 3. JIT wait (outside mutex)
#if JIT
  if (e->op == Event::OperationType::COMPUTE && e->jit_future.valid()) {
#if ENABLE_DPU_LOGGING >= 1
    logger.lock(logcat::QUEUE_JIT)
        << "Awaiting background JIT compilation for id=" << e->id << std::endl;
#endif
    e->jit_binary_path = e->jit_future.get();
  }
#endif

  // 4. Refined naming
  if (e->op == Event::OperationType::COMPUTE) {
    if (!e->rpn_ops.empty()) {
      e->slice_name = compact_rpn_name(e->rpn_ops);
    } else {
      e->slice_name = kernel_id_to_string(e->kid);
    }
    if (!e->jit_binary_path.empty())
      e->slice_name += " (from " + e->jit_binary_path + ")";
  }

#if ENABLE_DPU_LOGGING >= 1
  if (e->op == Event::OperationType::COMPUTE) {
    auto log = logger.lock(logcat::QUEUE_EXEC, 2);
    log.first() << "id=" << e->id << " launch=compute";
    log.second() << "name=\"" << e->slice_name << "\"";
    if (!e->jit_kernel_hash.empty())
      log << "  kernel_hash=" << e->jit_kernel_hash;
    log << std::endl;
  }
#endif
#if ENABLE_DPU_LOGGING >= 2
  if (e->op != Event::OperationType::COMPUTE) {
    auto log = logger.lock(logcat::QUEUE_EXEC);
    log.first() << "id=" << e->id << " " << describe_transfer_summary(e);
    std::string details = describe_transfer_details(e);
    if (!details.empty()) log.second() << details;
    log << std::endl;
  }
#endif

  // 5. Register with running events and start trace
  {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    bool start_trace_now = running_events_.empty();
    if (start_trace_now) trace::execution_begin(e);
    running_events_.push_back(e);
    current_event_ = e;
    trace::active_ops_counter(running_events_.size());
  }

#if PIPELINE && JIT
  if (e->op == Event::OperationType::COMPUTE &&
      (!e->rpn_ops.empty() || e->is_locked_for_jit)) {
    if (!e->is_locked_for_jit) {
      const char* raw_type_name = nullptr;
      if (e->output && e->output->type_name)
        raw_type_name = e->output->type_name;
      else if (!e->inputs.empty() && e->inputs[0])
        raw_type_name = e->inputs[0]->type_name;
      std::string type_name = canonical_jit_type_name(raw_type_name);
      std::pair<std::vector<uint8_t>, std::string> sig = {e->rpn_ops,
                                                          type_name};
      e->jit_kernel_hash = jit_signature_hash(sig);
      e->jit_binary_path = jit_compile({sig});
      e->jit_sub_kernel_idx = 0;
      e->is_locked_for_jit = true;
    }
  }
#endif

  // 6. Binary switching
  std::string required_binary;
  if (!e->jit_binary_path.empty()) {
    required_binary = e->jit_binary_path;
  } else if (e->op == Event::OperationType::COMPUTE) {
    required_binary = DpuRuntime::get().get_default_binary_path();
  } else {
    required_binary = current_binary_path_.empty()
                          ? DpuRuntime::get().get_default_binary_path()
                          : current_binary_path_;
  }

  if (!required_binary.empty() && required_binary != current_binary_path_) {
#if ENABLE_DPU_LOGGING >= 1
    logger.lock(logcat::QUEUE_JIT)
        << "Switching binary to " << describe_binary_path(required_binary)
        << " (was " << describe_binary_path(current_binary_path_) << ")"
        << std::endl;
#endif
    trace::jit_binary_switch(current_binary_path_, required_binary);

    while (true) {
      std::lock_guard<std::recursive_mutex> lock(mtx_);
      if (running_events_.size() <= 1) {
        if (running_events_.empty() || running_events_.front() == e) break;
      }
      std::this_thread::yield();
    }

    dpu_set_t& dpu_set = DpuRuntime::get().dpu_set();
#if ENABLE_DPU_LOGGING >= 1
    logger.lock(logcat::QUEUE_JIT)
        << "Loading binary onto " << DpuRuntime::get().num_dpus() << " DPUs..."
        << std::endl;
#endif
    VECTORDPU_NOTE(binary_switches);
    DPU_ASSERT(dpu_load(dpu_set, required_binary.c_str(), nullptr));
    current_binary_path_ = required_binary;
#if ENABLE_DPU_LOGGING >= 1
    logger.lock(logcat::QUEUE_JIT) << "Binary load successful." << std::endl;
#endif
  } else if (current_binary_path_.empty()) {
    current_binary_path_ = DpuRuntime::get().get_default_binary_path();
  }

  // 7. Dispatch
  try {
#if ENABLE_DPU_LOGGING >= 2
    logger.lock(logcat::QUEUE_DISPATCH, 2)
        << "id=" << e->id << " type=" << (int)e->op << " begin" << std::endl;
#endif
    switch (e->op) {
      case Event::OperationType::FENCE:
        VECTORDPU_NOTE(fences);
        this->add_fence(e);
        break;
      case Event::OperationType::COMPUTE:
        VECTORDPU_NOTE(compute_launches);
        e->started = true;
#if PIPELINE
        if (!e->rpn_ops.empty() || e->is_locked_for_jit) {
          KernelID dynamic_kid =
              e->is_locked_for_jit
                  ? (KernelID)(JIT_STATIC_KERNEL_COUNT + e->jit_sub_kernel_idx)
                  : e->pipeline_kid;
          detail::internal_launch_universal_pipeline(
              e->output, (e->inputs.empty() ? nullptr : e->inputs[0]),
              e->rpn_ops,
              (e->inputs.size() > 1
                   ? std::vector<detail::VectorDescRef>(e->inputs.begin() + 1,
                                                        e->inputs.end())
                   : std::vector<detail::VectorDescRef>()),
              dynamic_kid, e->scalars, e->extra_scalars, e->extra_outputs,
              e->jit_kernel_hash);
        } else if (e->cb) {
          e->cb();
        }
#else
        if (e->cb) e->cb();
#endif
        e->add_completion_callback(e);
        break;
      case Event::OperationType::DPU_TRANSFER:
        VECTORDPU_NOTE(dpu_transfers);
        e->started = true;
        e->cb();
        e->add_completion_callback(e);
        break;
      case Event::OperationType::HOST_TRANSFER:
        VECTORDPU_NOTE(host_transfers);
        e->started = true;
        e->cb();
        e->add_completion_callback(e);
        break;
      default:
        assert(false && "Unknown event type");
    }
#if ENABLE_DPU_LOGGING >= 2
    logger.lock(logcat::QUEUE_DISPATCH, 2)
        << "id=" << e->id << " type=" << (int)e->op << " end" << std::endl;
#endif
  } catch (const DpuOOMException& ex) {
#if ENABLE_DPU_LOGGING >= 1
    logger.lock(logcat::OOM)
        << "caught for event id=" << e->id << " started=" << e->started
        << " retries=" << e->oom_retries << std::endl;
#endif
#if !ENABLE_OOM_RECOVERY
    throw DpuOOMException("DPU OOM: event id=" + std::to_string(e->id));
#else
    VECTORDPU_NOTE(oom_retries);
    if (++e->oom_retries > OOM_RECOVERY_RETRIES)
      throw DpuOOMException("DPU OOM: event id=" + std::to_string(e->id) +
                            " failed after " +
                            std::to_string(OOM_RECOVERY_RETRIES) + " retries");
    e->started = false;
    std::vector<detail::VectorDescRef> outputs_to_free;
    if (e->output) outputs_to_free.push_back(e->output);
    for (auto& out : e->extra_outputs)
      if (out) outputs_to_free.push_back(out);
    {
      std::lock_guard<std::recursive_mutex> lock(mtx_);
      running_events_.remove(e);
      if (current_event_ == e) current_event_ = nullptr;
    }
#if ENABLE_DPU_LOGGING >= 1
    logger.lock(logcat::OOM) << "freed failed outputs for event id=" << e->id
                             << ", requeueing" << std::endl;
#endif
    auto& alloc = DpuRuntime::get().get_allocator();
    for (auto& out : outputs_to_free) alloc.deallocate_upmem_vector(out.get());
    {
      std::lock_guard<std::recursive_mutex> lock(mtx_);
      operations_.push_front(e);
      if (current_event_ == e) current_event_ = nullptr;
    }
#if ENABLE_DPU_LOGGING >= 1
    logger.lock(logcat::OOM)
        << "event id=" << e->id << " requeued" << std::endl;
#endif
    return NO_PROGRESS;
#endif
  }

  debug_active_events();
  debug_print_queue();
  return YES_PROGRESS;
}

void EventQueue::process_events(size_t wait_for_id) {
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock(logcat::QUEUE_WAIT, 2)
      << "begin wait_for_id=" << wait_for_id << std::endl;
#endif
  while (true) {
    bool progress = this->process_next();

    if (this->get_last_finished_id() >= wait_for_id) break;

    {
      std::lock_guard<std::recursive_mutex> lock(mtx_);
      if (operations_.empty() && running_events_.empty()) break;
    }

    auto& runtime = DpuRuntime::get();
    dpu_set_t& dpu_set = runtime.dpu_set();
#if ENABLE_DPU_LOGGING >= 2
    logger.lock(logcat::QUEUE_WAIT, 2)
        << "dpu_sync wait_for_id=" << wait_for_id
        << " last_finished=" << this->get_last_finished_id() << std::endl;
#endif
    CHECK_UPMEM(dpu_sync(dpu_set));
    finalize_finished_events();
#if ENABLE_DPU_LOGGING >= 2
    logger.lock(logcat::QUEUE_WAIT, 2)
        << "dpu_sync done wait_for_id=" << wait_for_id
        << " last_finished=" << this->get_last_finished_id() << std::endl;
#endif

    if (!progress) std::this_thread::sleep_for(std::chrono::milliseconds(1));
#if ENABLE_DPU_LOGGING >= 2
    static size_t loop_count = 0;
    if (++loop_count % 1000 == 0) {
      std::lock_guard<std::recursive_mutex> lock(mtx_);
      logger.lock(logcat::QUEUE_HEARTBEAT, 2)
          << "process_events waiting for " << wait_for_id
          << " (last_finished=" << this->get_last_finished_id()
          << " ops=" << operations_.size()
          << " running=" << running_events_.size() << ")" << std::endl;
    }
#endif
  }
#if ENABLE_DPU_LOGGING >= 2
  logger.lock(logcat::QUEUE_WAIT, 2)
      << "end wait_for_id=" << wait_for_id
      << " last_finished=" << this->get_last_finished_id() << std::endl;
#endif
}

void EventQueue::debug_print_queue() {
#if ENABLE_DPU_LOGGING >= 3
  Logger& logger = DpuRuntime::get().get_logger();
  if (!operations_.empty()) {
    logger.lock(logcat::EVENT_QUEUE, 3) << "pending queue" << std::endl;
    std::deque<std::shared_ptr<Event>> tmp = operations_;
    int i = 0;
    while (!tmp.empty()) {
      auto e = tmp.front();
      auto log = logger.lock(logcat::EVENT_QUEUE, 3);
      log.second() << i++ << ". id=" << e->id
                   << " type=" << operationtype_to_string(e->op)
                   << " started=" << e->started << " finished=" << e->finished
                   << std::endl;
      tmp.pop_front();
    }
  } else {
    logger.lock(logcat::EVENT_QUEUE, 3) << "pending queue empty" << std::endl;
  }
#endif
}

void EventQueue::debug_active_events() {
#if ENABLE_DPU_LOGGING >= 3
  Logger& logger = DpuRuntime::get().get_logger();
  auto& events = get_active_events();
  std::lock_guard<std::recursive_mutex> lock(get_mutex());
  if (!events.empty()) {
    logger.lock(logcat::EVENT_QUEUE, 3) << "active runners" << std::endl;
    int i = 0;
    for (const auto& e : events) {
      auto log = logger.lock(logcat::EVENT_QUEUE, 3);
      log.second() << i++ << ". id=" << e->id
                   << " type=" << operationtype_to_string(e->op)
                   << " started=" << e->started << " finished=" << e->finished
                   << std::endl;
    }
  } else {
    logger.lock(logcat::EVENT_QUEUE, 3) << "active runners empty" << std::endl;
  }
#endif
}

size_t EventQueue::count_internal_references(detail::VectorDescRef vec) {
  if (!vec) return 0;
  size_t count = 0;
  auto count_in = [&](std::shared_ptr<Event> ev) {
    if (!ev) return;
    if (ev->output == vec) count++;
    for (const auto& out : ev->extra_outputs)
      if (out == vec) count++;
    for (const auto& in : ev->inputs)
      if (in == vec) count++;
  };
  count_in(current_event_);
  for (auto& ev : operations_) count_in(ev);
  for (auto& ev : running_events_) count_in(ev);
  for (auto& ev : pending_jit_events_) count_in(ev);
  return count;
}

// Thin dispatcher: classifies the fusion as vertical or horizontal and
// delegates to the appropriate implementation in vfuse.cc / hfuse.cc.
bool EventQueue::try_fuse(std::shared_ptr<Event> last,
                          std::shared_ptr<Event> e) {
#if PIPELINE
  if (last->op != Event::OperationType::COMPUTE ||
      e->op != Event::OperationType::COMPUTE || last->output == nullptr)
    return false;
  if (last->is_locked_for_jit || e->is_locked_for_jit) return false;

  bool dependent = false;
  for (const auto& in : e->inputs) {
    if (in == last->output) {
      dependent = true;
      break;
    }
    for (const auto& out : last->extra_outputs)
      if (in == out) {
        dependent = true;
        break;
      }
  }

  if (dependent) return try_vfuse(last, e);

  // If e already inlined last's output via absorbed_rpn, horizontally fusing
  // last as a separate chain would duplicate that work and shift result slots.
  if (e->absorbed_producer_ids.find(last->id) != e->absorbed_producer_ids.end())
    return false;

  // Horizontal: independent chains, same element count, operand budget fits.
  if (last->inputs.empty() || e->inputs.empty()) return false;
  if (last->inputs[0]->num_elements != e->inputs[0]->num_elements) return false;

  std::vector<detail::VectorDescRef> unique = last->inputs;
  for (const auto& in : e->inputs) {
    bool found = false;
    for (const auto& u : unique)
      if (in == u) {
        found = true;
        break;
      }
    if (!found) unique.push_back(in);
  }
  if (unique.size() > MAX_COMBINED_INPUTS) return false;

  // Don't hfuse a freshly-submitted scalar-op intermediate — it will almost
  // certainly be consumed by an imminent vfuse candidate (e.g. vec*s followed
  // by acc+product).  Hfusing it now would lock it into an extra_output slot,
  // blocking the subsequent add from extending the primary chain (the linreg
  // `error = error + dx[j]*dw[j]` loop collapses to one deep vfuse chain
  // only if each per-dim scalar mul stays separate until its accumulator
  // add arrives and the two can vfuse together.  This also applies after a
  // reduction: in linreg's gradient fanout, `dx[j] >> shift` must wait for its
  // multiply/sum consumer so only complete reduction chains are hfused.
  if (e->is_scalar && e->rpn_ops.empty()) return false;

  // Don't hfuse an event whose output is marked for inline absorption.
  // The next consumer will expand it via absorbed_rpn; hfusing it now just
  // wastes an extra_output chain slot and duplicates computation.
  if (e->output && !e->output->absorbed_rpn.empty()) return false;

  return try_hfuse(last, e);
#else
  return false;
#endif
}

void EventQueue::submit(std::shared_ptr<Event> e) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  while (operations_.size() + running_events_.size() >= max_queue_depth_) {
    mtx_.unlock();
    this->process_next();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    mtx_.lock();
  }

  e->id = counter_++;
  e->max_id = e->id;
  VECTORDPU_NOTE(events_submitted);

#if PIPELINE
  expand_absorbed_inputs(e);
#endif

  trace::event_enqueued(e, operations_, running_events_);
  trace::active_ops_counter(operations_.size());

  bool fused = false;
#if PIPELINE
  if (e->op == Event::OperationType::COMPUTE && !operations_.empty()) {
    auto last = operations_.back();
    if (try_fuse(last, e)) {
      fused = true;
      // Retroactive chain fusion: collapse K1→K2→K3 into one kernel when each
      // step consumes the previous on-stack result.
      while (operations_.size() >= 2) {
        auto& prev = operations_[operations_.size() - 2];
        auto& tail = operations_.back();
        if (!try_fuse(prev, tail)) break;
        operations_.pop_back();
      }
    }
  }
#endif

  if (!fused) {
#if JIT
    // Flush the pending JIT batch only on boundaries where the queue cannot
    // grow the current batch further — a non-COMPUTE event (HOST_TRANSFER /
    // FENCE) terminates the run of fusable kernels.  Reduction COMPUTE events
    // are still candidates for hfuse into subsequent reduction chains and
    // must stay batchable; let process_next's look-ahead absorb them into the
    // in-flight batch so we don't emit a separate JIT binary for reductions
    // (which would force a mid-stream binary switch).
    if (e->op != Event::OperationType::COMPUTE) {
      bool any_locked = false;
      for (auto& op : operations_) {
        if (op->op == Event::OperationType::COMPUTE && !op->is_locked_for_jit) {
          lock_for_jit(op);
          any_locked = true;
        }
      }
      if (any_locked) flush_jit_batch();
    }
    if (e->op == Event::OperationType::COMPUTE && e->rpn_ops.empty() &&
        JIT_BATCH_SIZE > 0)
      e->kid = e->pipeline_kid;
#endif
    for (const auto& in : e->inputs)
      if (in && in->last_producer_id != 0)
        e->dependencies.insert(in->last_producer_id);
    if (e->output) e->output->last_producer_id = e->id;
    for (const auto& out : e->extra_outputs)
      if (out) out->last_producer_id = e->id;

    operations_.push_back(e);

#if PIPELINE
    // Set absorbed_rpn so expand_absorbed_inputs can inline standalone events
    // (e.g. unary negate) into the first consumer's event, enabling chain
    // growth.
    if (e->op == Event::OperationType::COMPUTE && !IS_OP_REDUCTION(e->opcode) &&
        e->output && !e->inputs.empty() && e->extra_outputs.empty()) {
      std::vector<uint8_t> rpn;
      std::vector<uint32_t> scalars;
      build_default_rpn(e, rpn, scalars);
      e->output->absorbed_rpn = rpn;
      e->output->absorbed_scalars = scalars;
      e->output->absorbed_inputs = e->inputs;
      e->output->is_shared_intermediate = true;
    }
#endif
  }
}
