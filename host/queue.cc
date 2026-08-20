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
// Perfetto slice name, set before dispatch so a trace shows what ran.
void name_event(const std::shared_ptr<Event>& e) {
  if (e->slice_name.empty()) e->slice_name = operationtype_to_string(e->op);
  if (e->op != Event::OperationType::COMPUTE) return;

  e->slice_name = kernel_id_to_string(e->kid);
#if PIPELINE
  if (!e->rpn_ops.empty()) e->slice_name = fused_pipeline_label(e->rpn_ops);
#endif
  if (!e->jit_binary_path.empty())
    e->slice_name += " (from " + e->jit_binary_path + ")";
}

// Its own JIT binary, the default one for a compute event, or whatever is
// already loaded for a transfer.
std::string required_binary_for(const std::shared_ptr<Event>& e,
                                const std::string& current) {
  if (!e->jit_binary_path.empty()) return e->jit_binary_path;
  if (e->op == Event::OperationType::COMPUTE)
    return DpuRuntime::get().get_default_binary_path();
  return current.empty() ? DpuRuntime::get().get_default_binary_path()
                         : current;
}

// A fused kernel that missed the JIT batch still needs a binary of its own.
void compile_kernel_if_unbatched(const std::shared_ptr<Event>& e) {
#if PIPELINE && JIT
  if (e->op != Event::OperationType::COMPUTE) return;
  if (e->rpn_ops.empty() && !e->is_locked_for_jit) return;
  if (e->is_locked_for_jit) return;

  Signature sig = event_kernel_signature(e);
  e->jit_kernel_hash = jit_signature_hash(sig);
  e->jit_binary_path = jit_compile({sig});
  e->jit_sub_kernel_idx = 0;
  e->is_locked_for_jit = true;
#else
  (void)e;
#endif
}

void launch_compute(const std::shared_ptr<Event>& e) {
#if PIPELINE
  if (e->rpn_ops.empty() && !e->is_locked_for_jit) {
    if (e->cb) e->cb();
    return;
  }

  // Batched kernels are addressed by their slot in the JIT binary, unbatched
  // ones by their static pipeline id.
  const KernelID kid =
      e->is_locked_for_jit
          ? (KernelID)(JIT_STATIC_KERNEL_COUNT + e->jit_sub_kernel_idx)
          : e->pipeline_kid;
  const std::vector<detail::VectorDescRef> operands =
      e->inputs.size() > 1 ? std::vector<detail::VectorDescRef>(
                                 e->inputs.begin() + 1, e->inputs.end())
                           : std::vector<detail::VectorDescRef>();

  detail::internal_launch_universal_pipeline(
      e->output, e->inputs.empty() ? nullptr : e->inputs[0], e->rpn_ops,
      operands, kid, e->scalars, e->extra_scalars, e->extra_outputs,
      e->jit_kernel_hash);
#else
  if (e->cb) e->cb();
#endif
}

// Logging wrappers, so the stages above read as control flow.  Each compiles
// away when its level is not enabled.

#if ENABLE_DPU_LOGGING >= 1
Logger& queue_logger() { return DpuRuntime::get().get_logger(); }
#endif

void log_next_operation(const std::shared_ptr<Event>& e) {
#if ENABLE_DPU_LOGGING >= 2
  queue_logger().lock(logcat::QUEUE_NEXT, 2)
      << "id=" << e->id << " type=" << (int)e->op
      << " deps=" << e->dependencies.size() << " started=" << (int)e->started
      << " finished=" << (int)e->finished.load() << std::endl;
  queue_logger().lock(logcat::EVENT, 2)
      << "id=" << e->id << " type=" << operationtype_to_string(e->op)
      << " phase=started" << std::endl;
#else
  (void)e;
#endif
}

void log_dependency_wait(const std::shared_ptr<Event>& e, size_t max_dep,
                         size_t finished) {
#if ENABLE_DPU_LOGGING >= 2
  if (finished >= max_dep) return;
  queue_logger().lock(logcat::QUEUE_WAIT, 2)
      << "id=" << e->id << " waiting for max_dep=" << max_dep
      << " (current=" << finished << ")" << std::endl;
#else
  (void)e;
  (void)max_dep;
  (void)finished;
#endif
}

void log_dependency_stall(const std::shared_ptr<Event>& e, size_t max_dep,
                          size_t finished) {
#if ENABLE_DPU_LOGGING >= 2
  queue_logger().lock(logcat::QUEUE_HEARTBEAT, 2)
      << "id=" << e->id << " dependency block on " << max_dep
      << " (current=" << finished << ")" << std::endl;
#else
  (void)e;
  (void)max_dep;
  (void)finished;
#endif
}

void log_lookahead_fusion(const std::shared_ptr<Event>& e,
                          const std::shared_ptr<Event>& absorbed, size_t slot) {
#if ENABLE_DPU_LOGGING >= 2
  queue_logger().lock(logcat::FUSION, 2)
      << "lookahead: child #" << absorbed->id << " -> top #" << e->id
      << "  slot " << slot << std::endl;
#else
  (void)e;
  (void)absorbed;
  (void)slot;
#endif
}

void log_jit_wait(const std::shared_ptr<Event>& e) {
#if ENABLE_DPU_LOGGING >= 1
  queue_logger().lock(logcat::QUEUE_JIT)
      << "Awaiting background JIT compilation for id=" << e->id << std::endl;
#else
  (void)e;
#endif
}

void log_launch(const std::shared_ptr<Event>& e) {
#if ENABLE_DPU_LOGGING >= 1
  if (e->op == Event::OperationType::COMPUTE) {
    auto log = queue_logger().lock(logcat::QUEUE_EXEC, 2);
    log.first() << "id=" << e->id << " launch=compute";
    log.second() << "name=\"" << e->slice_name << "\"";
    if (!e->jit_kernel_hash.empty())
      log << "  kernel_hash=" << e->jit_kernel_hash;
    log << std::endl;
  }
#endif
#if ENABLE_DPU_LOGGING >= 2
  if (e->op != Event::OperationType::COMPUTE) {
    auto log = queue_logger().lock(logcat::QUEUE_EXEC);
    log.first() << "id=" << e->id << " " << describe_transfer_summary(e);
    std::string details = describe_transfer_details(e);
    if (!details.empty()) log.second() << details;
    log << std::endl;
  }
#endif
#if ENABLE_DPU_LOGGING < 1
  (void)e;
#endif
}

void log_binary_switch(const std::string& from, const std::string& to) {
#if ENABLE_DPU_LOGGING >= 1
  queue_logger().lock(logcat::QUEUE_JIT)
      << "Switching binary to " << describe_binary_path(to) << " (was "
      << describe_binary_path(from) << ")" << std::endl;
  queue_logger().lock(logcat::QUEUE_JIT)
      << "Loading binary onto " << DpuRuntime::get().num_dpus() << " DPUs..."
      << std::endl;
#else
  (void)from;
  (void)to;
#endif
}

void log_binary_loaded() {
#if ENABLE_DPU_LOGGING >= 1
  queue_logger().lock(logcat::QUEUE_JIT)
      << "Binary load successful." << std::endl;
#endif
}

void log_dispatch(const std::shared_ptr<Event>& e, const char* phase) {
#if ENABLE_DPU_LOGGING >= 2
  queue_logger().lock(logcat::QUEUE_DISPATCH, 2)
      << "id=" << e->id << " type=" << (int)e->op << " " << phase << std::endl;
#else
  (void)e;
  (void)phase;
#endif
}

void log_oom_caught(const std::shared_ptr<Event>& e) {
#if ENABLE_DPU_LOGGING >= 1
  queue_logger().lock(logcat::OOM)
      << "caught for event id=" << e->id << " started=" << e->started
      << " retries=" << e->oom_retries << std::endl;
#else
  (void)e;
#endif
}

void log_oom_freeing(const std::shared_ptr<Event>& e) {
#if ENABLE_DPU_LOGGING >= 1
  queue_logger().lock(logcat::OOM)
      << "freed failed outputs for event id=" << e->id << ", requeueing"
      << std::endl;
#else
  (void)e;
#endif
}

void log_oom_requeued(const std::shared_ptr<Event>& e) {
#if ENABLE_DPU_LOGGING >= 1
  queue_logger().lock(logcat::OOM)
      << "event id=" << e->id << " requeued" << std::endl;
#else
  (void)e;
#endif
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

std::shared_ptr<Event> EventQueue::take_next_operation() {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  if (operations_.empty()) return nullptr;
  std::shared_ptr<Event> e = operations_.front();
  operations_.pop_front();
  return e;
}

void EventQueue::await_dependencies(const std::shared_ptr<Event>& e) {
  if (e->dependencies.empty()) return;

  finalize_finished_events();
  size_t max_dep = 0;
  for (size_t dep : e->dependencies)
    if (dep > max_dep) max_dep = dep;

  log_dependency_wait(e, max_dep, get_last_finished_id());
  size_t polls = 0;
  while (get_last_finished_id() < max_dep) {
    finalize_finished_events();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (++polls % 1000 == 0)
      log_dependency_stall(e, max_dep, get_last_finished_id());
  }
}

// Absorbs following events into `e` while fusion allows, then reserves `e` a
// slot in the pending JIT batch so it shares a binary with its neighbours.
void EventQueue::grow_fusion_batch(const std::shared_ptr<Event>& e) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);

#if PIPELINE
  if (e->op == Event::OperationType::COMPUTE) {
    size_t absorbed = 0;
    while (absorbed < FUSION_LOOKAHEAD && !operations_.empty()) {
      auto next = operations_.front();
      if (next->op != Event::OperationType::COMPUTE) break;
      if (!try_fuse(e, next)) break;
      operations_.pop_front();
      log_lookahead_fusion(e, next, ++absorbed);
      // Let a producer thread enqueue more fusable work.
      if (operations_.empty())
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
#endif

#if JIT
  if (e->op != Event::OperationType::COMPUTE || JIT_BATCH_SIZE <= 0) return;
  if (!e->is_locked_for_jit) lock_for_jit(e);
  if (e->jit_future.valid()) return;

  if (operations_.empty())
    std::this_thread::sleep_for(std::chrono::microseconds(200));

  // lock_for_jit hands `e` a future once the batch it lands in is dispatched.
  auto it = operations_.begin();
  while (pending_unique_kernels_.size() < jit_link_batch_limit() &&
         it != operations_.end()) {
    if ((*it)->op != Event::OperationType::COMPUTE) break;
    lock_for_jit(*it);
    if (e->jit_future.valid()) break;
    ++it;
  }

  if (!e->jit_future.valid()) flush_jit_batch();
#endif
}

void EventQueue::await_jit_binary(const std::shared_ptr<Event>& e) {
#if JIT
  if (e->op != Event::OperationType::COMPUTE || !e->jit_future.valid()) return;
  log_jit_wait(e);
  e->jit_binary_path = e->jit_future.get();
#else
  (void)e;
#endif
}

void EventQueue::begin_running(const std::shared_ptr<Event>& e) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  if (running_events_.empty()) trace::execution_begin(e);
  running_events_.push_back(e);
  current_event_ = e;
  trace::active_ops_counter(running_events_.size());
}

void EventQueue::switch_dpu_binary(const std::shared_ptr<Event>& e) {
  const std::string required = required_binary_for(e, current_binary_path_);
  if (required.empty() || required == current_binary_path_) {
    if (current_binary_path_.empty())
      current_binary_path_ = DpuRuntime::get().get_default_binary_path();
    return;
  }

  log_binary_switch(current_binary_path_, required);
  trace::jit_binary_switch(current_binary_path_, required);

  // dpu_load replaces the code on every DPU, so nothing else may be in flight.
  while (true) {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    if (running_events_.size() <= 1 &&
        (running_events_.empty() || running_events_.front() == e))
      break;
    std::this_thread::yield();
  }

  VECTORDPU_NOTE(binary_switches);
  DPU_ASSERT(dpu_load(DpuRuntime::get().dpu_set(), required.c_str(), nullptr));
  current_binary_path_ = required;
  log_binary_loaded();
}

void EventQueue::dispatch(const std::shared_ptr<Event>& e) {
  log_dispatch(e, "begin");
  switch (e->op) {
    case Event::OperationType::FENCE:
      VECTORDPU_NOTE(fences);
      add_fence(e);
      break;

    case Event::OperationType::COMPUTE:
      VECTORDPU_NOTE(compute_launches);
      e->started = true;
      launch_compute(e);
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
  log_dispatch(e, "end");
}

void EventQueue::requeue_after_oom(const std::shared_ptr<Event>& e) {
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

  log_oom_freeing(e);
  auto& alloc = DpuRuntime::get().get_allocator();
  for (auto& out : outputs_to_free) alloc.deallocate_upmem_vector(out.get());

  {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    operations_.push_front(e);
  }
  log_oom_requeued(e);
}

bool EventQueue::process_next() {
  std::shared_ptr<Event> e = take_next_operation();
  if (!e) return NO_PROGRESS;

  log_next_operation(e);
  trace::inqueue_end(e);

  await_dependencies(e);
  grow_fusion_batch(e);
  await_jit_binary(e);

  name_event(e);
  log_launch(e);

  begin_running(e);
  compile_kernel_if_unbatched(e);
  switch_dpu_binary(e);

  try {
    dispatch(e);
  } catch (const DpuOOMException&) {
    log_oom_caught(e);
#if !ENABLE_OOM_RECOVERY
    throw DpuOOMException("DPU OOM: event id=" + std::to_string(e->id));
#else
    VECTORDPU_NOTE(oom_retries);
    if (++e->oom_retries > OOM_RECOVERY_RETRIES)
      throw DpuOOMException("DPU OOM: event id=" + std::to_string(e->id) +
                            " failed after " +
                            std::to_string(OOM_RECOVERY_RETRIES) + " retries");
    requeue_after_oom(e);
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

  // Hold back events a vfuse candidate is likely to want: hfusing them now
  // spends an extra_output slot and blocks the deeper vertical chain.  A fresh
  // scalar op is usually about to be consumed by an accumulator (linreg's
  // `error += dx[j]*dw[j]`), and an output marked for absorption will be
  // inlined into its consumer anyway.
  if (e->is_scalar && e->rpn_ops.empty()) return false;
  if (e->output && !e->output->absorbed_rpn.empty()) return false;

  return try_hfuse(last, e);
#else
  return false;
#endif
}

// Waits for room in the queue, draining it if a producer has run ahead.
void EventQueue::await_queue_space() {
  while (operations_.size() + running_events_.size() >= max_queue_depth_) {
    // process_next needs the mutex; we hold it recursively via submit.
    mtx_.unlock();
    process_next();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    mtx_.lock();
  }
}

// Merges `e` into the tail of the queue, then walks backwards collapsing any
// chain the merge just made fusable (K1->K2->K3 becomes one kernel).
bool EventQueue::fuse_into_queue_tail(const std::shared_ptr<Event>& e) {
#if PIPELINE
  if (e->op != Event::OperationType::COMPUTE || operations_.empty())
    return false;
  if (!try_fuse(operations_.back(), e)) return false;

  while (operations_.size() >= 2) {
    auto& prev = operations_[operations_.size() - 2];
    auto& tail = operations_.back();
    if (!try_fuse(prev, tail)) break;
    operations_.pop_back();
  }
  return true;
#else
  (void)e;
  return false;
#endif
}

void EventQueue::enqueue(const std::shared_ptr<Event>& e) {
#if JIT
  // Only a non-COMPUTE event ends the run of fusable kernels, so only that
  // is a safe point to flush.  Reductions stay batchable so process_next's
  // look-ahead can absorb them instead of forcing a mid-stream binary swap.
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
  // Record how to recompute this output so a later consumer can inline it
  // instead of reading MRAM (see EventQueue::expand_absorbed_inputs).
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

void EventQueue::submit(std::shared_ptr<Event> e) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  await_queue_space();

  e->id = counter_++;
  e->max_id = e->id;
  VECTORDPU_NOTE(events_submitted);

#if PIPELINE
  expand_absorbed_inputs(e);
#endif

  trace::event_enqueued(e, operations_, running_events_);
  trace::active_ops_counter(operations_.size());

  if (!fuse_into_queue_tail(e)) enqueue(e);
}
