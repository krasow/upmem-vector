#pragma once

#include <common.h>
#include <config.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "vectordesc.h"

namespace logcat {
inline constexpr std::string_view EVENT = "EVENT";
inline constexpr std::string_view EVENT_QUEUE = "EVENT-QUEUE";
inline constexpr std::string_view FUSION = "FUSION";
inline constexpr std::string_view JIT_COMPILER = "JIT";
inline constexpr std::string_view JIT_DEBUG = "JIT-DBG";
inline constexpr std::string_view JIT_FOREACH = "JIT-FOREACH";
inline constexpr std::string_view MEMORY = "MEMORY";
inline constexpr std::string_view OOM = "OOM";
inline constexpr std::string_view QUEUE_APPEND = "QUEUE-APPEND";
inline constexpr std::string_view QUEUE_DISPATCH = "QUEUE-DISPATCH";
inline constexpr std::string_view QUEUE_EXEC = "QUEUE-EXEC";
inline constexpr std::string_view QUEUE_HEARTBEAT = "QUEUE-HEARTBEAT";
inline constexpr std::string_view QUEUE_JIT = "QUEUE-JIT";
inline constexpr std::string_view QUEUE_NEXT = "QUEUE-NEXT";
inline constexpr std::string_view QUEUE_WAIT = "QUEUE-WAIT";
inline constexpr std::string_view RUNTIME = "RUNTIME";
inline constexpr std::string_view TASK = "TASK";
inline constexpr std::string_view TRACE_IO = "TRACE";
inline constexpr std::string_view TRANSFER = "TRANSFER";
inline constexpr std::string_view VECTOR_DESC = "VECTOR-DESC";
}  // namespace logcat

// Where log lines go, and at what level, for the duration of a scope.
//
// Deliberately not a member of Logger: the logger is created when the runtime
// initializes and destroyed on shutdown, so a capture opened before the first
// allocation -- or still open when the runtime goes away -- has to outlive it.
// Frames are held by pointer so growing the stack cannot move a buffer a line
// is being written into.
//
// Pushes and pops belong on the thread doing the work.  The event queue drains
// on its caller, so that is also the thread the lines are emitted from; nothing
// here tries to take another thread's output out from under it.
class LogSink {
 public:
  static LogSink& get() {
    static LogSink instance;
    return instance;
  }

  // Redirect the logger into a fresh buffer.  A level of 0 or less redirects
  // without changing what is emitted.
  void push(int level) {
    auto frame = std::make_unique<Frame>();
    frame->level = level > 0 ? level : level_override();
    frames_.push_back(std::move(frame));
  }

  // Everything emitted since the matching push.  A nested capture takes the
  // lines of its own scope; the enclosing one does not also see them.
  std::string pop() {
    if (frames_.empty()) return {};
    std::string text = frames_.back()->buffer.str();
    frames_.pop_back();
    return text;
  }

  std::ostream* stream() {
    return frames_.empty() ? nullptr : &frames_.back()->buffer;
  }

  // 0 when nothing is capturing, so the logger falls back to its own level.
  int level_override() const {
    return frames_.empty() ? 0 : frames_.back()->level;
  }

  size_t depth() const { return frames_.size(); }

 private:
  struct Frame {
    std::ostringstream buffer;
    int level;
  };

  LogSink() = default;
  std::vector<std::unique_ptr<Frame>> frames_;
};

class Logger {
  using clock = std::chrono::steady_clock;

  std::recursive_mutex mtx_;
  std::ostream& default_stream_;
  clock::time_point start_;
  int level_;

  // A capture, if one is open, otherwise wherever this logger was pointed.
  std::ostream& sink() {
    std::ostream* captured = LogSink::get().stream();
    return captured ? *captured : default_stream_;
  }

  static int env_log_level() {
    const char* value = std::getenv("VECTORDPU_LOG_LEVEL");
    if (value == nullptr || value[0] == '\0') return 0;
    return std::atoi(value);
  }

  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(clock::now() - start_)
        .count();
  }

 public:
  Logger(std::ostream& stream = std::cout)
      : default_stream_(stream),
        start_(clock::now()),
        level_(env_log_level()) {}

  // What is emitted right now: the level of the innermost capture if one is
  // open, else the process level from VECTORDPU_LOG_LEVEL.  Neither can get
  // past the ENABLE_DPU_LOGGING the library was compiled with -- the call
  // sites above that are not in the binary at all.
  int level() const {
    int override_level = LogSink::get().level_override();
    return override_level > 0 ? override_level : level_;
  }

  bool enabled(int level = 1) const { return this->level() >= level; }

  // Proxy object that locks the mutex for the duration of the object
  struct Lock {
    std::ostream* stream;
    std::unique_lock<std::recursive_mutex> lock;
    bool enabled;

    Lock(Logger& logger, std::string_view category = {}, int level = 1)
        : stream(logger.enabled(level) ? &logger.sink() : nullptr),
          lock(stream ? std::unique_lock<std::recursive_mutex>(logger.mtx_)
                      : std::unique_lock<std::recursive_mutex>()),
          enabled(stream != nullptr) {
      if (!enabled) return;
      std::ios::fmtflags flags = stream->flags();
      std::streamsize precision = stream->precision();
      char fill = stream->fill();
      *stream << "[+" << std::fixed << std::setprecision(3)
              << logger.elapsed_ms() << "ms] ";
      if (!category.empty()) *stream << "[" << category << "] ";
      stream->flags(flags);
      stream->precision(precision);
      stream->fill(fill);
    }

    Lock* operator->() { return this; }

    Lock& first() { return *this; }

    Lock& second() {
      if (enabled) *stream << "\n\t\t";
      return *this;
    }

    // For generic types
    template <typename T>
    Lock& operator<<(const T& value) {
      if (enabled) *stream << value;
      return *this;
    }

    // For manipulators like std::endl
    using Manip = std::ostream& (*)(std::ostream&);
    Lock& operator<<(Manip manip) {
      if (enabled) manip(*stream);
      return *this;
    }
  };

  Lock lock(std::string_view category = {}, int level = 1) {
    return Lock(*this, category, level);
  }
};

char const* kernel_id_to_string(KernelID kernel_id);

void print_vector_desc(Logger& logger, detail::VectorDescRef desc,
                       uint32_t reserved);

void log_allocation(Logger& logger, const std::type_info& type, size_t n,
                    uint64_t vector_id, size_t memory_bytes,
                    bool has_materialized_layout, std::string_view debug_name,
                    const char* debug_file, int debug_line,
                    bool is_allocation = true);

void log_allocation(Logger& logger, const char* type_name, size_t n,
                    uint64_t vector_id, size_t memory_bytes,
                    bool has_materialized_layout, std::string_view debug_name,
                    const char* debug_file, int debug_line,
                    bool is_allocation = true);

#define log_deallocation(logger, type, n, vector_id, memory_bytes,            \
                         has_materialized_layout, debug_name, debug_file,     \
                         debug_line)                                          \
  log_allocation(logger, type, n, vector_id, memory_bytes,                    \
                 has_materialized_layout, debug_name, debug_file, debug_line, \
                 false)

void log_dpu_launch_args(Logger& logger, const DPU_LAUNCH_ARGS* args,
                         uint32_t nr_of_dpus, size_t fused_event_ops = 0,
                         size_t fused_event_chains = 0,
                         std::string_view kernel_hash = {});
