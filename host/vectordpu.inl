#pragma once

#include <cassert>
#include <cstdio>
#include <functional>
#include <memory>

#include "logger.h"
#include "runtime.h"
#include "vectordpu.h"

#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

// ============================
// DPU Vector
// ============================
template <typename T>
dpu_vector<T>::dpu_vector(uint32_t n, std::string_view name,
                          std::source_location loc)
    : size_(n),
      debug_name(name.data()),
      debug_file(loc.file_name()),
      debug_line(loc.line()) {
  auto& runtime = DpuRuntime::get();

  if (runtime.is_initialized() == false) {
    // throw std::runtime_error("DPU runtime not initialized!");
    runtime.init(NR_DPUS);
  }
  Logger& logger = runtime.get_logger();
  logger.lock() << "ALLOCATING DPU VECTOR " << debug_name << " OF SIZE " << n
                << " FROM " << debug_file << ":" << debug_line << std::endl;
  data_ = runtime.get_allocator().allocate_upmem_vector(n, sizeof(T));

#if ENABLE_DPU_LOGGING >= 1
  log_allocation(typeid(T), n, debug_name, debug_file, debug_line);
#endif
}

template <typename T>
dpu_vector<T>::dpu_vector(const dpu_vector& other) {
  if (this != &other) {
    data_ = other.data_;
    size_ = other.size_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
    copied = true;
  }
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[dpu_vector] COPY CONSTRUCTOR at " << debug_name
                << " OF SIZE " << size_ << " FROM " << debug_file << ":"
                << debug_line << std::endl;
#endif
}

template <typename T>
dpu_vector<T>::dpu_vector(dpu_vector&& other) noexcept {
  if (this != &other) {
    data_ = other.data_;
    size_ = other.size_;
    debug_name = other.debug_name;
    debug_file = other.debug_file;
    debug_line = other.debug_line;
    other.size_ = 0;
  }
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[dpu_vector] MOVE CONSTRUCTOR at " << debug_name
                << " OF SIZE " << size_ << " FROM " << debug_file << ":"
                << debug_line << std::endl;
#endif
}

template <typename T>
dpu_vector<T>::~dpu_vector() {
  if (copied == true) {
    // Nothing to deallocate
    return;
  }
  auto& runtime = DpuRuntime::get();
#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = runtime.get_logger();
  logger.lock() << "[dpu_vector] DEALLOCATING DPU VECTOR " << debug_name
                << " FROM " << debug_file << ":" << debug_line << std::endl;
#endif
  runtime.get_allocator().deallocate_upmem_vector(data_);
}

template <typename T>
vector<uint32_t> dpu_vector<T>::data() const {
  // data_ is vector_desc std::pair<vector<uint32_t>, vector<uint32_t>>
  // where first element is vector of pointers to DPU memory per DPU
  // and second element is vector of sizes per DPU
  return data_.first;
}

template <typename T>
uint32_t dpu_vector<T>::size() const {
  return size_;
}

void vec_xfer_to_dpu(char* cpu_vec, vector_desc& desc) {
  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;

  uint32_t idx_dpu = 0;
  size_t element = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &(cpu_vec[element])));
    element += desc.second[idx_dpu];
  }

  uint32_t mram_location = desc.first[0];
  size_t xfer_size = desc.second[0];

  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                            DPU_MRAM_HEAP_POINTER_NAME, mram_location,
                            xfer_size, DPU_XFER_ASYNC));
}

void vec_xfer_from_dpu(char* cpu_vec, vector_desc& desc) {
  auto& runtime = DpuRuntime::get();
  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;

  uint32_t idx_dpu = 0;
  size_t element = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &(cpu_vec[element])));
    element += desc.second[idx_dpu];
  }

  uint32_t mram_location = desc.first[0];
  size_t xfer_size = desc.second[0];

  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                            DPU_MRAM_HEAP_POINTER_NAME, mram_location,
                            xfer_size, DPU_XFER_ASYNC));
}

template <typename T>
dpu_vector<T> dpu_vector<T>::from_cpu(std::vector<T>& cpu_vec,
                                      std::string_view name,
                                      std::source_location loc) {
  dpu_vector<T> vec(cpu_vec.size(), name, loc);
  // .data returns a std::pair<vector<uint32_t>, vector<uint32_t>>
  // the first element is vector of pointers to DPU memory per DPU
  // the second element is vector of sizes per DPU
  auto desc = vec.data_desc();

#if ENABLE_DPU_LOGGING >= 2
  print_vector_desc(desc);
#endif

  char* cpu_buffer = reinterpret_cast<char*>(cpu_vec.data());
  auto bound_cb = std::bind(vec_xfer_to_dpu, cpu_buffer, std::ref(desc));

  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();
  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::DPU_TRANSFER, bound_cb);
  event_queue.submit(e);

  // TODO have some sort of dependency analysis
  while (e->finished == false) {
    event_queue.process_next();
  }

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] HOST->DPU XFER " << cpu_vec.size()
                << " elements to DPUs" << std::endl;
#endif
  return vec;
}

template <typename T>
vector<T> dpu_vector<T>::to_cpu() {
  auto desc = this->data_desc();  // pair< vector<uint32_t>, vector<uint32_t> >

#if ENABLE_DPU_LOGGING >= 2
  print_vector_desc(desc);
#endif

  // Allocate CPU buffer large enough to hold all data
  vector<T> cpu_vec(this->size());
  char* cpu_buffer = reinterpret_cast<char*>(cpu_vec.data());
  auto bound_cb = std::bind(vec_xfer_from_dpu, cpu_buffer, std::ref(desc));

  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::HOST_TRANSFER, bound_cb);
  event_queue.submit(e);

  // TODO have some sort of dependency analysis
  while (e->finished == false) {
    event_queue.process_next();
  }

#if ENABLE_DPU_LOGGING >= 2
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[queue-append] DPU->HOST XFER " << cpu_vec.size()
                << " elements from DPUs" << std::endl;
#endif
  return cpu_vec;
}

template <typename T>
void internal_launch_binop(dpu_vector<T>& res, const dpu_vector<T>& lhs,
                           const dpu_vector<T>& rhs, KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();

  uint32_t nr_of_dpus = runtime.num_dpus();
  DPU_LAUNCH_ARGS args[nr_of_dpus];

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].kernel = static_cast<uint32_t>(kernel_id);
    args[i].is_binary = true;
    args[i].num_elements = lhs.size();
    args[i].size_type = sizeof(T);
    args[i].binary.lhs_offset = reinterpret_cast<uint32_t>(lhs.data()[i]);
    args[i].binary.rhs_offset = reinterpret_cast<uint32_t>(rhs.data()[i]);
    args[i].binary.res_offset = reinterpret_cast<uint32_t>(res.data()[i]);
  }

#ifdef ENABLE_DPU_LOGGING
  log_dpu_launch_args(args, nr_of_dpus);
#endif

  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;
  uint32_t idx_dpu = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
  }
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
                            sizeof(args[0]), DPU_XFER_DEFAULT));
  CHECK_UPMEM(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));
}

template <typename T>
dpu_vector<T> launch_binop(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs,
                           KernelID kernel_id) {
  assert(lhs.size() == rhs.size());
  dpu_vector<T> res(lhs.size());

  auto bound_cb = std::bind(internal_launch_binop<T>, res, lhs, rhs, kernel_id);
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
  e->res = res;
  event_queue.submit(e);

  // TODO have some sort of dependency analysis
  while (e->finished == false) {
    event_queue.process_next();
  }

  return res;
}

template <typename T>
void internal_launch_unary(dpu_vector<T>& res, const dpu_vector<T>& a,
                           KernelID kernel_id) {
  auto& runtime = DpuRuntime::get();

  uint32_t nr_of_dpus = runtime.num_dpus();
  DPU_LAUNCH_ARGS args[nr_of_dpus];

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].kernel = static_cast<uint32_t>(kernel_id);
    args[i].is_binary = false;
    args[i].num_elements = a.size();
    args[i].size_type = sizeof(T);
    args[i].unary.rhs_offset = reinterpret_cast<uint32_t>(a.data()[i]);
    args[i].unary.res_offset = reinterpret_cast<uint32_t>(res.data()[i]);
  }

#ifdef ENABLE_DPU_LOGGING
  log_dpu_launch_args(args, nr_of_dpus);
#endif

  dpu_set_t& dpu_set = runtime.dpu_set();
  dpu_set_t dpu;
  uint32_t idx_dpu = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
  }
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
                            sizeof(args[0]), DPU_XFER_DEFAULT));
  CHECK_UPMEM(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));
}

template <typename T>
dpu_vector<T> launch_unary(const dpu_vector<T>& a, KernelID kernel_id) {
  dpu_vector<T> res(a.size());

  auto bound_cb = std::bind(internal_launch_unary<T>, res, a, kernel_id);
  auto& runtime = DpuRuntime::get();
  auto& event_queue = runtime.get_event_queue();

  std::shared_ptr<Event> e =
      std::make_shared<Event>(Event::OperationType::COMPUTE, bound_cb);
  e->res = res;
  event_queue.submit(e);

  // TODO have some sort of dependency analysis
  while (e->finished == false) {
    event_queue.process_next();
  }

  return res;
}