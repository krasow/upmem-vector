#include "allocator.h"

#include <algorithm>
#include <stdexcept>

#include "logger.inl"

allocator::allocator(uint32_t start_addr, std::size_t total_size,
                     std::size_t num_dpus)
    : start_addr_(start_addr), total_size_(total_size), num_dpus_(num_dpus) {
  // Initialize internal state, but do NOT pre-allocate vectors
  ptrs_.resize(num_dpus_, start_addr_);  // all start at base address
  sizes_.resize(num_dpus_, total_size_ / num_dpus_);
  offsets_.resize(num_dpus_, 0);
  free_list_.resize(num_dpus_);
  allocations_.clear();
}

vector_desc allocator::allocate_upmem_vector(std::size_t n,
                                             std::size_t size_type) {
  std::size_t num_dpus = this->num_dpus_;
  vector<uint32_t> vec_ptrs(num_dpus);
  vector<uint32_t> vec_sizes(num_dpus);

  size_t size_per_dpu = n / num_dpus;
  size_t remainder = n % num_dpus;

  for (size_t i = 0; i < num_dpus; i++) {
    size_t alloc_size = (size_per_dpu + (i < remainder ? 1 : 0)) * size_type;
    uint32_t addr = allocate(i, alloc_size);  // use bump/free-list allocator

    vec_ptrs[i] = addr;
    vec_sizes[i] = alloc_size;
  }

  return std::make_pair(vec_ptrs, vec_sizes);
}

void allocator::deallocate_upmem_vector(vector_desc& data) {
  for (size_t i = 0; i < num_dpus_; ++i) {
    uint32_t addr = data.first[i];
    size_t size = data.second[i];
    deallocate(i, addr, size);
  }
}

uint32_t allocator::allocate(std::size_t dpu_id, std::size_t n) {
  if (dpu_id >= num_dpus_) throw std::out_of_range("Invalid DPU ID");

  auto& flist = free_list_[dpu_id];

  // Best-fit free block
  auto best_it = flist.end();
  size_t best_size = SIZE_MAX;
  for (auto it = flist.begin(); it != flist.end(); ++it) {
    if (it->size >= n && it->size < best_size) {
      best_it = it;
      best_size = it->size;
    }
  }

  if (best_it != flist.end()) {
    uint32_t addr = best_it->addr;
    if (best_it->size > n) {
      best_it->addr += n;
      best_it->size -= n;
    } else {
      flist.erase(best_it);
    }
    allocations_.push_back(reinterpret_cast<void*>(addr));
    return addr;
  }

  // Bump pointer allocation
  if (offsets_[dpu_id] + n > sizes_[dpu_id]) {
    throw std::runtime_error("DPU out of memory!");
  }

  uint32_t addr = ptrs_[dpu_id] + offsets_[dpu_id];
  offsets_[dpu_id] += n;
  allocations_.push_back(reinterpret_cast<void*>(addr));
  return addr;
}

void allocator::deallocate(std::size_t dpu_id, uint32_t addr, size_t size) {
  if (dpu_id >= num_dpus_) throw std::out_of_range("Invalid DPU ID");

  FreeBlock new_block{addr, size};
  auto& flist = free_list_[dpu_id];

  // Insert sorted by address
  auto it = std::lower_bound(
      flist.begin(), flist.end(), new_block,
      [](const FreeBlock& a, const FreeBlock& b) { return a.addr < b.addr; });
  flist.insert(it, new_block);

  // Merge adjacent blocks
  for (size_t i = 0; i + 1 < flist.size();) {
    if (flist[i].addr + flist[i].size == flist[i + 1].addr) {
      flist[i].size += flist[i + 1].size;
      flist.erase(flist.begin() + i + 1);
    } else {
      ++i;
    }
  }

  auto ait = std::find(allocations_.begin(), allocations_.end(),
                       reinterpret_cast<void*>(addr));
  if (ait != allocations_.end()) allocations_.erase(ait);
}

vector_desc allocator::get_vector_desc() const {
  return std::make_pair(ptrs_, sizes_);
}
