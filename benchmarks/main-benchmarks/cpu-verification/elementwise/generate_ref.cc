#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "Param.h"

#define DoNotOptimize(x) asm volatile("" : : "r,m"(x) : "memory")

void write_bin(const std::string& filename, const void* data, size_t size) {
  std::ofstream out(filename, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "Failed to open " << filename << " for writing" << std::endl;
    std::exit(1);
  }
  out.write(reinterpret_cast<const char*>(data), size);
  out.flush();
  if (!out.good()) {
    std::cerr << "Short write to " << filename << " (expected " << size
              << " bytes)" << std::endl;
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  std::srand(seed);  // Use seed from Param.h

  std::cout << "Generating reference data for N=" << N << " using seed=" << seed
            << "..." << std::endl;

  std::vector<T> a(N);
  std::vector<T> b(N);
  std::vector<T> res(N);

  // Deterministic per-index RNG so the values do not depend on the
  // OpenMP partitioning of the parallel loop. Earlier code used a
  // thread-local rand_r seed, which made a[i]/b[i] depend on the chunk
  // boundaries libgomp picked — and at N > 2^31 the partitioning was
  // observed to drift between runs, which silently produced inconsistent
  // reference data across the three programming models.
  omp_set_num_threads(24);
  auto splitmix64 = [](uint64_t x) -> uint64_t {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
  };

#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < N; i++) {
    a[i] = (T)(splitmix64((uint64_t)seed * 2 + (i << 1)) % 10);
    b[i] = (T)(splitmix64((uint64_t)seed * 2 + (i << 1) + 1) % 10);
  }

  std::cout << "Starting warmup..." << std::endl;
  for (uint32_t iter = 0; iter < warmup_iterations; iter++) {
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; i++) {
      res[i] = OPERATION(a[i], b[i]);
    }
    DoNotOptimize(res.data());
  }

  std::cout << "Starting benchmark..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  for (uint32_t iter = 0; iter < iterations; iter++) {
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; i++) {
      res[i] = OPERATION(a[i], b[i]);
    }
    DoNotOptimize(res.data());
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = (end - start);

  std::cout << "cpu_baseline (ms): " << elapsed.count() << std::endl;

  std::cout << "Writing binary files to ./data/ ..." << std::endl;
  write_bin("data/ref_a.bin", a.data(), N * sizeof(T));
  write_bin("data/ref_b.bin", b.data(), N * sizeof(T));
  write_bin("data/ref_res.bin", res.data(), N * sizeof(T));

  std::cout << "Reference generation complete." << std::endl;
  return 0;
}
