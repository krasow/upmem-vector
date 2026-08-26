#include <benchmark.h>
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
  system("mkdir -p data");
  std::srand(seed);

  std::cout << "Generating red reference data for N=" << N << "..."
            << std::endl;

  std::vector<T> a(N);
  T res = 0;

  // Data Init
  for (uint64_t i = 0; i < N; i++) {
    a[i] = i % 1000;
  }

  if (bench_ref_data_only()) {
    std::cout << "Writing input files to ./data/ ..." << std::endl;
    write_bin("data/ref_t1.bin", a.data(), N * sizeof(T));
    return 0;
  }

  // Warmup
  for (uint32_t iter = 0; iter < warmup_iterations; iter++) {
    T local_res = 0;
#pragma omp parallel for reduction(+ : local_res)
    for (uint64_t i = 0; i < N; i++) {
      local_res += a[i];
    }
    res = local_res;
    DoNotOptimize(res);
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (uint32_t iter = 0; iter < iterations; iter++) {
    T local_res = 0;
#pragma omp parallel for reduction(+ : local_res)
    for (uint64_t i = 0; i < N; i++) {
      local_res += a[i];
    }
    res = local_res;
    DoNotOptimize(res);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = (end - start);

  std::cout << "cpu_baseline (ms): " << elapsed.count() << std::endl;
  std::cout << "CPU result: " << res << std::endl;

  std::cout << "Writing binary files to ./data/ ..." << std::endl;
  write_bin("data/ref_t1.bin", a.data(), N * sizeof(T));
  write_bin("data/ref_res.bin", &res, sizeof(T));

  return 0;
}
