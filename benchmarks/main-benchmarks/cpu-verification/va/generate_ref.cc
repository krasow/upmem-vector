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

  std::cout << "Generating va reference data for N=" << N << "..." << std::endl;

  std::vector<T> a(N);
  std::vector<T> b(N);
  std::vector<T> res(N);

  // Data Init
  for (uint64_t i = 0; i < N; i++) {
    a[i] = (1 + i) % 1000;
    b[i] = (2 + i) % 1000;
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (uint32_t iter = 0; iter < iterations; iter++) {
#pragma omp parallel for
    for (uint64_t i = 0; i < N; i++) {
      res[i] = a[i] + b[i];
    }
    DoNotOptimize(res.data());
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = (end - start);

  std::cout << "cpu_baseline (ms): " << elapsed.count() << std::endl;

  std::cout << "Writing binary files to ./data/ ..." << std::endl;
  write_bin("data/ref_t1.bin", a.data(), N * sizeof(T));
  write_bin("data/ref_t2.bin", b.data(), N * sizeof(T));
  write_bin("data/ref_res.bin", res.data(), N * sizeof(T));

  return 0;
}
