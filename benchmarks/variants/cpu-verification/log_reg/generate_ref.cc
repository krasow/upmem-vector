#include <omp.h>

#include <chrono>
#include <cmath>
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

static inline T sigmoid_cpu(T x) {
  if (x >= 15.0f) return 1.0f;
  if (x <= -15.0f) return 0.0f;
  if (x == 0.0f) return 0.5f;

  float sum = 1.0f;
  float temp = 1.0f;
  for (int i = 1; i < 101; ++i) {
    temp = temp * (-(float)x) / (float)i;
    sum = sum + temp;
  }
  return (T)(1.0f / (1.0f + sum));
}

int main(int argc, char** argv) {
  system("mkdir -p data");
  std::srand(seed);

  std::cout << "Generating log_reg reference data for N=" << N << "..."
            << std::endl;

  // elements: [X | Y] row-major, (dim+1) floats per row
  std::vector<T> elements(N * (dim + 1));
  std::vector<T> weights(dim, 0.0f);
  std::vector<T> grads(dim, 0.0f);

  // Data Init
  for (uint64_t i = 0; i < N; i++) {
    T dot = 0;
    for (uint32_t j = 0; j < dim; j++) {
      T v = (T)((int)(i + j) % 11 - 5);
      elements[i * (dim + 1) + j] = v;
      dot += v;
    }
    elements[i * (dim + 1) + dim] = (dot >= 0) ? 1.0f : 0.0f;
  }

  // Benchmark
  for (uint32_t iter = 0; iter < iterations; iter++) {
    std::fill(grads.begin(), grads.end(), 0.0f);
#pragma omp parallel
    {
      std::vector<T> local_grads(dim, 0.0f);
#pragma omp for
      for (uint64_t i = 0; i < N; i++) {
        T dot = 0;
        for (uint32_t j = 0; j < dim; j++) {
          dot += elements[i * (dim + 1) + j] * weights[j];
        }
        T e = sigmoid_cpu(dot) - elements[i * (dim + 1) + dim];
        for (uint32_t j = 0; j < dim; j++) {
          local_grads[j] += e * elements[i * (dim + 1) + j];
        }
      }
#pragma omp critical
      {
        for (uint32_t j = 0; j < dim; j++) grads[j] += local_grads[j];
      }
    }
    DoNotOptimize(grads.data());
  }

  std::cout << "Writing binary files to ./data/ ..." << std::endl;
  write_bin("data/ref_t1.bin", elements.data(), N * (dim + 1) * sizeof(T));
  write_bin("data/ref_w.bin", weights.data(), dim * sizeof(T));
  write_bin("data/ref_grads.bin", grads.data(), dim * sizeof(T));

  return 0;
}
