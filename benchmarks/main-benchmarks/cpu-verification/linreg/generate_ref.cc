#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Param.h"

#ifndef RED_T
typedef int64_t RED_T;
#endif

// Helper to prevent compiler from optimizing away calculations
template <typename T>
void DoNotOptimize(T const& value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

void save_bin(const std::string& filename, void* data, size_t size) {
  std::ofstream out(filename, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "Failed to open " << filename << " for writing!" << std::endl;
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

void save_padded_rows(const std::vector<std::vector<T>>& x_cols,
                      const std::vector<T>& y) {
  const size_t logical_elems = DIM + 1;
  const size_t row_bytes = (logical_elems * sizeof(T) + 7) & ~size_t(7);
  const size_t row_elems = row_bytes / sizeof(T);
  const uint64_t chunk_rows = 1u << 20;

  std::ofstream out("./data/AoS/rows.bin", std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "Failed to open ./data/AoS/rows.bin for writing!" << std::endl;
    std::exit(1);
  }

  std::vector<T> rows(chunk_rows * row_elems, 0);
  for (uint64_t base = 0; base < N; base += chunk_rows) {
    uint64_t count = std::min<uint64_t>(chunk_rows, N - base);
    std::fill(rows.begin(), rows.begin() + count * row_elems, 0);
    for (uint64_t i = 0; i < count; i++) {
      for (uint32_t j = 0; j < DIM; j++) {
        rows[i * row_elems + j] = x_cols[j][base + i];
      }
      rows[i * row_elems + DIM] = y[base + i];
    }
    out.write(reinterpret_cast<const char*>(rows.data()), count * row_bytes);
    if (!out.good()) {
      std::cerr << "Short write to ./data/AoS/rows.bin" << std::endl;
      std::exit(1);
    }
  }
}

int main() {
  // Ensure data directory exists
  system("mkdir -p ./data/AoS ./data/SoA");
  std::cout << "Generating reference data for N=" << N << ", DIM=" << DIM
            << "..." << std::endl;

  std::vector<std::vector<T>> host_x_cols(DIM, std::vector<T>(N));
  std::vector<T> host_y(N);

  // Initialize data
  for (uint32_t i = 0; i < N; i++) {
    for (uint32_t j = 0; j < DIM; j++) {
      host_x_cols[j][i] = (i * (DIM + 1) + j) % 256;
    }
    host_y[i] = (i * (DIM + 1) + DIM) % 256;
  }

  std::vector<RED_T> expected_grads(DIM, 0);

  // Warmup
  std::cout << "Starting warmup..." << std::endl;
  for (uint32_t iter = 0; iter < warmup_iterations; iter++) {
    std::vector<RED_T> local_grads(DIM, 0);
#pragma omp parallel
    {
      std::vector<RED_T> private_grads(DIM, 0);
#pragma omp for
      for (uint32_t i = 0; i < N; i++) {
        T err = -host_y[i];
        for (uint32_t j = 0; j < DIM; j++) {
          T s1 = host_x_cols[j][i] >> (scaling_shift / 2);
          T s2 = err >> (scaling_shift - scaling_shift / 2);
          private_grads[j] += (RED_T)s1 * (RED_T)s2;
        }
      }
#pragma omp critical
      {
        for (uint32_t j = 0; j < DIM; j++) {
          local_grads[j] += private_grads[j];
        }
      }
    }
    expected_grads = local_grads;
    DoNotOptimize(expected_grads.data());
  }

  std::cout << "Starting benchmark..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  for (uint32_t iter = 0; iter < iterations; iter++) {
    std::vector<RED_T> local_grads(DIM, 0);

#pragma omp parallel
    {
      std::vector<RED_T> private_grads(DIM, 0);
#pragma omp for
      for (uint32_t i = 0; i < N; i++) {
        T err = -host_y[i];
        for (uint32_t j = 0; j < DIM; j++) {
          T s1 = host_x_cols[j][i] >> (scaling_shift / 2);
          T s2 = err >> (scaling_shift - scaling_shift / 2);
          private_grads[j] += (RED_T)s1 * (RED_T)s2;
        }
      }

#pragma omp critical
      {
        for (uint32_t j = 0; j < DIM; j++) {
          local_grads[j] += private_grads[j];
        }
      }
    }
    expected_grads = local_grads;
    DoNotOptimize(expected_grads.data());
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = (end - start);

  std::cout << "cpu_baseline (ms): " << elapsed.count() << std::endl;

  std::cout << "Writing binary files to ./data/ ..." << std::endl;
  for (uint32_t j = 0; j < DIM; j++) {
    save_bin("./data/SoA/x_col_" + std::to_string(j) + ".bin",
             host_x_cols[j].data(), N * sizeof(T));
  }
  save_padded_rows(host_x_cols, host_y);
  save_bin("./data/SoA/y.bin", host_y.data(), N * sizeof(T));
  save_bin("./data/ref_grads.bin", expected_grads.data(), DIM * sizeof(RED_T));

  std::cout << "Final gradients: ";
  for (uint32_t i = 0; i < DIM; i++) {
    std::cout << (long long)expected_grads[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
