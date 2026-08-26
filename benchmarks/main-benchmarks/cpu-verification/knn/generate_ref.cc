#include <benchmark.h>
#include <omp.h>

#include <algorithm>
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
    std::cerr << "Failed to open " << filename << std::endl;
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

void write_row_major_data() {
  const uint64_t chunk_rows = 1u << 20;
  std::ofstream out("data/AoS/rows.bin", std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "Failed to open data/AoS/rows.bin" << std::endl;
    std::exit(1);
  }

  std::vector<T> rows(chunk_rows * DIM);
  for (uint64_t base = 0; base < N; base += chunk_rows) {
    uint64_t count = std::min<uint64_t>(chunk_rows, N - base);
    for (uint64_t i = 0; i < count; i++) {
      for (uint32_t d = 0; d < DIM; d++) {
        rows[i * DIM + d] = (T)(((base + i) * (DIM + 1) + d) % 256);
      }
    }
    out.write(reinterpret_cast<const char*>(rows.data()),
              count * DIM * sizeof(T));
    if (!out.good()) {
      std::cerr << "Short write to data/AoS/rows.bin" << std::endl;
      std::exit(1);
    }
  }
}

int main() {
  system("mkdir -p data/AoS data/SoA");
  std::cout << "Generating knn reference: N=" << N << " DIM=" << DIM
            << " K=" << K << std::endl;

  std::vector<T> query(DIM);
  for (uint32_t d = 0; d < DIM; d++) {
    query[d] = (T)(d * 17 % 128);
  }
  write_bin("data/ref_query.bin", query.data(), DIM * sizeof(T));

  const bool data_only = bench_ref_data_only();

  // Process one column at a time to avoid DIM*N host allocation
  std::vector<RED_T> sq_dists;
  if (!data_only) {
    sq_dists.assign(N, 0);
  }
  for (uint32_t d = 0; d < DIM; d++) {
    std::vector<T> col(N);
    for (uint64_t i = 0; i < N; i++) {
      col[i] = (T)((i * (DIM + 1) + d) % 256);
    }

    if (!data_only) {
      RED_T qd = query[d];
#pragma omp parallel for
      for (uint64_t i = 0; i < N; i++) {
        RED_T diff = (RED_T)col[i] - qd;
        sq_dists[i] += diff * diff;
      }
    }

    write_bin("data/SoA/col_" + std::to_string(d) + ".bin", col.data(),
              N * sizeof(T));
  }
  write_row_major_data();

  if (data_only) {
    std::cout << "Reference input generation complete." << std::endl;
    return 0;
  }

  // Benchmark (time only the distance+selection computation, not data init)
  auto t0 = std::chrono::high_resolution_clock::now();
  for (uint32_t iter = 0; iter < std::max(iterations, 1u); iter++) {
    std::vector<RED_T> tmp(N, 0);
    for (uint32_t d = 0; d < DIM; d++) {
      RED_T qd = query[d];
// Re-derive col on the fly to avoid re-loading
#pragma omp parallel for
      for (uint64_t i = 0; i < N; i++) {
        RED_T diff = (RED_T)((T)((i * (DIM + 1) + d) % 256)) - qd;
        tmp[i] += diff * diff;
      }
    }
    if (K == 1) {
      RED_T m = *std::min_element(tmp.begin(), tmp.end());
      DoNotOptimize(m);
    } else {
      std::partial_sort(tmp.begin(), tmp.begin() + K, tmp.end());
      DoNotOptimize(tmp[0]);
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() /
              std::max(iterations, 1u);
  std::cout << "cpu_baseline (ms): " << ms << std::endl;

  // Find K nearest from pre-computed sq_dists
  std::vector<RED_T> result(K);
  if (K == 1) {
    result[0] = *std::min_element(sq_dists.begin(), sq_dists.end());
  } else {
    std::partial_sort(sq_dists.begin(), sq_dists.begin() + K, sq_dists.end());
    for (uint32_t k = 0; k < K; k++) {
      result[k] = sq_dists[k];
    }
  }

  std::cout << "K nearest squared distances:";
  for (uint32_t k = 0; k < K; k++) {
    std::cout << " " << result[k];
  }
  std::cout << std::endl;

  write_bin("data/ref_res.bin", result.data(), K * sizeof(RED_T));
  std::cout << "Done." << std::endl;
  return 0;
}
