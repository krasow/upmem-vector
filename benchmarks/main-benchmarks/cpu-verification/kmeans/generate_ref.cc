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

int divRoundClosest(const int n, const int d) {
  return ((n < 0) ^ (d < 0)) ? ((n - d / 2) / d) : ((n + d / 2) / d);
}

void write_columns(const std::vector<T>& elements) {
  const uint64_t chunk_rows = 1u << 20;
  std::vector<T> col(chunk_rows);
  for (uint32_t d = 0; d < dim; d++) {
    std::ofstream out("data/SoA/col_" + std::to_string(d) + ".bin",
                      std::ios::binary | std::ios::trunc);
    if (!out) {
      std::cerr << "Failed to open data/SoA/col_" << d << ".bin for writing"
                << std::endl;
      std::exit(1);
    }
    for (uint64_t base = 0; base < N; base += chunk_rows) {
      uint64_t count = std::min<uint64_t>(chunk_rows, N - base);
      for (uint64_t i = 0; i < count; i++) {
        col[i] = elements[(base + i) * dim + d];
      }
      out.write(reinterpret_cast<const char*>(col.data()), count * sizeof(T));
      if (!out.good()) {
        std::cerr << "Short write to data/SoA/col_" << d << ".bin" << std::endl;
        std::exit(1);
      }
    }
  }
}

int main(int argc, char** argv) {
  system("mkdir -p data/AoS data/SoA");
  std::srand(seed);

  std::cout << "Generating kmeans reference data for N=" << N << ", K=" << k
            << "..." << std::endl;

  std::vector<T> elements(N * dim);
  std::vector<T> centroids(k * dim);
  std::vector<T> centroids_init(k * dim);

  // Data Init
  for (uint64_t i = 0; i < N; i++) {
    for (uint32_t j = 0; j < dim; j++) {
      elements[i * dim + j] = (T)((i + j) % 1000);
    }
  }
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < dim; j++) {
      centroids_init[i * dim + j] = elements[i * dim + j];
    }
  }
  centroids = centroids_init;

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (uint32_t m = 0; m < iterations; m++) {
    std::vector<uint32_t> counts(k, 0);
    std::vector<int64_t> sums(k * dim, 0);

#pragma omp parallel
    {
      std::vector<uint32_t> local_counts(k, 0);
      std::vector<int64_t> local_sums(k * dim, 0);
#pragma omp for
      for (uint64_t i = 0; i < N; i++) {
        uint32_t best = 0;
        uint64_t best_dist = (uint64_t)-1;
        for (uint32_t j = 0; j < k; j++) {
          uint64_t dist = 0;
          for (uint32_t d = 0; d < dim; d++) {
            T tmp = elements[i * dim + d] - centroids[j * dim + d];
            dist += (uint64_t)tmp * tmp;
          }
          if (dist < best_dist) {
            best = j;
            best_dist = dist;
          }
        }
        local_counts[best]++;
        for (uint32_t d = 0; d < dim; d++) {
          local_sums[best * dim + d] += elements[i * dim + d];
        }
      }
#pragma omp critical
      {
        for (int j = 0; j < k; j++) {
          counts[j] += local_counts[j];
          for (int d = 0; d < dim; d++) {
            sums[j * dim + d] += local_sums[j * dim + d];
          }
        }
      }
    }

    for (uint32_t j = 0; j < k; j++) {
      if (counts[j] > 0) {
        for (uint32_t d = 0; d < dim; d++) {
          centroids[j * dim + d] =
              (T)divRoundClosest((int)sums[j * dim + d], (int)counts[j]);
        }
      }
    }
    DoNotOptimize(centroids.data());
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = (end - start);

  std::cout << "cpu_baseline (ms): " << elapsed.count() << std::endl;

  std::cout << "Writing binary files to ./data/ ..." << std::endl;
  write_bin("data/AoS/rows.bin", elements.data(), N * dim * sizeof(T));
  write_columns(elements);
  write_bin("data/ref_c_init.bin", centroids_init.data(), k * dim * sizeof(T));
  write_bin("data/ref_c_final.bin", centroids.data(), k * dim * sizeof(T));

  return 0;
}
