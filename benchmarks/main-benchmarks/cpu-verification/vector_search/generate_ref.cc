#include <benchmark.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Param.h"

#define DoNotOptimize(x) asm volatile("" : : "r,m"(x) : "memory")

static void write_bin(const std::string& filename, const void* data,
                      size_t size) {
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

/*
 * The dataset is materialized in every layout its consumers transfer directly,
 * so each variant's load stage is a plain sequential read rather than a
 * host-side repack that would land in the same bar.
 *   SoA/col_<d>.bin  - one column per dimension (polymerpim, julia)
 *   AoS/rows.bin     - row major, stride DIM (baseline)
 *   AoS/records.bin  - row major, stride DIM + 1, trailing global row index
 *                      (simplepim)
 */
static const uint64_t CHUNK_ROWS = 1u << 20;

static void write_columns() {
  std::vector<T> col(std::min<uint64_t>(CHUNK_ROWS, N));
  for (uint32_t d = 0; d < DIM; d++) {
    const std::string path = "data/SoA/col_" + std::to_string(d) + ".bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      std::cerr << "Failed to open " << path << " for writing" << std::endl;
      std::exit(1);
    }
    for (uint64_t base = 0; base < N; base += CHUNK_ROWS) {
      const uint64_t count = std::min<uint64_t>(CHUNK_ROWS, N - base);
#pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < count; i++) {
        col[i] = vector_search_dataset_value(seed, base + i, d, DIM);
      }
      out.write(reinterpret_cast<const char*>(col.data()), count * sizeof(T));
      if (!out.good()) {
        std::cerr << "Short write to " << path << std::endl;
        std::exit(1);
      }
    }
  }
}

// stride == DIM leaves the rows unpadded; stride == DIM + 1 appends the global
// row index that SimplePIM's scatter layout carries alongside each record.
static void write_rows(const std::string& path, uint32_t stride) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "Failed to open " << path << " for writing" << std::endl;
    std::exit(1);
  }
  std::vector<T> rows(std::min<uint64_t>(CHUNK_ROWS, N) * stride);
  for (uint64_t base = 0; base < N; base += CHUNK_ROWS) {
    const uint64_t count = std::min<uint64_t>(CHUNK_ROWS, N - base);
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < count; i++) {
      for (uint32_t d = 0; d < DIM; d++) {
        rows[i * stride + d] =
            vector_search_dataset_value(seed, base + i, d, DIM);
      }
      if (stride > DIM) {
        rows[i * stride + DIM] = (T)(base + i);
      }
    }
    out.write(reinterpret_cast<const char*>(rows.data()),
              count * stride * sizeof(T));
    if (!out.good()) {
      std::cerr << "Short write to " << path << std::endl;
      std::exit(1);
    }
  }
}

int main() {
  if (!vector_search_key_range_is_valid(N, DIM)) {
    std::cerr << "Invalid Vector search configuration" << std::endl;
    return 2;
  }

  system("mkdir -p data/AoS data/SoA");
  std::cout << "Generating vector_search reference data for N=" << N
            << " DIM=" << DIM << "..." << std::endl;

  write_columns();
  write_rows("data/AoS/rows.bin", DIM);
  write_rows("data/AoS/records.bin", DIM + 1);

  if (bench_ref_data_only()) {
    std::cout << "Reference input generation complete." << std::endl;
    return 0;
  }

  std::vector<T> query(DIM);
  vector_search_result_t answer;
  vector_search_result_init(&answer);

  const uint32_t rounds = std::max(iterations, 1u);
  auto begin = std::chrono::high_resolution_clock::now();
  for (uint32_t it = 0; it < rounds; ++it) {
    const uint64_t query_id = (uint64_t)warmup_iterations + it;
    for (uint32_t d = 0; d < DIM; ++d) {
      query[d] = vector_search_query_value(seed, query_id, d);
    }

    vector_search_result_init(&answer);
#pragma omp parallel
    {
      vector_search_result_t local;
      vector_search_result_init(&local);
#pragma omp for nowait
      for (uint64_t i = 0; i < N; ++i) {
        int32_t score = 0;
        for (uint32_t d = 0; d < DIM; ++d) {
          score += vector_search_dataset_value(seed, i, d, DIM) * query[d];
        }
        vector_search_result_insert(&local, score, (uint32_t)i);
      }
#pragma omp critical
      vector_search_result_merge(&answer, &local);
    }
    DoNotOptimize(answer.score);
  }
  auto end = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(end - begin).count() / rounds;
  std::cout << "cpu_baseline (ms): " << ms << '\n';

  std::cout << "best additive match: (id=" << answer.index
            << ", score=" << answer.score << "/" << DIM << ")" << std::endl;
  return 0;
}
