#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "Param.h"

#define DoNotOptimize(x) asm volatile("" : : "r,m"(x) : "memory")

int main() {
  if (!vector_search_key_range_is_valid(N, DIM)) {
    std::cerr << "Invalid Vector search configuration" << std::endl;
    return 2;
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
          score += vector_search_dataset_value(seed, i, d, DIM) + query[d];
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
