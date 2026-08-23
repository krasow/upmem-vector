#include <benchmark.h>
#include <polymerpim.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "Param.h"

using namespace polymerpim;

static vector_search_result_t cpu_best(const std::vector<T> &query) {
  vector_search_result_t result;
  vector_search_result_init(&result);
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
      vector_search_result_insert(&local,
                                  vector_search_pack_key(score, i, N, DIM));
    }
#pragma omp critical
    vector_search_result_merge(&result, &local);
  }
  return result;
}

int main() {
  try {
    const char *nr_dpus_env = std::getenv("NR_DPUS");
    const uint32_t nr_dpus = nr_dpus_env ? std::stoi(nr_dpus_env) : 64;
    if (N % nr_dpus != 0 || !vector_search_key_range_is_valid(N, DIM)) {
      std::cerr
          << "Invalid Vector search configuration: require N divisible by DPUs "
             "and (4*DIM+1)*N <= INT32_MAX"
          << std::endl;
      return 2;
    }

    BenchStages stages, warm_stages;
    bench_stages_init(&stages);
    bench_stages_init(&warm_stages);

    bench_stage_begin(&stages, BENCH_STAGE_INIT);
    init(nr_dpus);
    bench_stage_end(&stages);

    std::vector<DPUVector<T>> columns;
    std::vector<std::string> names;
    columns.reserve(DIM + 1);
    names.reserve(DIM + 1);

    /* Build and scatter each SoA column once; every DPU receives N/nr_dpus. */
    for (uint32_t d = 0; d < DIM; ++d) {
      std::vector<T> host_column;
      bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
      host_column.resize(N);
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel for
      for (uint64_t i = 0; i < N; ++i) {
        host_column[i] = vector_search_dataset_value(seed, i, d, DIM);
      }
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_WRITE);
      names.push_back("x" + std::to_string(d));
      columns.emplace_back(host_column, names.back());
      fence(columns.back());
      bench_stage_end(&stages);
    }

    std::vector<T> tie_breakers(N);
#pragma omp parallel for
    for (uint64_t i = 0; i < N; ++i) {
      tie_breakers[i] = (T)(N - 1 - i);
    }
    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    names.push_back("tie_breaker");
    columns.emplace_back(tie_breakers, names.back());
    fence(columns.back());
    bench_stage_end(&stages);
    tie_breakers.clear();
    tie_breakers.shrink_to_fit();

    std::vector<T> query(DIM);
    uint64_t query_id = 0;

    auto run_queries = [&](uint32_t count, BenchStages &query_stages) {
      std::vector<DpuFuture<T>> pending_bests;
      pending_bests.reserve(count);

      for (uint32_t q = 0; q < count; ++q) {
        bench_stage_begin(&query_stages, BENCH_STAGE_WRITE);
        for (uint32_t d = 0; d < DIM; ++d) {
          query[d] = vector_search_query_value(seed, query_id, d);
        }
        ++query_id;
        bench_stage_end(&query_stages);

        bench_stage_begin(&query_stages, BENCH_STAGE_KERNEL);
        auto score = columns[0] + query[0];
        for (uint32_t d = 1; d < DIM; ++d) {
          score = score + columns[d] + query[d];
        }
        auto packed = (score + (T)(2 * DIM)) * (T)N + columns[DIM];
        pending_bests.push_back(maximum(packed));
        bench_stage_end(&query_stages);
      }

      /* Expose the full batch to the queue before execution. Independent
       * query reductions can now be emitted as horizontal chains. */
      bench_stage_begin(&query_stages, BENCH_STAGE_KERNEL);
      sync();
      bench_stage_end(&query_stages);

      bench_stage_begin(&query_stages, BENCH_STAGE_READ);
      std::vector<typename DPUVector<T>::reduction_result_t> best_values;
      best_values.reserve(pending_bests.size());
      for (auto &best : pending_bests) {
        best_values.push_back(best.get());
      }
      bench_stage_end(&query_stages);

      vector_search_result_t global;
      vector_search_result_init(&global);
      bench_stage_begin(&query_stages, BENCH_STAGE_MERGE);
      if (!best_values.empty()) {
        vector_search_result_insert(&global, (T)best_values.back());
      }
      bench_stage_end(&query_stages);
      return global;
    };

    BenchStats warmup_stats;
    bench_stats_init(&warmup_stats);
    BenchTimer timer;
    vector_search_result_t result;
    vector_search_result_init(&result);
    if (warmup_iterations) {
      bench_start(&timer, 0);
      result = run_queries(warmup_iterations, warm_stages);
      bench_stop(&timer, 0);
      bench_stats_update(&warmup_stats,
                         timer.time[0] / std::max(warmup_iterations, 1u));
    }
    if (warmup_iterations) {
      bench_stats_print("polymerpim_warmup", &warmup_stats);
    }

    BenchStats stats;
    bench_stats_init(&stats);
    if (iterations) {
      bench_start(&timer, 0);
      result = run_queries(iterations, stages);
      bench_stop(&timer, 0);
      bench_stats_update(&stats, timer.time[0] / std::max(iterations, 1u));
    }

    bench_stats_print("polymerpim", &stats);
    bench_stages_report("polymerpim", &stages);
    bench_stages_report("polymerpim_cold", &warm_stages);

    if (check_correctness && iterations) {
      vector_search_result_t expected = cpu_best(query);
      bool ok = result.key == expected.key;
      if (ok) {
        std::cout << "the result is correct" << std::endl;
      } else {
        std::cout << "Mismatch: got";
        std::cout << " " << result.key;
        std::cout << ", expected";
        std::cout << " " << expected.key;
        std::cout << std::endl;
      }
    }

    /* Release MRAM-backed vectors while the runtime/allocator is still live. */
    columns.clear();
    shutdown();
    return 0;
  } catch (const OutOfMemory &) {
    std::cerr << "DPU OOM: Not enough memory for requested size." << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
