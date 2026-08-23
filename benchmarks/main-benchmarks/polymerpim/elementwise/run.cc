#include <benchmark.h>
#include <omp.h>
#include <polymerpim.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "Param.h"

using namespace polymerpim;

inline DPUVector<T> compute(const DPUVector<T>& a, const DPUVector<T>& b) {
  return OPERATION(a, b);
}

void compare_cpu_dpu_vectors(const std::vector<T>& a, const std::vector<T>& b,
                             const T* dpu_result, uint32_t iterations) {
  const size_t n = a.size();
  std::vector<T> cpu_result(n);

  if (load_ref) {
    std::cout << "Loading expected results from " << ref_path << "..."
              << std::endl;
    bench_load_bin((std::string(ref_path) + "/ref_res.bin").c_str(),
                   cpu_result.data(), n * sizeof(T));
  } else {
    for (size_t i = 0; i < n; i++) {
      cpu_result[i] = OPERATION(a[i], b[i]);
    }
  }

  for (size_t i = 0; i < n; i++) {
    if (cpu_result[i] != dpu_result[i]) {
      std::cerr << "Mismatch at index " << i
                << ": CPU result = " << cpu_result[i]
                << ", DPU result = " << dpu_result[i] << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "All results match after " << iterations << " iterations."
            << std::endl;
}

int main() {
  try {
    const char* nr_dpus_env = std::getenv("NR_DPUS");
    int nr_dpus = nr_dpus_env ? std::stoi(nr_dpus_env) : 64;
    BenchStages stages;  // steady-loop stages (+ one-time setup)
    BenchStages
        warm_stages;  // cold warmup-loop stages (the cold-start premium)
    bench_stages_init(&stages);
    bench_stages_init(&warm_stages);
    bench_stage_begin(&stages, BENCH_STAGE_INIT);
    init(nr_dpus);
    bench_stage_end(&stages);
    {
      bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
      std::vector<T> a(N), b(N);
      bench_stage_end(&stages);
      if (load_ref) {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
        std::cout << "Loading reference data from " << ref_path << "..."
                  << std::endl;
        bench_load_bin((std::string(ref_path) + "/ref_a.bin").c_str(), a.data(),
                       N * sizeof(T));
        bench_load_bin((std::string(ref_path) + "/ref_b.bin").c_str(), b.data(),
                       N * sizeof(T));
        bench_stage_end(&stages);
      } else {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel
        {
          unsigned int seed_thread = seed + omp_get_thread_num();
#pragma omp for
          for (uint32_t i = 0; i < N; i++) {
            a[i] = rand_r(&seed_thread) % 10;
            b[i] = rand_r(&seed_thread) % 10;
          }
        }
        bench_stage_end(&stages);
      }

      std::vector<T> result;

      // PolymerPIM executes lazily: from_cpu/compute only enqueue work. Fence
      // at each stage boundary so write/kernel time is attributed to its own
      // stage instead of all landing in to_cpu()'s blocking read.
      auto run_round_trip = [&](BenchStages& stages) {
        bench_stage_begin(&stages, BENCH_STAGE_WRITE);
        DPUVector<T> da(a, "a");
        DPUVector<T> db(b, "b");
        sync();
        bench_stage_end(&stages);
        bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
        DPUVector<T> res = compute(da, db);
        sync();
        bench_stage_end(&stages);
        bench_stage_begin(&stages, BENCH_STAGE_READ);
        result = res.to_cpu();  // implicit runtime fence
        bench_stage_end(&stages);
      };

      BenchTimer warmup_timer;
      BenchStats warmup_stats;
      bench_stats_init(&warmup_stats);
      for (uint32_t i = 0; i < warmup_iterations; i++) {
        bench_start(&warmup_timer, 0);
        run_round_trip(warm_stages);
        bench_stop(&warmup_timer, 0);
        bench_stats_update(&warmup_stats, warmup_timer.time[0]);
      }
      if (warmup_iterations > 0) {
        bench_stats_print("polymerpim_warmup", &warmup_stats);
      }

      BenchStats stats;
      bench_stats_init(&stats);
      BenchTimer timer;
      for (uint32_t i = 0; i < iterations; i++) {
        bench_start(&timer, 0);
        run_round_trip(stages);
        bench_stop(&timer, 0);
        bench_stats_update(&stats, timer.time[0]);
      }
      bench_stats_print("polymerpim", &stats);
      bench_stages_report("polymerpim", &stages);
      bench_stages_report("polymerpim_cold", &warm_stages);

      if (check_correctness) {
        compare_cpu_dpu_vectors(a, b, result.data(), iterations);
      }
    }

    shutdown();

    return 0;
  } catch (const OutOfMemory& e) {
    std::cerr << "DPU OOM: Not enough memory for requested size." << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
