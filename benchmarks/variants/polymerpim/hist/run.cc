#include <benchmark.h>
#include <omp.h>
#include <vectordpu.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Param.h"

void compare_results(const std::vector<RED_T>& dpu_hist) {
  std::vector<uint32_t> cpu_hist(BINS);

  if (load_ref) {
    std::cout << "Loading expected results from " << ref_path << "..."
              << std::endl;
    bench_load_bin((std::string(ref_path) + "/ref_res.bin").c_str(),
                   cpu_hist.data(), BINS * sizeof(uint32_t));
  } else {
    std::cerr << "Reference data must be loaded for correctness check."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bool correct = true;
  for (int i = 0; i < BINS; i++) {
    if (static_cast<RED_T>(cpu_hist[i]) != dpu_hist[i]) {
      std::cerr << "Mismatch at bin " << i << ": CPU result = " << cpu_hist[i]
                << ", DPU result = " << dpu_hist[i] << std::endl;
      correct = false;
    }
  }

  if (!correct) std::exit(EXIT_FAILURE);
  std::cout << "the result is correct" << std::endl;
}

int main() {
  try {
#if !JIT
    std::cerr << "Exception: polymerpim_hist requires JIT mode in the current "
                 "implementation"
              << std::endl;
    return 1;
#else
    const char* nr_dpus_env = std::getenv("NR_DPUS");
    int nr_dpus = nr_dpus_env ? std::stoi(nr_dpus_env) : 64;
    BenchStages stages;  // steady-loop stages (+ one-time setup)
    BenchStages
        warm_stages;  // cold warmup-loop stages (the cold-start premium)
    bench_stages_init(&stages);
    bench_stages_init(&warm_stages);
    bench_stage_begin(&stages, BENCH_STAGE_INIT);
    DpuRuntime::get().init(nr_dpus);
    bench_stage_end(&stages);
    {
      bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
      std::vector<T> a(N);
      bench_stage_end(&stages);
      if (load_ref) {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
        std::cout << "Loading reference data from " << ref_path << "..."
                  << std::endl;
        bench_load_bin((std::string(ref_path) + "/ref_t1.bin").c_str(),
                       a.data(), N * sizeof(T));
        bench_stage_end(&stages);
      } else {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel
        {
          unsigned int seed_thread = seed + omp_get_thread_num();
#pragma omp for
          for (uint32_t i = 0; i < N; i++) {
            a[i] = rand_r(&seed_thread) % 4096;
          }
        }
        bench_stage_end(&stages);
      }

      std::vector<RED_T> result_hist(BINS);

      // Lazy runtime: from_cpu and the jit_foreach only enqueue work. Fence at
      // each stage boundary so write/kernel time stays in its own stage rather
      // than landing in to_cpu()'s blocking read.
      auto run_round_trip = [&](std::vector<RED_T>& res_hist,
                                BenchStages& stages) {
        bench_stage_begin(&stages, BENCH_STAGE_WRITE);
        dpu_vector<T> da =
            dpu_vector<T>::from_cpu(a, "a", VECTORDPU_SOURCE_LOCATION);
        dpu_fence();
        bench_stage_end(&stages);
        bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
        dpu_local_vector<T> local_hist(BINS);
        auto buckets = ((da * (T)BINS) >> (T)DEPTH);
        dpu_jit_foreach<T>(da.size(),
                           [&](auto i) { local_hist[buckets[i]].apply((T)1); });
        dpu_fence();
        bench_stage_end(&stages);
        bench_stage_begin(&stages, BENCH_STAGE_READ);
        res_hist = local_hist.to_cpu();  // implicit runtime fence
        bench_stage_end(&stages);
      };

      std::cout << "Starting warmup..." << std::endl;
      Timer warmup_timer;
      BenchStats warmup_stats;
      bench_stats_init(&warmup_stats);
      for (uint32_t i = 0; i < warmup_iterations; i++) {
        bench_start(&warmup_timer, 0);
        run_round_trip(result_hist, warm_stages);
        bench_stop(&warmup_timer, 0);
        bench_stats_update(&warmup_stats, warmup_timer.time[0]);
      }
      if (warmup_iterations > 0)
        bench_stats_print("polymerpim_warmup", &warmup_stats);

      std::cout << "Starting benchmark iterations=" << iterations << "..."
                << std::endl;
      BenchStats stats;
      bench_stats_init(&stats);
      Timer timer;
      for (uint32_t i = 0; i < iterations; i++) {
        bench_start(&timer, 0);
        run_round_trip(result_hist, stages);
        bench_stop(&timer, 0);
        bench_stats_update(&stats, timer.time[0]);
      }
      bench_stats_print("polymerpim", &stats);
      bench_stages_report("polymerpim", &stages);
      bench_stages_report("polymerpim_cold", &warm_stages);

      if (check_correctness) {
        compare_results(result_hist);
      }
    }

    DpuRuntime::get().shutdown();

    return 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
