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

void compare_results(const DPUVector<T>::reduction_result_t& dpu_result) {
  DPUVector<T>::reduction_result_t cpu_result;

  if (load_ref) {
    std::cout << "Loading expected results from " << ref_path << "..."
              << std::endl;
    bench_load_bin((std::string(ref_path) + "/ref_res.bin").c_str(),
                   &cpu_result, sizeof(cpu_result));
  } else {
    std::cerr << "Reference data must be loaded for correctness check."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (cpu_result != dpu_result) {
    std::cerr << "Mismatch: CPU result = " << cpu_result
              << ", DPU result = " << dpu_result << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::cout << "the result is correct" << std::endl;
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
            a[i] = rand_r(&seed_thread) % 10;
          }
        }
        bench_stage_end(&stages);
      }

      DPUVector<T>::reduction_result_t result{};

      // The runtime is async: from_cpu/sum only enqueue work. Fence at each
      // stage boundary so the timer captures that stage's real cost instead of
      // letting it all land in the read's blocking .get().
      auto run_round_trip = [&](BenchStages& stages) {
        bench_stage_begin(&stages, BENCH_STAGE_WRITE);
        DPUVector<T> da(a, "a");
        sync();
        bench_stage_end(&stages);
        bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
        auto pending = sum(da);
        sync();
        bench_stage_end(&stages);
        bench_stage_begin(&stages, BENCH_STAGE_READ);
        result = pending.get();
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
        compare_results(result);
      }
    }

    shutdown();

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
