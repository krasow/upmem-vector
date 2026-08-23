#include <benchmark.h>
#include <omp.h>
#include <polymerpim.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "Param.h"

using namespace polymerpim;

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
      std::vector<T> query;
      bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
      query.resize(DIM);
      bench_stage_end(&stages);
      if (load_ref) {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
        bench_load_bin((std::string(ref_path) + "/ref_query.bin").c_str(),
                       query.data(), DIM * sizeof(T));
        bench_stage_end(&stages);
      } else {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
        for (uint32_t d = 0; d < DIM; d++) {
          query[d] = (T)(d * 17 % 128);
        }
        bench_stage_end(&stages);
      }

      // Load/transfer one column at a time to avoid DIM*N host allocation
      std::vector<std::string> dpu_names;
      std::vector<DPUVector<T>> da;
      da.reserve(DIM);
      dpu_names.reserve(DIM);
      for (uint32_t d = 0; d < DIM; d++) {
        std::vector<T> col;
        bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
        col.resize(N);
        bench_stage_end(&stages);
        if (load_ref) {
          bench_stage_begin(&stages, BENCH_STAGE_LOAD);
          bench_load_bin(
              (std::string(ref_path) + "/SoA/col_" + std::to_string(d) + ".bin")
                  .c_str(),
              col.data(), N * sizeof(T));
          bench_stage_end(&stages);
        } else {
          bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel for
          for (uint64_t i = 0; i < N; i++) {
            col[i] = (T)((i * (DIM + 1) + d) % 256);
          }
          bench_stage_end(&stages);
        }
        bench_stage_begin(&stages, BENCH_STAGE_WRITE);
        dpu_names.push_back("x" + std::to_string(d));
        da.emplace_back(col, dpu_names.back());
        fence(da.back());  // flush before col goes out of scope
        bench_stage_end(&stages);
      }

      RED_T result;

      auto run_knn = [&](BenchStages& stages) {
        bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
        auto dist = sqr(da[0] - query[0]);
        for (uint32_t d = 1; d < DIM; d++) {
          dist = dist + sqr(da[d] - query[d]);
        }
        auto pending = minimum(dist);
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
        run_knn(warm_stages);
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
        run_knn(stages);
        bench_stop(&timer, 0);
        bench_stats_update(&stats, timer.time[0]);
      }
      bench_stats_print("polymerpim", &stats);
      bench_stages_report("polymerpim", &stages);
      bench_stages_report("polymerpim_cold", &warm_stages);

      if (check_correctness) {
        RED_T expected;
        bench_load_bin((std::string(ref_path) + "/ref_res.bin").c_str(),
                       &expected, sizeof(RED_T));
        if (result != expected) {
          std::cout << "Mismatch: got " << result << ", expected " << expected
                    << std::endl;
        } else {
          std::cout << "the result is correct" << std::endl;
        }
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
