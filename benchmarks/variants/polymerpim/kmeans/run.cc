#include <benchmark.h>
#include <vectordpu.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "Param.h"

static int divRoundClosest(const int n, const int d) {
  return ((n < 0) ^ (d < 0)) ? ((n - d / 2) / d) : ((n + d / 2) / d);
}

int main() {
  try {
#if !JIT
    std::cerr
        << "Exception: polymerpim_kmeans requires JIT mode in the current "
           "implementation"
        << std::endl;
    return 1;
#else
    const char *nr_dpus_env = std::getenv("NR_DPUS");
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
      std::vector<T> centroids_init;

      bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
      centroids_init.resize(K * DIM);
      bench_stage_end(&stages);
      if (load_ref) {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
        bench_load_bin((std::string(ref_path) + "/ref_c_init.bin").c_str(),
                       centroids_init.data(), K * DIM * sizeof(T));
        bench_stage_end(&stages);
      } else {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
        for (uint32_t j = 0; j < K; j++)
          for (uint32_t d = 0; d < DIM; d++)
            centroids_init[j * DIM + d] = (T)((j + d) % 1000);
        bench_stage_end(&stages);
      }

      std::vector<std::string> dpu_names;
      std::vector<dpu_vector<T> > da;
      da.reserve(DIM);
      dpu_names.reserve(DIM);
      for (uint32_t d = 0; d < DIM; d++) {
        bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
        std::vector<T> col(N);
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
          for (uint64_t i = 0; i < N; i++) col[i] = (T)((i + d) % 1000);
          bench_stage_end(&stages);
        }
        bench_stage_begin(&stages, BENCH_STAGE_WRITE);
        dpu_names.push_back("x" + std::to_string(d));
        da.push_back(dpu_vector<T>::from_cpu(col, dpu_names.back(),
                                             VECTORDPU_SOURCE_LOCATION));
        da.back().add_fence();
        bench_stage_end(&stages);
      }

      std::vector<T> centroids = centroids_init;

      auto run_kmeans = [&](BenchStages &stages) {
        bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
        // Build the nearest-centroid expression once, then append local
        // accumulator side effects so assignment and update share one scan.
        std::vector<uint32_t> centroid_scalars(K * DIM);
        for (uint32_t j = 0; j < K; j++) {
          for (uint32_t d = 0; d < DIM; d++) {
            centroid_scalars[j * DIM + d] = (uint32_t)centroids[j * DIM + d];
          }
        }

        using ReductionResult = typename dpu_vector<T>::reduction_result_t;
        dpu_local_vector<T> local_stats(K * (DIM + 1));
        std::vector<dpu_vector<T> > dist_operands(da.begin() + 1, da.end());

        dpu_jit_foreach<T>(
            da[0], dist_operands, centroid_scalars,
            [&](const std::vector<dpu_expr<T> > &x,
                dpu_pipeline_context<T> &ctx) {
              auto distance_to_centroid = [&](uint32_t j) {
                auto dist = (x[0] - dpu_expr<T>::scalar_var(j * DIM + 0)).sqr();
                for (uint32_t d = 1; d < DIM; d++) {
                  dist = dist +
                         (x[d] - dpu_expr<T>::scalar_var(j * DIM + d)).sqr();
                }
                return dist;
              };

              // One variadic argmin over the K candidate distances replaces the
              // compare+dual-select chain; .label is the winning centroid.
              std::vector<dpu_expr<T> > dists;
              dists.reserve(K);
              for (uint32_t j = 0; j < K; j++)
                dists.push_back(distance_to_centroid(j));
              auto best_label = argmin(dists).label;

              auto base = best_label * (T)(DIM + 1);
              ctx.local_sum(local_stats, base, (T)1);

              for (uint32_t d = 0; d < DIM; d++) {
                ctx.local_sum(local_stats, base + (T)(d + 1), x[d]);
              }
            });

        dpu_fence();  // finish fused assignment/update before closing the stage
        bench_stage_end(&stages);

        bench_stage_begin(&stages, BENCH_STAGE_READ);
        std::vector<ReductionResult> stats = local_stats.to_cpu();
        std::vector<ReductionResult> counts(K);
        std::vector<ReductionResult> sums(K * DIM);

        for (uint32_t j = 0; j < K; j++) {
          counts[j] = stats[j * (DIM + 1)];
          for (uint32_t d = 0; d < DIM; d++) {
            sums[j * DIM + d] = stats[j * (DIM + 1) + d + 1];
          }
        }

        bench_stage_end(&stages);

        bench_stage_begin(&stages, BENCH_STAGE_MERGE);
        for (uint32_t j = 0; j < K; j++) {
          ReductionResult count_j = counts[j];
          if (count_j <= 0) continue;
          for (uint32_t d = 0; d < DIM; d++) {
            ReductionResult s = sums[j * DIM + d];
            centroids[j * DIM + d] = (T)divRoundClosest((int)s, (int)count_j);
          }
        }
        bench_stage_end(&stages);
      };

      Timer warmup_timer;
      BenchStats warmup_stats;
      bench_stats_init(&warmup_stats);
      for (uint32_t i = 0; i < warmup_iterations; i++) {
        centroids = centroids_init;
        bench_start(&warmup_timer, 0);
        run_kmeans(warm_stages);
        bench_stop(&warmup_timer, 0);
        bench_stats_update(&warmup_stats, warmup_timer.time[0]);
      }
      if (warmup_iterations > 0)
        bench_stats_print("polymerpim_warmup", &warmup_stats);

      centroids = centroids_init;
      BenchStats stats;
      bench_stats_init(&stats);
      Timer timer;
      for (uint32_t i = 0; i < iterations; i++) {
        bench_start(&timer, 0);
        run_kmeans(stages);
        bench_stop(&timer, 0);
        bench_stats_update(&stats, timer.time[0]);
      }
      bench_stats_print("polymerpim", &stats);
      bench_stages_report("polymerpim", &stages);
      bench_stages_report("polymerpim_cold", &warm_stages);

      if (check_correctness) {
        std::vector<T> expected(K * DIM);
        bench_load_bin((std::string(ref_path) + "/ref_c_final.bin").c_str(),
                       expected.data(), K * DIM * sizeof(T));
        bool ok = true;
        for (uint32_t i = 0; i < K * DIM && ok; i++) {
          if (centroids[i] != expected[i]) {
            std::cerr << "Mismatch at centroid[" << i / DIM << "][" << i % DIM
                      << "]: got " << centroids[i] << ", expected "
                      << expected[i] << std::endl;
            ok = false;
          }
        }
        if (ok) std::cout << "the result is correct" << std::endl;
      }
    }

    DpuRuntime::get().shutdown();
    return 0;
#endif
  } catch (const DpuOOMException &e) {
    std::cerr << "DPU OOM: Not enough memory for requested size." << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
