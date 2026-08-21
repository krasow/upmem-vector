#include <benchmark.h>
#include <runtime.h>
#include <vectordpu.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "Param.h"

#ifndef RED_T
typedef int64_t RED_T;
#endif

#ifndef LINREG_PIPELINE_TERM_CHUNK
#define LINREG_PIPELINE_TERM_CHUNK 10
#endif

static dpu_vector<T> compute_error(dpu_vector<T>& dy,
                                   std::vector<dpu_vector<T>>& dx_cols,
                                   std::vector<T>& dw_scalar) {
#if PIPELINE && !JIT
  dpu_vector<T> error = -dy;
  dpu_fence();

  for (uint32_t j = 0; j < DIM;) {
    dpu_vector<T> term = dx_cols[j] * dw_scalar[j];
    j++;
    for (uint32_t k = 1; k < LINREG_PIPELINE_TERM_CHUNK && j < DIM; k++, j++) {
      term = term + dx_cols[j] * dw_scalar[j];
    }
    dpu_fence();
    error += term;
    if (j < DIM) dpu_fence();
  }
  dpu_fence();
  return error;
#else
  dpu_vector<T> error = -dy;
  for (uint32_t j = 0; j < DIM; j++) {
    error = error + dx_cols[j] * dw_scalar[j];
  }
  return error;
#endif
}

static void run_iter(dpu_vector<T>& dy, std::vector<dpu_vector<T>>& dx_cols,
                     std::vector<T>& dw_scalar, std::vector<int64_t>& grads,
                     BenchStages& stages) {
  bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
  // The unshifted error has no use after this expression.  Keeping it in a
  // named handle until the fence forces the conservative fusion policy to
  // materialize an otherwise-inlined full-size intermediate.
  dpu_vector<T> error_shifted = compute_error(dy, dx_cols, dw_scalar) >>
                                (T)(scaling_shift - scaling_shift / 2);
#if PIPELINE && !JIT
  dpu_fence();
#endif
  dpu_future<T> lazy_grads[DIM];
  for (uint32_t j = 0; j < DIM; j++)
    lazy_grads[j] = sum((dx_cols[j] >> (T)(scaling_shift / 2)) * error_shifted);
  dpu_fence();  // finish the gradient kernels before closing the kernel stage
  bench_stage_end(&stages);

  bench_stage_begin(&stages, BENCH_STAGE_READ);
  for (uint32_t j = 0; j < DIM; j++) grads[j] = lazy_grads[j].get();

  dpu_fence();
  bench_stage_end(&stages);
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
    DpuRuntime::get().init(nr_dpus);
    bench_stage_end(&stages);
    {
      std::vector<std::vector<T>> host_x_cols;
      std::vector<T> host_y;

      bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
      host_x_cols.resize(DIM);
      for (uint32_t j = 0; j < DIM; j++) {
        host_x_cols[j].resize(N);
      }
      host_y.resize(N);
      bench_stage_end(&stages);
      if (load_ref) {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
        for (uint32_t j = 0; j < DIM; j++) {
          char path[512];
          snprintf(path, sizeof(path), "%s/SoA/x_col_%u.bin", ref_path, j);
          bench_load_bin(path, host_x_cols[j].data(), N * sizeof(T));
        }
        char path[512];
        snprintf(path, sizeof(path), "%s/SoA/y.bin", ref_path);
        bench_load_bin(path, host_y.data(), N * sizeof(T));
        bench_stage_end(&stages);
      } else {
        bench_stage_begin(&stages, BENCH_STAGE_LOAD);
        for (uint32_t i = 0; i < N; i++) {
          for (uint32_t j = 0; j < DIM; j++)
            host_x_cols[j][i] = (i * (DIM + 1) + j) % 256;
          host_y[i] = (i * (DIM + 1) + DIM) % 256;
        }
        bench_stage_end(&stages);
      }

      bench_stage_begin(&stages, BENCH_STAGE_WRITE);
      dpu_vector<T> dy =
          dpu_vector<T>::from_cpu(host_y, "y", VECTORDPU_SOURCE_LOCATION);
      std::vector<std::string> dx_names;
      std::vector<dpu_vector<T>> dx_cols;
      dx_names.reserve(DIM);
      for (uint32_t j = 0; j < DIM; j++)
        dx_names.push_back("x" + std::to_string(j));
      dx_cols.reserve(DIM);
      for (uint32_t j = 0; j < DIM; j++)
        dx_cols.push_back(dpu_vector<T>::from_cpu(host_x_cols[j], dx_names[j],
                                                  VECTORDPU_SOURCE_LOCATION));
      dpu_fence();  // finish all column uploads before closing the write stage
      bench_stage_end(&stages);

      std::vector<T> dw_scalar(DIM, 0);
      std::vector<int64_t> grads(DIM);

      BenchTimer warmup_timer;
      BenchStats warmup_stats;
      bench_stats_init(&warmup_stats);
      for (uint32_t i = 0; i < warmup_iterations; i++) {
        bench_start(&warmup_timer, 0);
        run_iter(dy, dx_cols, dw_scalar, grads, warm_stages);
        bench_stop(&warmup_timer, 0);
        bench_stats_update(&warmup_stats, warmup_timer.time[0]);
      }
      if (warmup_iterations > 0)
        bench_stats_print("polymerpim_warmup", &warmup_stats);

      BenchStats stats;
      bench_stats_init(&stats);
      BenchTimer timer;
      for (uint32_t i = 0; i < iterations; i++) {
        bench_start(&timer, 0);
        run_iter(dy, dx_cols, dw_scalar, grads, stages);
        bench_stop(&timer, 0);
        bench_stats_update(&stats, timer.time[0]);
      }

      bench_stats_print("polymerpim", &stats);
      bench_stages_report("polymerpim", &stages);
      bench_stages_report("polymerpim_cold", &warm_stages);

      if (check_correctness && load_ref) {
        std::vector<RED_T> expected_grads(DIM);
        char path[512];
        snprintf(path, sizeof(path), "%s/ref_grads.bin", ref_path);
        bench_load_bin(path, expected_grads.data(), DIM * sizeof(RED_T));
        bool ok = true;
        for (uint32_t i = 0; i < DIM; i++) {
          if (grads[i] != (int64_t)expected_grads[i]) {
            std::cerr << "Mismatch at gradient " << i << ": got " << grads[i]
                      << ", expected " << expected_grads[i] << std::endl;
            ok = false;
          }
        }
        if (ok)
          std::cout << "All results match after " << iterations
                    << " iterations." << std::endl;
      }
    }
    DpuRuntime::get().shutdown();
    return 0;
  } catch (const DpuOOMException& e) {
    std::cerr << "DPU OOM: Not enough memory for requested size." << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
