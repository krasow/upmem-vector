#include <benchmark.h>
#include <runtime.h>
#include <stats.h>
#include <vectordpu.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "Param.h"

static T target_pixel(uint64_t index) {
  return (T)(32 + ((index * 17 + seed) % 192));
}

int main() {
  try {
    if (warmup_iterations != 0) {
      std::cerr << "adaptive_image requires warmup=0" << std::endl;
      return 2;
    }

    const char* nr_dpus_env = std::getenv("NR_DPUS");
    const uint32_t nr_dpus = nr_dpus_env ? std::stoi(nr_dpus_env) : 64;
    BenchStages stages;
    bench_stages_init(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_INIT);
    DpuRuntime::get().init(nr_dpus);
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
    std::vector<T> host_image(N, 0);
    std::vector<T> host_target(N);
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel for
    for (uint64_t i = 0; i < N; ++i) host_target[i] = target_pixel(i);
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    dpu_vector<T> image = dpu_vector<T>::from_cpu(host_image, "image");
    dpu_vector<T> target = dpu_vector<T>::from_cpu(host_target, "target");
    dpu_fence();
    bench_stage_end(&stages);

    BenchStats stats;
    BenchTimer timer;
    bench_stats_init(&stats);
    uint32_t fine_iterations = 0;
    T last_error = 0;
#if JIT_PIPELINE_FALLBACK
    StatsSnapshot runtime_before = RuntimeStats::get().snapshot();
#endif

    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      bench_start(&timer, 0);

      bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
      auto pending_error = max(abs(target - image));
      dpu_fence();
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_READ);
      last_error = pending_error.get();
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
      if (last_error < tolerance) {
        image = image + ((target - image + (T)3) >> (T)2);
        ++fine_iterations;
      } else {
        image = image + ((target - image) >> (T)1);
      }
      dpu_fence();
      bench_stage_end(&stages);

      bench_stop(&timer, 0);
      bench_stats_update(&stats, timer.time[0]);
    }

    bench_stage_begin(&stages, BENCH_STAGE_READ);
    std::vector<T> result = image.to_cpu();
    bench_stage_end(&stages);

#if JIT_PIPELINE_FALLBACK
    StatsSnapshot fallback_stats =
        RuntimeStats::get().snapshot() - runtime_before;
    size_t jit_pipeline_fallbacks = fallback_stats.jit_pipeline_fallbacks;
    size_t jit_eager_fallbacks = fallback_stats.jit_eager_fallbacks;
#else
    size_t jit_pipeline_fallbacks = 0;
    size_t jit_eager_fallbacks = 0;
#endif
    bench_stats_print("polymerpim", &stats);
    bench_stages_report("polymerpim", &stages);
    std::cout << "adaptive_image: coarse_iterations="
              << iterations - fine_iterations
              << " fine_iterations=" << fine_iterations
              << " final_error=" << last_error
              << " jit_pipeline_fallbacks=" << jit_pipeline_fallbacks
              << " jit_eager_fallbacks=" << jit_eager_fallbacks << std::endl;

    if (check_correctness) {
      std::vector<T> expected(N, 0);
      for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        T error = 0;
#pragma omp parallel for reduction(max : error)
        for (uint64_t i = 0; i < N; ++i) {
          T delta = host_target[i] - expected[i];
          if (delta < 0) delta = -delta;
          if (delta > error) error = delta;
        }
#pragma omp parallel for
        for (uint64_t i = 0; i < N; ++i) {
          T delta = host_target[i] - expected[i];
          expected[i] += error < tolerance ? ((delta + 3) >> 2) : (delta >> 1);
        }
      }
      for (uint64_t i = 0; i < N; ++i) {
        if (result[i] != expected[i]) {
          std::cerr << "Mismatch at index " << i << ": got " << result[i]
                    << ", expected " << expected[i] << std::endl;
          return 1;
        }
      }
      std::cout << "All results match after " << iterations
                << " dynamic iterations." << std::endl;
    }

    DpuRuntime::get().shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Exception: " << error.what() << std::endl;
    return 1;
  }
}
