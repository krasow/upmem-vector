#include <benchmark.h>
#include <runtime.h>
#include <stats.h>
#include <vectordpu.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "Param.h"

static T target_pixel(uint64_t index, uint32_t channel) {
  return (T)(32 + ((index * 17 + channel * 29 + seed) % 192));
}

static T channel_tolerance(uint32_t channel) {
  return tolerance << (2 * channel);
}

int main() {
  try {
    if (warmup_iterations != 0) {
      std::cerr << "adaptive_image requires warmup=0" << std::endl;
      return 2;
    }
    if (channels == 0 || channels > 4 || check_interval == 0) {
      std::cerr << "adaptive_image requires 1-4 channels and check_interval>0"
                << std::endl;
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
    std::vector<T> host_buffer(N, 0);
    bench_stage_end(&stages);

    std::vector<dpu_vector<T>> images;
    std::vector<dpu_vector<T>> targets;
    images.reserve(channels);
    targets.reserve(channels);

    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    for (uint32_t channel = 0; channel < channels; ++channel)
      images.push_back(dpu_vector<T>::from_cpu(host_buffer, "image"));
    dpu_fence();
    bench_stage_end(&stages);

    for (uint32_t channel = 0; channel < channels; ++channel) {
      bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel for
      for (uint64_t i = 0; i < N; ++i)
        host_buffer[i] = target_pixel(i, channel);
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_WRITE);
      targets.push_back(dpu_vector<T>::from_cpu(host_buffer, "target"));
      dpu_fence();
      bench_stage_end(&stages);
    }

    BenchStats stats;
    BenchTimer timer;
    bench_stats_init(&stats);
    uint32_t fine_updates = 0;
    std::vector<T> last_errors(channels, 0);
    StatsSnapshot runtime_before = RuntimeStats::get().snapshot();

    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      bench_start(&timer, 0);

      if (iteration % check_interval == 0) {
        bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
        dpu_future_vector<T> pending_errors;
        pending_errors.reserve(channels);
        for (uint32_t channel = 0; channel < channels; ++channel)
          pending_errors.push_back(
              max(abs(targets[channel] - images[channel])));
        dpu_fence();
        bench_stage_end(&stages);

        bench_stage_begin(&stages, BENCH_STAGE_READ);
        last_errors = pending_errors.get();
        bench_stage_end(&stages);
      }

      bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
      for (uint32_t channel = 0; channel < channels; ++channel) {
        if (last_errors[channel] < channel_tolerance(channel)) {
          images[channel] =
              images[channel] +
              ((targets[channel] - images[channel] + (T)3) >> (T)2);
          ++fine_updates;
        } else {
          images[channel] =
              images[channel] + ((targets[channel] - images[channel]) >> (T)1);
        }
      }
      dpu_fence();
      bench_stage_end(&stages);

      bench_stop(&timer, 0);
      bench_stats_update(&stats, timer.time[0]);
    }

    std::vector<T> expected;
    std::vector<T> target;
    if (check_correctness) {
      expected.resize(N);
      target.resize(N);
    }
    for (uint32_t channel = 0; channel < channels; ++channel) {
      bench_stage_begin(&stages, BENCH_STAGE_READ);
      std::vector<T> result = images[channel].to_cpu();
      bench_stage_end(&stages);

      if (check_correctness) {
        std::fill(expected.begin(), expected.end(), 0);
#pragma omp parallel for
        for (uint64_t i = 0; i < N; ++i) target[i] = target_pixel(i, channel);
        T error = 0;
        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
          if (iteration % check_interval == 0) {
            error = 0;
#pragma omp parallel for reduction(max : error)
            for (uint64_t i = 0; i < N; ++i) {
              T delta = target[i] - expected[i];
              if (delta < 0) delta = -delta;
              if (delta > error) error = delta;
            }
          }
#pragma omp parallel for
          for (uint64_t i = 0; i < N; ++i) {
            T delta = target[i] - expected[i];
            expected[i] += error < channel_tolerance(channel)
                               ? ((delta + 3) >> 2)
                               : (delta >> 1);
          }
        }
        for (uint64_t i = 0; i < N; ++i) {
          if (result[i] != expected[i]) {
            std::cerr << "Mismatch in channel " << channel << " at index " << i
                      << ": got " << result[i] << ", expected " << expected[i]
                      << std::endl;
            return 1;
          }
        }
      }
    }

    StatsSnapshot runtime_stats =
        RuntimeStats::get().snapshot() - runtime_before;
#if JIT_PIPELINE_FALLBACK
    size_t jit_pipeline_fallbacks = runtime_stats.jit_pipeline_fallbacks;
    size_t jit_eager_fallbacks = runtime_stats.jit_eager_fallbacks;
#else
    size_t jit_pipeline_fallbacks = 0;
    size_t jit_eager_fallbacks = 0;
#endif
    bench_stats_print("polymerpim", &stats);
    bench_stages_report("polymerpim", &stages);
    T final_error = 0;
    for (T error : last_errors) final_error = std::max(final_error, error);
    std::cout << "adaptive_image: channels=" << channels
              << " coarse_updates=" << iterations * channels - fine_updates
              << " fine_updates=" << fine_updates
              << " final_error=" << final_error
              << " compute_launches=" << runtime_stats.compute_launches
              << " vertical_fusions=" << runtime_stats.vertical_fusions
              << " horizontal_fusions=" << runtime_stats.horizontal_fusions
              << " jit_kernel_compiles=" << runtime_stats.jit_kernel_compiles
              << " jit_kernel_cache_hits="
              << runtime_stats.jit_kernel_cache_hits
              << " binary_switches=" << runtime_stats.binary_switches
              << " jit_pipeline_fallbacks=" << jit_pipeline_fallbacks
              << " jit_eager_fallbacks=" << jit_eager_fallbacks << std::endl;

    if (check_correctness) {
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
