#include <benchmark.h>
#include <polymerpim.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../../multitask_classifier_common.h"
#include "Param.h"

using namespace polymerpim;

static_assert(FEATURES == FLOW_FEATURES,
              "intrusion workload requires 8 features");
static_assert(CLASSES == FLOW_CLASSES, "intrusion workload requires 4 classes");

int main() {
  try {
#if !JIT
    std::cerr << "Exception: polymerpim_multitask_classifier requires JIT mode"
              << std::endl;
    return 1;
#else
    const char* nr_dpus_env = std::getenv("NR_DPUS");
    int nr_dpus = nr_dpus_env ? std::stoi(nr_dpus_env) : 64;

    BenchStages stages;
    BenchStages warm_stages;
    bench_stages_init(&stages);
    bench_stages_init(&warm_stages);

    bench_stage_begin(&stages, BENCH_STAGE_INIT);
    init(nr_dpus);
    bench_stage_end(&stages);

    {
      std::vector<std::vector<T> > host_features(FEATURES);
      std::vector<T> host_classes;
      bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
      for (auto& feature : host_features) {
        feature.resize(N);
      }
      host_classes.resize(N);
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_LOAD);
      if (load_ref) {
        for (uint32_t d = 0; d < FEATURES; d++) {
          std::string path =
              std::string(ref_path) + "/SoA/col_" + std::to_string(d) + ".bin";
          bench_load_bin(path.c_str(), host_features[d].data(), N * sizeof(T));
        }
        bench_load_bin((std::string(ref_path) + "/SoA/class_ids.bin").c_str(),
                       host_classes.data(), N * sizeof(T));
      } else {
        for (uint64_t i = 0; i < N; i++) {
          host_classes[i] = (T)flow_class_for_row(i);
          for (uint32_t d = 0; d < FEATURES; d++) {
            host_features[d][i] = (T)flow_feature_value(i, d);
          }
        }
      }
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_WRITE);
      std::vector<std::string> feature_names;
      std::vector<DPUVector<T> > features;
      feature_names.reserve(FEATURES);
      features.reserve(FEATURES);
      for (uint32_t d = 0; d < FEATURES; d++) {
        feature_names.push_back("flow_feature_" + std::to_string(d));
        features.emplace_back(host_features[d], feature_names.back());
      }
      DPUVector<T> class_ids(host_classes, "flow_class");
      sync();
      bench_stage_end(&stages);

      using ReductionResult = typename DPUVector<T>::reduction_result_t;
      std::vector<T> weights((uint64_t)CLASSES * FEATURES, 0);
      svm_metrics_t metrics{};

      auto run_epoch = [&](BenchStages& active_stages) {
        std::vector<DpuFuture<T> > gradient_futures;
        std::vector<DpuFuture<T> > violation_futures;
        std::vector<DPUVector<T> > factors;
        gradient_futures.reserve((uint64_t)CLASSES * FEATURES);
        violation_futures.reserve(CLASSES);
        factors.reserve(CLASSES);

        bench_stage_begin(&active_stages, BENCH_STAGE_KERNEL);
        for (uint32_t c = 0; c < CLASSES; c++) {
          DPUVector<T> score(features[0].size());
          for (uint32_t d = 0; d < FEATURES; ++d) {
            score = score + features[d] * weights[(uint64_t)c * FEATURES + d];
          }

          auto label = (class_ids == (T)c) * (T)2 - (T)1;
          auto active = label * score < (T)SVM_MARGIN;
          factors.emplace_back(active * -label);

          violation_futures.push_back(sum(sqr(factors.back())));
          for (uint32_t d = 0; d < FEATURES; ++d) {
            gradient_futures.push_back(sum(factors.back() * features[d]));
          }
        }
        sync();
        bench_stage_end(&active_stages);

        bench_stage_begin(&active_stages, BENCH_STAGE_READ);
        std::vector<int64_t> gradients((uint64_t)CLASSES * FEATURES);
        metrics.margin_violations = 0;
        for (uint32_t c = 0; c < CLASSES; c++) {
          metrics.margin_violations +=
              (uint64_t)(ReductionResult)violation_futures[c].get();
        }
        for (uint32_t i = 0; i < CLASSES * FEATURES; i++) {
          gradients[i] = (int64_t)(ReductionResult)gradient_futures[i].get();
        }
        bench_stage_end(&active_stages);

        bench_stage_begin(&active_stages, BENCH_STAGE_MERGE);
        for (uint32_t i = 0; i < CLASSES * FEATURES; i++) {
          weights[i] = (T)svm_update_weight(weights[i], gradients[i], N);
        }
        bench_stage_end(&active_stages);

        bench_stage_begin(&active_stages, BENCH_STAGE_KERNEL);
        std::vector<DpuLazy<T> > scores;
        scores.reserve(CLASSES);
        for (uint32_t c = 0; c < CLASSES; ++c) {
          DPUVector<T> score(features[0].size());
          for (uint32_t d = 0; d < FEATURES; ++d) {
            score = score + features[d] * weights[(uint64_t)c * FEATURES + d];
          }
          scores.push_back(score);
        }
        auto correct_future = sum(argmax(scores) == class_ids);
        sync();
        bench_stage_end(&active_stages);

        bench_stage_begin(&active_stages, BENCH_STAGE_READ);
        metrics.correct_predictions =
            (uint64_t)(ReductionResult)correct_future.get();
        bench_stage_end(&active_stages);
      };

      BenchTimer warmup_timer;
      BenchStats warmup_stats;
      bench_stats_init(&warmup_stats);
      for (uint32_t i = 0; i < warmup_iterations; i++) {
        std::fill(weights.begin(), weights.end(), 0);
        bench_start(&warmup_timer, 0);
        run_epoch(warm_stages);
        bench_stop(&warmup_timer, 0);
        bench_stats_update(&warmup_stats, warmup_timer.time[0]);
      }
      if (warmup_iterations > 0) {
        bench_stats_print("polymerpim_warmup", &warmup_stats);
      }

      std::fill(weights.begin(), weights.end(), 0);
      BenchStats stats;
      bench_stats_init(&stats);
      BenchTimer timer;
      for (uint32_t i = 0; i < iterations; i++) {
        bench_start(&timer, 0);
        run_epoch(stages);
        bench_stop(&timer, 0);
        bench_stats_update(&stats, timer.time[0]);
      }

      bench_stats_print("polymerpim", &stats);
      bench_stages_report("polymerpim", &stages);
      bench_stages_report("polymerpim_cold", &warm_stages);
      std::cout << "polymerpim_result margin_violations="
                << metrics.margin_violations
                << " accuracy=" << metrics.correct_predictions << "/" << N
                << std::endl;

      if (check_correctness && load_ref) {
        std::vector<T> expected(weights.size());
        svm_metrics_t expected_metrics{};
        bench_load_bin((std::string(ref_path) + "/ref_weights.bin").c_str(),
                       expected.data(), expected.size() * sizeof(T));
        bench_load_bin((std::string(ref_path) + "/ref_metrics.bin").c_str(),
                       &expected_metrics, sizeof(expected_metrics));
        bool ok =
            weights == expected &&
            metrics.margin_violations == expected_metrics.margin_violations &&
            metrics.correct_predictions == expected_metrics.correct_predictions;
        if (!ok) {
          std::cout << "Mismatch: got violations=" << metrics.margin_violations
                    << " accuracy=" << metrics.correct_predictions << std::endl;
        } else {
          std::cout << "the result is correct" << std::endl;
        }
      }
    }

    shutdown();
    return 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
