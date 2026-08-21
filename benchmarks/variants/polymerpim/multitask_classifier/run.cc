#include <benchmark.h>
#include <vectordpu.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../../multitask_classifier_common.h"
#include "Param.h"

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
    DpuRuntime::get().init(nr_dpus);
    bench_stage_end(&stages);

    {
      std::vector<std::vector<T> > host_features(FEATURES);
      std::vector<T> host_classes;
      bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
      for (auto& feature : host_features) feature.resize(N);
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
          for (uint32_t d = 0; d < FEATURES; d++)
            host_features[d][i] = (T)flow_feature_value(i, d);
        }
      }
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_WRITE);
      std::vector<std::string> feature_names;
      std::vector<dpu_vector<T> > features;
      feature_names.reserve(FEATURES);
      features.reserve(FEATURES);
      for (uint32_t d = 0; d < FEATURES; d++) {
        feature_names.push_back("flow_feature_" + std::to_string(d));
        features.push_back(dpu_vector<T>::from_cpu(
            host_features[d], feature_names.back(), VECTORDPU_SOURCE_LOCATION));
      }
      auto class_ids = dpu_vector<T>::from_cpu(host_classes, "flow_class",
                                               VECTORDPU_SOURCE_LOCATION);
      dpu_fence();
      bench_stage_end(&stages);

      using ReductionResult = typename dpu_vector<T>::reduction_result_t;
      std::vector<T> weights((uint64_t)CLASSES * FEATURES, 0);
      svm_metrics_t metrics{};

      auto run_epoch = [&](BenchStages& active_stages) {
        std::vector<dpu_future<T> > gradient_futures;
        std::vector<dpu_future<T> > violation_futures;
        std::vector<dpu_vector<T> > factors;
        std::vector<std::string> factor_names;
        gradient_futures.reserve((uint64_t)CLASSES * FEATURES);
        violation_futures.reserve(CLASSES);
        factors.reserve(CLASSES);
        factor_names.reserve(CLASSES);

        bench_stage_begin(&active_stages, BENCH_STAGE_KERNEL);
        for (uint32_t c = 0; c < CLASSES; c++) {
          std::vector<uint32_t> model_weights(FEATURES);
          for (uint32_t d = 0; d < FEATURES; d++)
            model_weights[d] = (uint32_t)weights[(uint64_t)c * FEATURES + d];

          std::vector<dpu_vector<T> > operands(features.begin() + 1,
                                               features.end());
          operands.push_back(class_ids);
          auto factor =
              features[0]
                  .transform(
                      [c](const std::vector<dpu_expr<T> >& x) {
                        auto score = x[0] * dpu_expr<T>::scalar_var(0);
                        for (uint32_t d = 1; d < FEATURES; d++)
                          score = score +
                                  x[d] * dpu_expr<T>::scalar_var((uint8_t)d);

                        auto label = (x[FEATURES] == (T)c) * (T)2 - (T)1;
                        auto active = (label * score) <
                                      dpu_expr<T>::scalar((T)SVM_MARGIN);
                        return active * (dpu_expr<T>::scalar((T)-1) * label);
                      },
                      operands, model_weights)
                  .vec;

          factor_names.push_back("active_class_" + std::to_string(c));
          factor.data_desc_ref()->debug_name = factor_names.back().c_str();
          factors.push_back(std::move(factor));

          auto batch = factors.back().reduction_batch();
          violation_futures.push_back(
              batch.add([](const std::vector<dpu_expr<T> >& x) {
                return x[0].sqr().sum();
              }));
          for (uint32_t d = 0; d < FEATURES; d++) {
            gradient_futures.push_back(batch.add(
                [](const std::vector<dpu_expr<T> >& x) {
                  return (x[0] * x[1]).sum();
                },
                {features[d]}));
          }
          batch.submit();
        }
        dpu_fence();
        bench_stage_end(&active_stages);

        bench_stage_begin(&active_stages, BENCH_STAGE_READ);
        std::vector<int64_t> gradients((uint64_t)CLASSES * FEATURES);
        metrics.margin_violations = 0;
        for (uint32_t c = 0; c < CLASSES; c++)
          metrics.margin_violations +=
              (uint64_t)(ReductionResult)violation_futures[c].get();
        for (uint32_t i = 0; i < CLASSES * FEATURES; i++)
          gradients[i] = (int64_t)(ReductionResult)gradient_futures[i].get();
        bench_stage_end(&active_stages);

        bench_stage_begin(&active_stages, BENCH_STAGE_MERGE);
        for (uint32_t i = 0; i < CLASSES * FEATURES; i++)
          weights[i] = (T)svm_update_weight(weights[i], gradients[i], N);
        bench_stage_end(&active_stages);

        bench_stage_begin(&active_stages, BENCH_STAGE_KERNEL);
        std::vector<uint32_t> all_weights((uint64_t)CLASSES * FEATURES);
        for (uint32_t i = 0; i < CLASSES * FEATURES; i++)
          all_weights[i] = (uint32_t)weights[i];
        std::vector<dpu_vector<T> > eval_operands(features.begin() + 1,
                                                  features.end());
        eval_operands.push_back(class_ids);
        auto correct_future = features[0].reduce(
            [&](const std::vector<dpu_expr<T> >& x) {
              auto score_for = [&](uint32_t c) {
                auto score =
                    x[0] * dpu_expr<T>::scalar_var((uint8_t)(c * FEATURES));
                for (uint32_t d = 1; d < FEATURES; d++) {
                  score = score + x[d] * dpu_expr<T>::scalar_var(
                                             (uint8_t)(c * FEATURES + d));
                }
                return score;
              };

              // One variadic argmax over the CLASSES scores replaces the
              // compare+dual-select chain; .label is the predicted class.
              std::vector<dpu_expr<T> > scores;
              scores.reserve(CLASSES);
              for (uint32_t c = 0; c < CLASSES; c++)
                scores.push_back(score_for(c));
              auto best_class = argmax(scores).label;

              auto actual = x[FEATURES];
              auto correct = (best_class - actual) == (T)0;
              return correct.sum();
            },
            eval_operands, all_weights);
        dpu_fence();
        bench_stage_end(&active_stages);

        bench_stage_begin(&active_stages, BENCH_STAGE_READ);
        metrics.correct_predictions =
            (uint64_t)(ReductionResult)correct_future.get();
        bench_stage_end(&active_stages);
      };

      Timer warmup_timer;
      BenchStats warmup_stats;
      bench_stats_init(&warmup_stats);
      for (uint32_t i = 0; i < warmup_iterations; i++) {
        std::fill(weights.begin(), weights.end(), 0);
        bench_start(&warmup_timer, 0);
        run_epoch(warm_stages);
        bench_stop(&warmup_timer, 0);
        bench_stats_update(&warmup_stats, warmup_timer.time[0]);
      }
      if (warmup_iterations > 0)
        bench_stats_print("polymerpim_warmup", &warmup_stats);

      std::fill(weights.begin(), weights.end(), 0);
      BenchStats stats;
      bench_stats_init(&stats);
      Timer timer;
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

    DpuRuntime::get().shutdown();
    return 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
