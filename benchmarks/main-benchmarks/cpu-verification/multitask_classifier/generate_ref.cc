#include <benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../../multitask_classifier_common.h"
#include "Param.h"

#define DoNotOptimize(x) asm volatile("" : : "r,m"(x) : "memory")

static_assert(FEATURES == FLOW_FEATURES,
              "intrusion workload requires 8 features");
static_assert(CLASSES == FLOW_CLASSES, "intrusion workload requires 4 classes");

static svm_metrics_t run_classifier(const std::vector<T>& rows,
                                    std::vector<T>& weights) {
  std::fill(weights.begin(), weights.end(), 0);
  std::vector<int64_t> gradients((uint64_t)CLASSES * FEATURES);
  svm_metrics_t metrics{};

  for (uint32_t it = 0; it < iterations; it++) {
    std::fill(gradients.begin(), gradients.end(), 0);
    metrics.margin_violations = 0;

    for (uint64_t i = 0; i < N; i++) {
      const T* row = &rows[i * FLOW_ROW_WORDS];
      int32_t class_id = row[FEATURES];

      for (uint32_t c = 0; c < CLASSES; c++) {
        uint64_t base = (uint64_t)c * FEATURES;
        int32_t score = 0;
        for (uint32_t d = 0; d < FEATURES; d++) {
          score += weights[base + d] * row[d];
        }

        int32_t label = svm_signed_label(class_id, c);
        if (label * score < SVM_MARGIN) {
          metrics.margin_violations++;
          for (uint32_t d = 0; d < FEATURES; d++) {
            gradients[base + d] += -label * row[d];
          }
        }
      }
    }

    for (uint32_t c = 0; c < CLASSES; c++) {
      for (uint32_t d = 0; d < FEATURES; d++) {
        uint64_t index = (uint64_t)c * FEATURES + d;
        weights[index] =
            (T)svm_update_weight(weights[index], gradients[index], N);
      }
    }

    metrics.correct_predictions = 0;
    for (uint64_t i = 0; i < N; i++) {
      const T* row = &rows[i * FLOW_ROW_WORDS];
      int32_t best_score = 0;
      uint32_t best_class = 0;
      for (uint32_t c = 0; c < CLASSES; c++) {
        int32_t score = 0;
        for (uint32_t d = 0; d < FEATURES; d++) {
          score += weights[(uint64_t)c * FEATURES + d] * row[d];
        }
        if (c == 0 || score > best_score) {
          best_score = score;
          best_class = c;
        }
      }
      metrics.correct_predictions += best_class == (uint32_t)row[FEATURES];
    }
  }

  return metrics;
}

template <typename U>
static bool write_vector(const std::string& path,
                         const std::vector<U>& values) {
  FILE* file = fopen(path.c_str(), "wb");
  if (!file) {
    return false;
  }
  bool ok =
      fwrite(values.data(), sizeof(U), values.size(), file) == values.size();
  fclose(file);
  return ok;
}

int main() {
  system("mkdir -p data/AoS data/SoA");
  std::cout << "Generating intrusion classifier reference: N=" << N
            << " FEATURES=" << FEATURES << " CLASSES=" << CLASSES << std::endl;

  std::vector<T> rows((uint64_t)N * FLOW_ROW_WORDS, 0);
  std::vector<std::vector<T>> columns(FEATURES, std::vector<T>(N));
  std::vector<T> class_ids(N);

  for (uint64_t i = 0; i < N; i++) {
    T class_id = (T)flow_class_for_row(i);
    class_ids[i] = class_id;
    rows[i * FLOW_ROW_WORDS + FEATURES] = class_id;
    for (uint32_t d = 0; d < FEATURES; d++) {
      T value = (T)flow_feature_value(i, d);
      rows[i * FLOW_ROW_WORDS + d] = value;
      columns[d][i] = value;
    }
  }

  if (!write_vector("data/AoS/rows.bin", rows)) {
    std::cerr << "failed to write data/AoS/rows.bin" << std::endl;
    return 1;
  }
  for (uint32_t d = 0; d < FEATURES; d++) {
    std::string path = "data/SoA/col_" + std::to_string(d) + ".bin";
    if (!write_vector(path, columns[d])) {
      std::cerr << "failed to write " << path << std::endl;
      return 1;
    }
  }
  if (!write_vector("data/SoA/class_ids.bin", class_ids)) {
    std::cerr << "failed to write data/SoA/class_ids.bin" << std::endl;
    return 1;
  }

  if (bench_ref_data_only()) {
    std::cout << "Reference input generation complete." << std::endl;
    return 0;
  }

  std::vector<T> weights((uint64_t)CLASSES * FEATURES);
  auto start = std::chrono::high_resolution_clock::now();
  svm_metrics_t metrics = run_classifier(rows, weights);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count() /
              std::max(iterations, 1u);
  DoNotOptimize(weights.data());
  std::cout << "cpu_baseline (ms): " << ms << std::endl;

  if (!write_vector("data/ref_weights.bin", weights)) {
    std::cerr << "failed to write data/ref_weights.bin" << std::endl;
    return 1;
  }
  std::vector<svm_metrics_t> metric_file{metrics};
  if (!write_vector("data/ref_metrics.bin", metric_file)) {
    std::cerr << "failed to write data/ref_metrics.bin" << std::endl;
    return 1;
  }

  std::cout << "Final margin violations: " << metrics.margin_violations
            << ", accuracy: " << metrics.correct_predictions << "/" << N
            << std::endl;
  return 0;
}
