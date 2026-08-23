#include <benchmark.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dpu>
#include <vector>

#include "../../../multitask_classifier_common.h"
#include "../Param.h"

#define CHECK(x) DPU_ASSERT(x)
#define NR_TASKLETS 12
#define RESULT_WORDS (FLOW_FEATURES + 2u)

typedef struct {
  uint32_t rows_offset;
  uint32_t weights_offset;
  uint32_t result_offset;
  uint32_t num_elements;
  uint32_t mode;
  uint32_t classifier;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

static_assert(FEATURES == FLOW_FEATURES,
              "intrusion workload requires 8 features");
static_assert(CLASSES == FLOW_CLASSES, "intrusion workload requires 4 classes");

int main() {
  struct dpu_set_t dpu_set, dpu;
  uint32_t nr_dpus = dpu_number;

  BenchStages stages;
  BenchStages warm_stages;
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  CHECK(dpu_alloc(
      nr_dpus, getenv("UPMEM_PROFILE") ? getenv("UPMEM_PROFILE") : "backend=hw",
      &dpu_set));
  CHECK(dpu_load(dpu_set, "./bin/baseline.dpu", NULL));
  bench_stage_end(&stages);

  uint64_t rows_per_dpu = N / nr_dpus;
  uint32_t rows_bytes = (uint32_t)(rows_per_dpu * FLOW_ROW_WORDS * sizeof(T));
  uint32_t weights_offset = (rows_bytes + 7u) & ~7u;
  uint32_t weights_bytes = CLASSES * FEATURES * sizeof(T);
  uint32_t result_offset = (weights_offset + weights_bytes + 7u) & ~7u;
  uint32_t result_bytes = NR_TASKLETS * RESULT_WORDS * sizeof(RED_T);

  std::vector<DPU_LAUNCH_ARGS> args(nr_dpus);
  for (uint32_t i = 0; i < nr_dpus; i++) {
    args[i].rows_offset = 0;
    args[i].weights_offset = weights_offset;
    args[i].result_offset = result_offset;
    args[i].num_elements = (uint32_t)rows_per_dpu;
    args[i].mode = SVM_MODE_TRAIN;
    args[i].classifier = 0;
  }

  std::vector<T> rows;
  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  rows.resize((uint64_t)N * FLOW_ROW_WORDS);
  bench_stage_end(&stages);

  bench_stage_begin(&stages, BENCH_STAGE_LOAD);
  if (load_ref) {
    char path[512];
    snprintf(path, sizeof(path), "%s/AoS/rows.bin", ref_path);
    bench_load_bin(path, rows.data(), rows.size() * sizeof(T));
  } else {
    for (uint64_t i = 0; i < N; i++) {
      for (uint32_t d = 0; d < FEATURES; d++)
        rows[i * FLOW_ROW_WORDS + d] = (T)flow_feature_value(i, d);
      rows[i * FLOW_ROW_WORDS + FEATURES] = (T)flow_class_for_row(i);
      rows[i * FLOW_ROW_WORDS + FEATURES + 1] = 0;
    }
  }
  bench_stage_end(&stages);

  uint32_t dpu_index;
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  DPU_FOREACH(dpu_set, dpu, dpu_index) {
    CHECK(dpu_prepare_xfer(
        dpu, &rows[(uint64_t)dpu_index * rows_per_dpu * FLOW_ROW_WORDS]));
  }
  CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                      rows_bytes, DPU_XFER_DEFAULT));
  bench_stage_end(&stages);

  std::vector<T> weights((uint64_t)CLASSES * FEATURES, 0);
  std::vector<int64_t> gradients((uint64_t)CLASSES * FEATURES);
  std::vector<RED_T> partials((uint64_t)nr_dpus * NR_TASKLETS * RESULT_WORDS);
  svm_metrics_t metrics{};

  auto send_args = [&]() {
    DPU_FOREACH(dpu_set, dpu, dpu_index)
    CHECK(dpu_prepare_xfer(dpu, &args[dpu_index]));
    CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0, sizeof(args[0]),
                        DPU_XFER_DEFAULT));
  };

  auto gather_results = [&]() {
    DPU_FOREACH(dpu_set, dpu, dpu_index) {
      CHECK(dpu_prepare_xfer(
          dpu, &partials[(uint64_t)dpu_index * NR_TASKLETS * RESULT_WORDS]));
    }
    CHECK(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                        result_offset, result_bytes, DPU_XFER_DEFAULT));
  };

  auto run_epoch = [&](BenchStages& active_stages) {
    std::fill(gradients.begin(), gradients.end(), 0);
    metrics.margin_violations = 0;

    bench_stage_begin(&active_stages, BENCH_STAGE_WRITE);
    CHECK(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, weights_offset,
                           weights.data(), weights_bytes, DPU_XFER_DEFAULT));
    bench_stage_end(&active_stages);

    for (uint32_t c = 0; c < CLASSES; c++) {
      bench_stage_begin(&active_stages, BENCH_STAGE_WRITE);
      for (auto& arg : args) {
        arg.mode = SVM_MODE_TRAIN;
        arg.classifier = c;
      }
      send_args();
      bench_stage_end(&active_stages);

      bench_stage_begin(&active_stages, BENCH_STAGE_KERNEL);
      CHECK(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
      bench_stage_end(&active_stages);

      bench_stage_begin(&active_stages, BENCH_STAGE_READ);
      gather_results();
      bench_stage_end(&active_stages);

      bench_stage_begin(&active_stages, BENCH_STAGE_MERGE);
      for (uint64_t i = 0; i < (uint64_t)nr_dpus * NR_TASKLETS; i++) {
        RED_T* partial = &partials[i * RESULT_WORDS];
        for (uint32_t d = 0; d < FEATURES; d++)
          gradients[(uint64_t)c * FEATURES + d] += partial[d];
        metrics.margin_violations += partial[FEATURES];
      }
      bench_stage_end(&active_stages);
    }

    bench_stage_begin(&active_stages, BENCH_STAGE_MERGE);
    for (uint32_t i = 0; i < CLASSES * FEATURES; i++)
      weights[i] = (T)svm_update_weight(weights[i], gradients[i], N);
    bench_stage_end(&active_stages);

    bench_stage_begin(&active_stages, BENCH_STAGE_WRITE);
    CHECK(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, weights_offset,
                           weights.data(), weights_bytes, DPU_XFER_DEFAULT));
    for (auto& arg : args) {
      arg.mode = SVM_MODE_EVALUATE;
      arg.classifier = 0;
    }
    send_args();
    bench_stage_end(&active_stages);

    bench_stage_begin(&active_stages, BENCH_STAGE_KERNEL);
    CHECK(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    bench_stage_end(&active_stages);

    bench_stage_begin(&active_stages, BENCH_STAGE_READ);
    gather_results();
    bench_stage_end(&active_stages);

    bench_stage_begin(&active_stages, BENCH_STAGE_MERGE);
    metrics.correct_predictions = 0;
    for (uint64_t i = 0; i < (uint64_t)nr_dpus * NR_TASKLETS; i++)
      metrics.correct_predictions += partials[i * RESULT_WORDS];
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
  if (warmup_iterations > 0)
    bench_stats_print("baseline_warmup", &warmup_stats);

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

  bench_stats_print("baseline", &stats);
  bench_stages_report("baseline", &stages);
  bench_stages_report("baseline_cold", &warm_stages);
  printf("baseline_result margin_violations=%llu accuracy=%llu/%llu\n",
         (unsigned long long)metrics.margin_violations,
         (unsigned long long)metrics.correct_predictions,
         (unsigned long long)N);

  if (check_correctness && load_ref) {
    std::vector<T> expected(weights.size());
    svm_metrics_t expected_metrics{};
    char path[512];
    snprintf(path, sizeof(path), "%s/ref_weights.bin", ref_path);
    bench_load_bin(path, expected.data(), expected.size() * sizeof(T));
    snprintf(path, sizeof(path), "%s/ref_metrics.bin", ref_path);
    bench_load_bin(path, &expected_metrics, sizeof(expected_metrics));
    bool ok =
        weights == expected &&
        metrics.margin_violations == expected_metrics.margin_violations &&
        metrics.correct_predictions == expected_metrics.correct_predictions;
    if (ok)
      printf("the result is correct\n");
    else
      printf("Mismatch: got violations=%llu accuracy=%llu\n",
             (unsigned long long)metrics.margin_violations,
             (unsigned long long)metrics.correct_predictions);
  }

  CHECK(dpu_free(dpu_set));
  return 0;
}
