#ifndef MULTITASK_CLASSIFIER_COMMON_H
#define MULTITASK_CLASSIFIER_COMMON_H

#include <stdint.h>

#define FLOW_FEATURES 8u
#define FLOW_CLASSES 4u
#define FLOW_ROW_WORDS (FLOW_FEATURES + 2u)
#define SVM_MARGIN 12
#define SVM_WEIGHT_DECAY 8
#define SVM_MODE_TRAIN 0
#define SVM_MODE_EVALUATE 1
#define SVM_STATE_WORDS (4u + FLOW_CLASSES * FLOW_FEATURES)

typedef struct {
  uint64_t margin_violations;
  uint64_t correct_predictions;
} svm_metrics_t;

static inline uint32_t flow_class_for_row(uint64_t row) {
  return (uint32_t)((row * 5u + 1u) & (FLOW_CLASSES - 1u));
}

static inline int32_t flow_feature_value(uint64_t row, uint32_t feature) {
  uint32_t mixed = (uint32_t)row * 1103515245u + 12345u + feature * 2654435761u;
  mixed ^= mixed >> 16;

  /* Two standardized indicators are characteristic of each class. */
  int32_t center = flow_class_for_row(row) == feature / 2u ? 2 : -1;
  int32_t noise = (int32_t)(mixed % 3u) - 1;
  return center + noise;
}

static inline int32_t svm_signed_label(int32_t class_id, uint32_t classifier) {
  return class_id == (int32_t)classifier ? 1 : -1;
}

static inline int64_t svm_div_round_closest(int64_t value, int64_t divisor) {
  if (value < 0) {
    return -((-value + divisor / 2) / divisor);
  }
  return (value + divisor / 2) / divisor;
}

static inline int32_t svm_update_weight(int32_t weight, int64_t gradient,
                                        uint64_t rows) {
  int64_t gradient_divisor = (int64_t)(rows / 2u);
  if (gradient_divisor < 1) {
    gradient_divisor = 1;
  }

  int64_t data_step = svm_div_round_closest(gradient, gradient_divisor);
  int64_t decay_step = svm_div_round_closest(weight, SVM_WEIGHT_DECAY);
  return (int32_t)((int64_t)weight - data_step - decay_step);
}

#endif
