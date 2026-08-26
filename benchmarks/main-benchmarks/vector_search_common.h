#ifndef VECTOR_SEARCH_COMMON_H
#define VECTOR_SEARCH_COMMON_H

#include <limits.h>
#include <stdint.h>

/*
 * Dataset and query entries are random bipolar values.  This experimental
 * variant scores each dimension by addition instead of multiplication.  The
 * counter-based generator makes the dataset reproducible without storing a
 * second reference copy on disk.
 */
static inline uint64_t vector_search_mix64(uint64_t x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

static inline int32_t vector_search_dataset_value(uint32_t seed, uint64_t row,
                                                  uint32_t dim,
                                                  uint32_t dimensions) {
  uint64_t counter = row * (uint64_t)dimensions + dim;
  return (vector_search_mix64(((uint64_t)seed << 32) ^ counter) & 1) ? 1 : -1;
}

static inline int32_t vector_search_query_value(uint32_t seed,
                                                uint64_t query_id,
                                                uint32_t dim) {
  uint64_t counter = UINT64_C(0xd1b54a32d192ed03) ^
                     (query_id * UINT64_C(0x9e3779b97f4a7c15)) ^ dim;
  return (vector_search_mix64(((uint64_t)seed << 32) ^ counter) & 1) ? 1 : -1;
}

typedef struct {
  int32_t score;
  uint32_t index;
} vector_search_result_t;

static inline void vector_search_result_init(vector_search_result_t *result) {
  result->score = INT32_MIN;
  result->index = UINT32_MAX;
}

/* Retain the best score and use the lowest global row index for ties. */
static inline void vector_search_result_insert(vector_search_result_t *result,
                                               int32_t score, uint32_t index) {
  if (score > result->score ||
      (score == result->score && index < result->index)) {
    result->score = score;
    result->index = index;
  }
}

static inline void vector_search_result_merge(
    vector_search_result_t *dest, const vector_search_result_t *src) {
  vector_search_result_insert(dest, src->score, src->index);
}

static inline int vector_search_key_range_is_valid(uint64_t n,
                                                   uint32_t dimensions) {
  (void)dimensions;
  return n > 0 && n <= UINT32_MAX;
}

#endif
