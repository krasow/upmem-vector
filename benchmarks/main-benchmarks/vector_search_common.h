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
  int32_t key;
  uint32_t reserved;
} vector_search_result_t;

typedef struct {
  int32_t score;
  uint32_t index;
} vector_search_match_t;

static inline void vector_search_result_init(vector_search_result_t *result) {
  result->key = -1;
  result->reserved = 0;
}

/* Retain the single best packed score/ID key. */
static inline void vector_search_result_insert(vector_search_result_t *result,
                                               int32_t key) {
  if (key > result->key) {
    result->key = key;
  }
}

static inline void vector_search_result_merge(
    vector_search_result_t *dest, const vector_search_result_t *src) {
  vector_search_result_insert(dest, src->key);
}

/*
 * Pack score and ID into one signed 32-bit reduction key.  Larger is better;
 * ties in additive score choose the smaller global dataset index.
 */
static inline int32_t vector_search_pack_key(int32_t score, uint32_t index,
                                             uint64_t n, uint32_t dimensions) {
  return (int32_t)(((int64_t)score + 2 * dimensions) * (int64_t)n +
                   ((int64_t)n - 1 - index));
}

static inline vector_search_match_t vector_search_unpack_key(
    int32_t key, uint64_t n, uint32_t dimensions) {
  vector_search_match_t match;
  match.score = key / (int64_t)n - 2 * (int32_t)dimensions;
  match.index = (uint32_t)((int64_t)n - 1 - (key % (int64_t)n));
  return match;
}

static inline int vector_search_key_range_is_valid(uint64_t n,
                                                   uint32_t dimensions) {
  return n > 0 && (((uint64_t)4 * dimensions + 1) * n - 1) <= INT32_MAX;
}

#endif
