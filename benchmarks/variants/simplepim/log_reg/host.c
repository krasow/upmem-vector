#include <assert.h>
#include <benchmark.h>
#include <dpu.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "Param.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/gen_red/GenRed.h"

void init_input(T* elements) {
  /* Generate synthetic data: each row has dim features + 1 label.
     Features are in range [-5, 5], labels in {0, 1}. */
  for (uint64_t i = 0; i < num_elements; i++) {
    T dot = 0;
    for (uint32_t j = 0; j < dim; j++) {
      T v = (T)((int)(i + j) % 11 - 5); /* features in [-5,5] */
      elements[i * (dim + 1) + j] = v;
      dot += v;
    }
    /* simple linear rule for labels */
    elements[i * (dim + 1) + dim] = (dot >= 0) ? (T)1.0f : (T)0.0f;
  }
}

void load_bin(const char* filename, void* data, size_t size) {
  FILE* f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open %s for reading\n", filename);
    exit(1);
  }
  size_t got = fread(data, 1, size, f);
  if (got != size) {
    fprintf(stderr, "Short read from %s: got %zu, expected %zu\n", filename,
            got, size);
    exit(1);
  }
  fclose(f);
}

void run() {
  simplepim_management_t* table_management = table_management_init(dpu_number);
  printf("dim: %d, num_elem: %lu, iter: %d, lr: %f\n", dim, num_elements, iter,
         lr);

  /* inputs: [X | Y] stored row-major, (dim+1) floats per row */
  T* elements = (T*)malloc_scatter_aligned(num_elements, (dim + 1) * sizeof(T),
                                           table_management);
  T* weights =
      (T*)malloc_broadcast_aligned(1, sizeof(T) * dim, table_management);
  T* cpu_grads = (T*)calloc(dim, sizeof(T));

  if (load_ref) {
    char path[1024];
    printf("Loading reference data from %s...\n", ref_path);
    sprintf(path, "%s/ref_t1.bin", ref_path);
    load_bin(path, elements, num_elements * (dim + 1) * sizeof(T));
    sprintf(path, "%s/ref_w.bin", ref_path);
    load_bin(path, weights, dim * sizeof(T));
    sprintf(path, "%s/ref_grads.bin", ref_path);
    load_bin(path, cpu_grads, dim * sizeof(T));
  } else {
    init_input(elements);
    for (int i = 0; i < dim; i++) weights[i] = 0.0f;
  }

  simplepim_scatter("t1", elements, num_elements, (dim + 1) * sizeof(T),
                    table_management);
  uint32_t data_offset = lookup_table("t1", table_management)->end;
  simplepim_broadcast("t2", weights, 1, dim * sizeof(T), table_management);

  handle_t* va_handle = create_handle("log_reg_funcs", REDUCE);

  BenchStats stats;
  bench_stats_init(&stats);
  Timer timer;
  bench_start(&timer, 0);
  T* res = table_gen_red("t1", "t3", dim * sizeof(T), 1, va_handle,
                         table_management, data_offset);
  bench_stop(&timer, 0);
  bench_stats_update(&stats, timer.time[0]);
  bench_stats_print("simplepim", &stats);

  /* check correctness */
  if (load_ref && check_correctness) {
    int correct = 1;
    for (int i = 0; i < dim; i++) {
      float diff = (float)(res[i] - cpu_grads[i]);
      float rel =
          diff / (cpu_grads[i] != 0.0f ? (float)fabs(cpu_grads[i]) : 1.0f);
      if (rel < -0.01f || rel > 0.01f) {
        printf("Gradient mismatch at dim %d: dpu=%e cpu=%e (rel=%e)\n", i,
               (double)res[i], (double)cpu_grads[i], (double)rel);
        correct = 0;
      }
    }
    if (correct) printf("the result is correct\n");
  }

  free(res);
  free(cpu_grads);
}

int main(int argc, char** argv) {
  run();
  return 0;
}
