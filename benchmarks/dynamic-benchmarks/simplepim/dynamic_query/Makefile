ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

ifndef SIMPLE_PIM_LIB
$(error SIMPLE_PIM_LIB is not defined)
endif

BIN_DIR := bin
HOST := $(BIN_DIR)/host
GENERATED := .queries.generated
HOST_SRC := host.c
SIMPLE_PIM_SRC := \
	$(SIMPLE_PIM_LIB)/processing/ProcessingHelperHost.c \
	$(SIMPLE_PIM_LIB)/communication/CommHelper.c \
	$(SIMPLE_PIM_LIB)/communication/CommOps.c \
	$(SIMPLE_PIM_LIB)/management/SmallTableInit.c \
	$(SIMPLE_PIM_LIB)/management/Management.c \
	$(SIMPLE_PIM_LIB)/processing/gen_red/GenRed.c

BENCHMARK_INCLUDE := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../../../main-benchmarks)
CFLAGS := -std=c99 -O3 -fopenmp -I$(SIMPLE_PIM_LIB) -I$(BENCHMARK_INCLUDE)
LIBS := -lm -ldl `dpu-pkg-config --cflags --libs dpu`

.PHONY: all clean

all: $(HOST)

$(GENERATED): generate_queries.py Param.h $(wildcard Param.generated.h) ../../dynamic_query.csv
	python3 generate_queries.py
	@touch $@

$(HOST): $(HOST_SRC) $(SIMPLE_PIM_SRC) $(GENERATED)
	@mkdir -p $(BIN_DIR)
	gcc $(CFLAGS) $(EXTRA_FLAGS) $(HOST_SRC) $(SIMPLE_PIM_SRC) -o $@ $(LIBS)

clean:
	rm -rf $(BIN_DIR) query_*_funcs $(GENERATED)
