ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

NR_TASKLETS ?= 12

BIN_DIR   := bin
HOST      := $(BIN_DIR)/host_baseline
DPU       := $(BIN_DIR)/baseline.dpu

TESTS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../..)
CFLAGS    := -O3 -I$(TESTS_DIR)
LIBS      := -lm -ldl `dpu-pkg-config --cflags --libs dpu`
DPU_FLAGS := -O3 -DNR_TASKLETS=$(NR_TASKLETS) $(EXTRA_FLAGS)

all: $(HOST) $(DPU)

$(HOST): host/baseline.cc Param.h | $(BIN_DIR)
	g++ $(CFLAGS) $(EXTRA_FLAGS) host/baseline.cc -o $@ $(LIBS)

$(DPU): dpu/dpu_baseline.c Param.h | $(BIN_DIR)
	dpu-upmem-dpurte-clang $(DPU_FLAGS) -o $@ $<

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR)
