# https://github.com/CMU-SAFARI/prim-benchmarks/tree/main 
# leveraged the above repository to create Makefile for DPU and host code compilation

DPU_DIR := dpu
HOST_DIR := host
TEST_DIR := test
BUILDDIR ?= bin
NR_DPUS ?= 16
NR_TASKLETS ?= 16

CONFIG_FLAGS ?= -DENABLE_DPU_LOGGING=2 

HOST_TARGET := ${BUILDDIR}/libvectordpu
DPU_TARGET := ${BUILDDIR}/runtime.dpu
TEST_TARGET := ${TEST_DIR}/vectordpu_test

COMMON_INCLUDES := common
HOST_INCLUDES := host
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.cc)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)
TEST_SOURCES := $(wildcard ${TEST_DIR}/*.cc)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}
HOST_FLAGS := ${COMMON_FLAGS} -O3 `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS} ${CONFIG_FLAGS}
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS}

all: ${HOST_TARGET} ${DPU_TARGET}

${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES}
	$(CXX) -shared -fPIC -o $@.so ${HOST_SOURCES} ${HOST_FLAGS} -Wl,-rpath,$(abspath $(dir $@))


${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

$(TEST_TARGET): all
	$(CXX) -o $@ $(TEST_SOURCES) -I$(HOST_INCLUDES) ${COMMON_FLAGS} -O3 -L$(BUILDDIR) -lvectordpu

clean:
	$(RM) -r $(BUILDDIR) $(TEST_TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)