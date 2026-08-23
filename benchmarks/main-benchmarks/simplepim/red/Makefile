ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

ifndef SIMPLE_PIM_LIB
$(error SIMPLE_PIM_LIB is not defined)
endif

BIN_DIR := bin
HOST := $(BIN_DIR)/host

# Sources
HOST_SRC := host.c
SIMPLE_PIM_SRC := \
    $(SIMPLE_PIM_LIB)/processing/ProcessingHelperHost.c \
    $(SIMPLE_PIM_LIB)/communication/CommHelper.c \
    $(SIMPLE_PIM_LIB)/communication/CommOps.c \
    $(SIMPLE_PIM_LIB)/management/SmallTableInit.c \
    $(SIMPLE_PIM_LIB)/management/Management.c \
	$(SIMPLE_PIM_LIB)/processing/gen_red/GenRed.c

# Compiler flags
TESTS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../..)
CFLAGS := -std=c99 -O3 -fopenmp -I$(SIMPLE_PIM_LIB) -I$(TESTS_DIR)
LIBS := -lm -ldl `dpu-pkg-config --cflags --libs dpu`

# Targets
all: $(HOST)

$(HOST): $(HOST_SRC) $(SIMPLE_PIM_SRC)
	@mkdir -p $(BIN_DIR)
	gcc $(CFLAGS) $(EXTRA_FLAGS) $(HOST_SRC) $(SIMPLE_PIM_SRC) -o $(HOST) $(LIBS)

clean:
	rm -rf $(BIN_DIR)
