# Makefile for CUDA Matrix Multiplication Project

CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

SRC_DIR = src
INC_DIR = inc
BUILD_DIR = build

SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRC_FILES))

TARGET = matrix_project

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) -I$(INC_DIR) -c $< -o $@

$(TARGET): $(OBJ_FILES)
	$(NVCC) $^ -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean

# Added CNN compilation rules
cnn: src/CNN.cu src/Matrix.cu src/DataLoader.cu src/main.cu
	nvcc -o cnn src/CNN.cu src/Matrix.cu src/DataLoader.cu src/main.cu -Iinc