#include "../inc/Matrix.h"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols) {}

float& Matrix::operator()(int row, int col) {
    return data[row * cols + col];
}

const float& Matrix::operator()(int row, int col) const {
    return data[row * cols + col];
}

int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}

void Matrix::randomize() {
    for (auto& val : data) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << "\n";
    }
}

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int ARows, int ACols, int BCols) {
    // Define shared memory for tiles of A and B
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0;

    for (int t = 0; t < (ACols + 15) / 16; ++t) {
        // Load tiles into shared memory
        if (row < ARows && t * 16 + threadIdx.x < ACols) {
            tileA[threadIdx.y][threadIdx.x] = A[row * ACols + t * 16 + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < BCols && t * 16 + threadIdx.y < ACols) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * BCols + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform computation for the tile
        for (int k = 0; k < 16; ++k) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < ARows && col < BCols) {
        C[row * BCols + col] = value;
    }
}

void multiplyMatricesCUDA(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    int ARows = A.getRows();
    int ACols = A.getCols();
    int BCols = B.getCols();

    size_t sizeA = ARows * ACols * sizeof(float);
    size_t sizeB = ACols * BCols * sizeof(float);
    size_t sizeC = ARows * BCols * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.getData().data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.getData().data(), sizeB, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((BCols + blockDim.x - 1) / blockDim.x, (ARows + blockDim.y - 1) / blockDim.y);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    matrixMultiplyKernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, ARows, ACols, BCols);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpyAsync(C.getData().data(), d_C, sizeC, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}