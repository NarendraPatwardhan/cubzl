#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B) {
    // Compute the indices for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_A && col < cols_B) {
        float sum = 0.0f;
        for (int k = 0; k < cols_A; ++k) {
            sum += A[row * cols_A + k] * B[k * cols_B + col];
        }
        C[row * cols_B + col] = sum;
    }
}

void matrixMultiplicationGPU(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B) {
    // Allocate memory on the GPU
    float* d_A, *d_B, *d_C;
    size_t size_A = rows_A * cols_A * sizeof(float);
    size_t size_B = cols_A * cols_B * sizeof(float);
    size_t size_C = rows_A * cols_B * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy data from the host (CPU) to the device (GPU)
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for the kernel launch
    dim3 blockDim(32, 32);
    dim3 gridDim((cols_B + blockDim.x - 1) / blockDim.x, (rows_A + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel
    matrixMultiplicationKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows_A, cols_A, cols_B);

    // Copy the result back from the device (GPU) to the host (CPU)
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}