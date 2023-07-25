#include <iostream>
#include <chrono>
#include "kernel.h"

// Function to initialize a matrix with random values
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void matrixMultiplicationCPU(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B) {
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cols_A; ++k) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            C[i * cols_B + j] = sum;
        }
    }
}

int main() {
    const int rows_A = 512; // Adjust the size of the matrices as needed
    const int cols_A = 1024;
    const int cols_B = 1024;

    // Allocate memory for the matrices on the CPU
    float* A = new float[rows_A * cols_A];
    float* B = new float[cols_A * cols_B];
    float* C_CPU = new float[rows_A * cols_B];
    float* C_GPU = new float[rows_A * cols_B];

    // Initialize matrices A and B with random values
    initializeMatrix(A, rows_A, cols_A);
    initializeMatrix(B, cols_A, cols_B);

    // Matrix multiplication on CPU
    auto start_CPU = std::chrono::high_resolution_clock::now();
    matrixMultiplicationCPU(A, B, C_CPU, rows_A, cols_A, cols_B);
    auto end_CPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_CPU = end_CPU - start_CPU;

    // Matrix multiplication on GPU
    auto start_GPU = std::chrono::high_resolution_clock::now();
    matrixMultiplicationGPU(A, B, C_GPU, rows_A, cols_A, cols_B);
    auto end_GPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_GPU = end_GPU - start_GPU;

    // Output the time taken for both CPU and GPU calculations
    std::cout << "Time taken for matrix multiplication on CPU: " << duration_CPU.count() << " seconds\n";
    std::cout << "Time taken for matrix multiplication on GPU: " << duration_GPU.count() << " seconds\n";
    std::cout << "Speedup: " << duration_CPU.count() / duration_GPU.count() << "x\n";

    // Verification: Compare C_CPU and C_GPU to check for correctness if needed.
    auto diff = 0.0f;
    for (int i = 0; i < rows_A * cols_B; ++i) {
        diff += std::abs(C_CPU[i] - C_GPU[i]);
    }
    diff /= rows_A * cols_B;
    std::cout << "Difference between CPU and GPU: " << diff << std::endl;

    // Free allocated memory
    delete[] A;
    delete[] B;
    delete[] C_CPU;
    delete[] C_GPU;

    return 0;
}