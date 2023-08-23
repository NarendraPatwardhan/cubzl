#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <bits/types/clock_t.h>
#include <math.h>
#include "kernel.h"

// Function to initialize a matrix with random values
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)rand() / (float)RAND_MAX;
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
    float* A = (float*)malloc(rows_A * cols_A * sizeof(float));
    float* B = (float*)malloc(cols_A * cols_B * sizeof(float));
    float* C_CPU = (float*)malloc(rows_A * cols_B * sizeof(float));
    float* C_GPU = (float*)malloc(rows_A * cols_B * sizeof(float));

    // Initialize matrices A and B with random values
    srand(time(NULL)); // Seed the random number generator
    initializeMatrix(A, rows_A, cols_A);
    initializeMatrix(B, cols_A, cols_B);

    // Matrix multiplication on CPU
    clock_t start_CPU = clock();
    matrixMultiplicationCPU(A, B, C_CPU, rows_A, cols_A, cols_B);
    clock_t end_CPU = clock();
    double duration_CPU = ((double)(end_CPU - start_CPU)) / CLOCKS_PER_SEC;

    // Matrix multiplication on GPU
    clock_t start_GPU = clock();
    matrixMultiplicationGPU(A, B, C_GPU, rows_A, cols_A, cols_B);
    clock_t end_GPU = clock();
    double duration_GPU = ((double)(end_GPU - start_GPU)) / CLOCKS_PER_SEC;
    
    // Output the time taken for CPU calculation
    printf("Time taken for matrix multiplication on CPU: %f seconds\n", duration_CPU);
    printf("Time taken for matrix multiplication on GPU: %f seconds\n", duration_GPU);
    printf("Sppedup: %f\n",duration_CPU/duration_GPU);
    
    // Verification: Compare C_CPU and C_GPU to check for correctness if needed.
    float diff = 0.0f;
    for (int i = 0; i < rows_A * cols_B; ++i) {
        diff += fabs(C_CPU[i] - C_GPU[i]);
    }
    diff /= rows_A * cols_B;
    printf("Difference between CPU and GPU: %f\n", diff);
    
    // Free allocated memory
    free(A);
    free(B);
    free(C_CPU);
    free(C_GPU);

    return 0;
}

