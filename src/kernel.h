#ifndef KERNEL_H
#define KERNEL_H
#ifdef __cplusplus
extern "C" {
#endif // CPP

void matrixMultiplicationGPU(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B);

#ifdef __cplusplus
}
#endif // DEBUG
#endif  // KERNEL_H
