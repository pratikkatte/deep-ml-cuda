#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void transpose_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    // Implement the kernel to transpose the matrix
    // input is rows x cols, output should be cols x rows

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        output[col * rows + row] =
            input[row * cols + col];
    }
}

std::vector<std::vector<float>> transpose_matrix(const std::vector<std::vector<float>>& matrix) {
    // 1. Allocate device memory
    int rows = matrix.size();
    int cols = matrix[0].size();

    std::vector<float> flatMatrix(rows*cols);
    for(int i=0; i<rows;i++){
        for (int j=0; j<cols; j++){
            flatMatrix[i*cols+j] = matrix[i][j];
        }
    }
    float *flatMatrix_d, *result_d;
    // 2. Copy data to device
    cudaMalloc(&flatMatrix_d, rows * cols * sizeof(float));
    cudaMalloc(&result_d, rows * cols * sizeof(float));

    cudaMemcpy(flatMatrix_d, flatMatrix.data(),
               rows * cols * sizeof(float),
               cudaMemcpyHostToDevice);

    // Kernel launch

    dim3 threads(16, 16);  // start small
    dim3 blocks(
    (cols + threads.x - 1) / threads.x,
    (rows + threads.y - 1) / threads.y
);

    transpose_kernel<<<blocks, threads>>>(
        flatMatrix_d, result_d, rows, cols
    );

    // Copy back
    std::vector<float> flatResult(rows * cols);
    cudaMemcpy(flatResult.data(), result_d,
               rows * cols * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(flatMatrix_d);
    cudaFree(result_d);

    // Reshape
    std::vector<std::vector<float>> result(
        cols, std::vector<float>(rows)
    );

    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            result[col][row] = flatResult[col * rows + row];
        }
    }

    return result;
}