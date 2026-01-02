#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_vector_dot_kernel(
    const float* matrix,
    const float* vector,
    float* result,
    int rows,
    int cols
) {
    // Implement the kernel to compute matrix-vector dot product
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	// Each thread computes one element of the result vector
	if (row < rows){
		float sum = 0.0f;
		for (int j=0; j<cols;j++){
			sum += matrix[row*cols+j]*vector[j];
		}
		result[row] = sum;
	}
    
}

std::vector<float> matrix_dot_vector(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {

    // Return empty vector if dimensions don't match

	int rows = matrix.size();
	if (rows==0) return std::vector<float>{-1.0f};

	int cols = matrix[0].size();

	if (cols!=vec.size()){
		return std::vector<float>{-1.0f};
	}

	std::vector<float> flat_matrix(rows*cols);
	for(int i=0;i<rows;i++){
		for (int j=0; j<cols;j++){
			flat_matrix[i*cols+j] = matrix[i][j];
		}
	}

	std::vector<float> result(rows);

    // 1. Allocate device memory
	float *d_matrix, *d_vector, *d_result;
	cudaMalloc(&d_matrix, rows * cols * sizeof(float));
	cudaMalloc(&d_vector, cols * sizeof(float));
	cudaMalloc(&d_result, cols * sizeof(float));

    // 2. Copy data to device
	cudaMemcpy(d_matrix, flat_matrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vec.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Launch kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

	matrix_vector_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_matrix, d_vector, d_result, rows, cols
    );
	
    // 4. Copy result back
	cudaMemcpy(result.data(), d_result, rows * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Free memory and return result
	cudaFree(d_matrix);
	cudaFree(d_vector);
	cudaFree(d_result);

    return result;
}