# Matrix Transpose (CUDA)

[Deep-ML problem 2](https://www.deep-ml.com/problems/2)

CUDA implementation of a 2D matrix transpose, converting a matrix of shape `(rows × cols)` into `(cols × rows)` using a custom GPU kernel.

## Features
- 2D grid, 1 thread per element in the matrix
- Row-major flattened memory layout
- Bounds-checked CUDA kernel
- Explicit host ↔ device memory management
- Correct reconstruction into `std::vector<std::vector<float>>`

## Build
```bash
nvcc solution.cu -o matrix_transpose
```

## Run
```bash
./matrix_transpose
```