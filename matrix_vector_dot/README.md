# Matrix-Vector Dot Product (CUDA)

[Deep-ml problem 1](https://www.deep-ml.com/problems/1)
## Features
- 1 thread per matrix row
- Row-major flattened memory
- Bounds-checked kernel
- Clean host-device memory management


## Build
```bash
nvcc matrix_vector_dot.cu -o matrix_dot

## Run
```bash
./matrix_dot
```
