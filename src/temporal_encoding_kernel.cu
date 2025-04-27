#include "temporal_encoding_kernel.h"
#include <cuda_runtime.h>


// CUDA 核函数：广播相加
__global__ void broadcastAddKernel(const float* pos_enc, const float* temp_code, float* dst, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for (size_t col = 0; col < cols; ++col) {
            dst[row * cols + col] = pos_enc[row * cols + col] + temp_code[col];
        }
    }
}


void launchAddKernel(  
    const void* pos_enc,
    const void* temp_code,
    void* dst,
    int rows,
    int cols
) {
    size_t threads_per_block = 256;
    size_t blocks = (rows + threads_per_block - 1) / threads_per_block;
    // 调用核函数
    broadcastAddKernel<<<blocks, threads_per_block>>>(
        static_cast<const float*>(pos_enc),
        static_cast<const float*>(temp_code),
        static_cast<float*>(dst),
        rows,
        cols
    );

}
