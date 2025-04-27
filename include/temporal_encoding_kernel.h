#ifndef TEMPORAL_ENCODING_KERNEL_H
#define TEMPORAL_ENCODING_KERNEL_H

#include <cuda_runtime.h>

/**
 * @brief CUDA核函数：执行位置编码与时序编码的加法 (pos_enc + temp_code)
 * 
 * @param dst 输出矩阵 [rows, cols]
 * @param pos_enc 输入位置编码 [rows, cols]
 * @param temp_code 输入时序编码 [cols] (将被广播到所有行)
 * @param rows 矩阵行数 (1024)
 * @param cols 矩阵列数 (64)
 */
__global__ void broadcastAddKernel(const float* pos_enc, 
    const float* temp_code, float* dst, size_t rows, size_t cols);

/**
 * @brief 启动核函数的封装接口
 * 
 * @param dst 设备指针，输出结果
 * @param pos_enc 设备指针，输入位置编码
 * @param temp_code 设备指针，输入时序编码
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 */
void launchAddKernel(
    const void* pos_enc,
    const void* temp_code,
    void* dst,
    int rows,
    int cols
);

#endif // TEMPORAL_ENCODING_KERNEL_H