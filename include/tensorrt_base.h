#ifndef TENSORRT_MODEL_BASE_H
#define TENSORRT_MODEL_BASE_H

#include <NvInfer.h>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <filesystem>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

// Utility function to check CUDA errors
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

class TensorRTModelBase {
public:
    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<void*> device_buffers_;
    std::vector<void*> host_buffers_;
    std::vector<nvinfer1::Dims> input_dims_;
    std::vector<nvinfer1::Dims> output_dims_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

public:
// 默认构造函数
    TensorRTModelBase() = default;
    // 带参数的构造函数
    TensorRTModelBase(const std::string& onnx_path, const std::string& engine_path);
     ~TensorRTModelBase();
    
    void buildEngine(const std::string& onnx_path, const std::string& engine_path);
    void buildEngineFromONNX(const std::string& onnx_path, const std::string& engine_path);
    void saveEngine(const std::string& engine_path);
    void loadEngine(const std::string& engine_path);
    void initialize();
    void allocateBuffers();
    static size_t getSizeFromDims(const nvinfer1::Dims& dims);
    void* getDeviceBuffer(const std::string& tensor_name);
    virtual void infer();
    void printModelInfo();
};

#endif // TENSORRT_MODEL_BASE_H