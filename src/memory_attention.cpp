#include "memory_attention.h"

MemoryAttention::MemoryAttention(const std::string& onnx_path,const std::string& engine_path) {
    buildEngine(onnx_path, engine_path);
    printModelInfo();
}
MemoryAttention::~MemoryAttention(){

}      
// Perform inference
void MemoryAttention::infer(const void* vision_feat, 
            const void* vision_pos_embed, 
            const void* memory_0, 
            const void* memory_1,
            const void* memory_pos_embed,
            size_t memory_0_size,
            size_t memory_1_size,
            size_t memory_pos_embed_size,
            void** output_image_embed) {
    // 获取设备缓冲区并验证
    TensorRTModelBase::allocateBuffers();
    void* vision_feats_buf = getDeviceBuffer("vision_feats");
    void* vision_pos_embed_buf = getDeviceBuffer("vision_pos_embed");
    void* memory_0_buf = getDeviceBuffer("memory_0");
    void* memory_1_buf = getDeviceBuffer("memory_1");
    void* memory_pos_embed_buf = getDeviceBuffer("memory_pos_embed");
    
    if (!vision_feats_buf || !vision_pos_embed_buf || !memory_0_buf || 
        !memory_1_buf || !memory_pos_embed_buf) {
        throw std::runtime_error("One or more device buffers are not allocated");
    }

    // 计算各输入的实际字节大小
    size_t vision_feat_bytes = getSizeFromDims(input_dims_[0]);
    size_t vision_pos_embed_bytes = getSizeFromDims(input_dims_[1]);
    size_t memory_0_bytes = 16 * 256 * sizeof(float);
    size_t memory_1_bytes = 7 * 64 * 32 * 32 * sizeof(float);
    size_t memory_pos_embed_bytes = 7232 * 64 * sizeof(float);

    // 拷贝数据到设备
    
    CHECK_CUDA_ERROR(cudaMemcpy(vision_feats_buf, vision_feat, vision_feat_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(vision_pos_embed_buf, vision_pos_embed, vision_pos_embed_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(memory_0_buf, memory_0, memory_0_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(memory_1_buf, memory_1, memory_1_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(memory_pos_embed_buf, memory_pos_embed, memory_pos_embed_bytes, cudaMemcpyDeviceToDevice));
    
    // 确保所有拷贝完成
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 运行推理
    TensorRTModelBase::infer();
    
    // 设置输出指针
    *output_image_embed = getDeviceBuffer("image_embed");
    if (*output_image_embed == nullptr) {
        throw std::runtime_error("Output buffer image_embed is not allocated");
    }
}
