#include "memory_encoder.h"

MemoryEncoder::MemoryEncoder(const std::string& onnx_path, const std::string& engine_path) {
    buildEngine(onnx_path, engine_path);
    printModelInfo();
    
}
MemoryEncoder::~MemoryEncoder(){

}       
// Perform inference
void MemoryEncoder::infer(const void* mask_for_mem, 
            const void* pix_feat,
            void** output_maskmem_features,
            void** output_maskmem_pos_enc,
            void** output_temporal_code) {
    TensorRTModelBase::allocateBuffers();
    // mask_for_mem is already on device (from image_decoder output)
    CHECK_CUDA_ERROR(cudaMemcpy(getDeviceBuffer("mask_for_mem"), mask_for_mem, 
                                getSizeFromDims(input_dims_[0]), cudaMemcpyDeviceToDevice));
                                cudaDeviceSynchronize();
    // pix_feat is already on device (from image_encoder output)
    CHECK_CUDA_ERROR(cudaMemcpy(getDeviceBuffer("pix_feat"), pix_feat, 
                                getSizeFromDims(input_dims_[1]), cudaMemcpyDeviceToDevice));
                                cudaDeviceSynchronize();
    // Run inference
    TensorRTModelBase::infer();
    
    // Set output pointers
    *output_maskmem_features = getDeviceBuffer("maskmem_features");
    cudaDeviceSynchronize();
    *output_maskmem_pos_enc = getDeviceBuffer("maskmem_pos_enc");
    cudaDeviceSynchronize();
    *output_temporal_code = getDeviceBuffer("temporal_code");
    cudaDeviceSynchronize();
}
