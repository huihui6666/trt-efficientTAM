#include "image_decoder.h"

ImageDecoder::ImageDecoder(const std::string& onnx_path,const std::string& engine_path) {
    buildEngine(onnx_path,engine_path);
    printModelInfo();
    }
ImageDecoder::~ImageDecoder(){
    
}
        
void ImageDecoder::infer(const void* point_coords, 
                    const void* point_label, 
                    const void* image_embed,
                    void** output_obj_ptr,
                    void** output_mask_for_mem,
                    void** output_pred_mask) {
    TensorRTModelBase::allocateBuffers();
    // Copy inputs to device
    CHECK_CUDA_ERROR(cudaMemcpy(getDeviceBuffer("point_coords"), point_coords, 
                                getSizeFromDims(input_dims_[0]), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(getDeviceBuffer("point_labels"), point_label, 
                                getSizeFromDims(input_dims_[1]), cudaMemcpyHostToDevice));
    
    // image_embed is already on device (from memory_attention output)
    CHECK_CUDA_ERROR(cudaMemcpy(getDeviceBuffer("image_embed"), image_embed, 
                                getSizeFromDims(input_dims_[2]), cudaMemcpyDeviceToDevice));
                                cudaDeviceSynchronize();
    
    // Run inference
    TensorRTModelBase::infer();
    cudaDeviceSynchronize();
    
    // Set output pointers
    *output_obj_ptr = getDeviceBuffer("obj_ptr");
    *output_mask_for_mem = getDeviceBuffer("mask_for_mem");
    *output_pred_mask = getDeviceBuffer("pred_mask");
}
