#include "tensorrt_base.h"
#include "image_encoder.h"

ImageEncoder::ImageEncoder(const std::string& onnx_path,const std::string& engine_path) {
    buildEngine(onnx_path, engine_path);
    printModelInfo();
    std::cout << "ImageEncoder initialized" << std::endl;
}
ImageEncoder::~ImageEncoder(){
    
} 


// // Perform inference and get output buffers
void ImageEncoder::infer(const void* input_image, 
            void** output_pix_feat, 
            void** output_vision_feat, 
            void** output_vision_pos_embed) {
    TensorRTModelBase::allocateBuffers();
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(getDeviceBuffer("image"), input_image, 
                                getSizeFromDims(input_dims_[0]), cudaMemcpyHostToDevice));
    
    // Run inference
    TensorRTModelBase::infer();
    
    // Set output pointers
    *output_pix_feat = getDeviceBuffer("pix_feat");
    *output_vision_feat = getDeviceBuffer("vision_feats");
    *output_vision_pos_embed = getDeviceBuffer("vision_pos_embed");
}
