#ifndef IMAGE_ENCODER_H
#define IMAGE_ENCODER_H

#include "tensorrt_base.h"
#include <string>

class ImageEncoder : public TensorRTModelBase {
public:
    ImageEncoder(const std::string& onnx_path,const std::string& engine_path);
    ~ImageEncoder();    
    void infer(const void* input_image,  //[1, 3, 512, 512]
               void** output_pix_feat,  // [1024, 1, 256]
               void** output_vision_feat, // [1, 256, 32, 32]
               void** output_vision_pos_embed);  // [1024, 1, 256]
};

#endif // IMAGE_ENCODER_H