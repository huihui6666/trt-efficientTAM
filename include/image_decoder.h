#ifndef IMAGE_DECODER_H
#define IMAGE_DECODER_H

#include "tensorrt_base.h"
#include <string>

class ImageDecoder : public TensorRTModelBase {
public:
    ImageDecoder(const std::string& onnx_path,const std::string& engine_path);
    ~ImageDecoder();
    void infer(const void* point_coords,    // [1, 2, 2]
               const void* point_label,     // [1, 2]
               const void* image_embed,     // [1, 256, 32, 32]
               void** output_obj_ptr,       // [1, 256]
               void** output_mask_for_mem,  // [1, 1, 512, 512]
               void** output_pred_mask);    // [1, 1, 512, 512]
};

#endif // IMAGE_DECODER_H