#ifndef MEMORY_ENCODER_H
#define MEMORY_ENCODER_H

#include "tensorrt_base.h"
#include <string>

class MemoryEncoder : public TensorRTModelBase {
public:
    MemoryEncoder(const std::string& onnx_path,const std::string& engine_path);
    ~MemoryEncoder();
    void infer(const void* mask_for_mem,        // [1, 1, 512, 512]
               const void* pix_feat,            // [1024, 1, 256]
               void** output_maskmem_features,  // [1, 64, 32, 32]
               void** output_maskmem_pos_enc,   // [1024, 1, 64]
               void** output_temporal_code);    // [7, 1, 1, 64]
};

#endif // MEMORY_ENCODER_H