#ifndef MEMORY_ATTENTION_H
#define MEMORY_ATTENTION_H

#include "tensorrt_base.h"
#include <string>
#include <fstream>
#include <sstream>

class MemoryAttention : public TensorRTModelBase {
public:
    MemoryAttention(const std::string& onnx_path,const std::string& engine_path);
    ~MemoryAttention();
    void infer(const void* vision_feat,         // [1, 256, 32, 32]
                const void* vision_pos_embed,    // [1024, 1, 256]
                const void* memory_0,            // [16, 256]
                const void* memory_1,            // [n, 64, 32, 32]
                const void* memory_pos_embed,    // [n*1024+4*m, 1, 64]
                size_t memory_0_size,
                size_t memory_1_size,
                size_t memory_pos_embed_size,
                void** output_image_embed);      // [1, 256, 32, 32]
};

#endif // MEMORY_ATTENTION_H