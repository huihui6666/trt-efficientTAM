#pragma once
#include "image_encoder.h"
#include "image_decoder.h"
#include "memory_encoder.h"
#include "memory_attention.h"
#include "memory_manager.h"
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

class EfficientTAM {
private:
    std::shared_ptr<ImageEncoder> image_encoder_;
    std::shared_ptr<ImageDecoder> image_decoder_;
    std::shared_ptr<MemoryEncoder> memory_encoder_;
    std::shared_ptr<MemoryAttention> memory_attention_;
    std::shared_ptr<GPUMemoryManager> memory_manager_;

public:
    EfficientTAM(const std::string& model_dir);
    ~EfficientTAM();
    void inference(const void* input_image,    // [1, 3, 512, 512]
                    const void* point_coords,    // [1, 2, 2]
                    const void* point_labels,     // [1, 2]
                    void** pred_mask,int frame_idx);           // [1, 1, 512, 512]
};