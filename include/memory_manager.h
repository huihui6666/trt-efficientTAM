#pragma once
#include <deque>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <chrono>
#include <thread>
#include <cuda_runtime_api.h>
#include "tensorrt_base.h"
#include "temporal_encoding_kernel.h"

class GPUMemoryManager {
private:
    struct MemoryEncoderOutput {
        void* features = nullptr;     // [1,64,32,32]
        void* pos_enc = nullptr ;      // [1024,1,64]
        void* temporal_code = nullptr;// [7,1,1,64]
    };

    // History storage
    void* first_frame_obj_ptr_ = nullptr;
    std::deque<void*> recent_obj_ptrs_;  // Max 15 frames
    std::deque<MemoryEncoderOutput> mem_encoder_outputs_;  // Max 7 frames
    int current_frame_ = 0;

public:
    GPUMemoryManager();
    ~GPUMemoryManager();

    // Frame management
    bool isFirstFrame() const { return current_frame_ == 0; }
    void incrementFrame() { current_frame_++; }
    int getCurrentFrame() const { return current_frame_; }

    // Object pointer management
    void setFirstFrameObjPtr(void* obj_ptr);
    void addRecentObjPtr(void* obj_ptr);
    void* getMemory0();  // [1-16, 256]

    // Memory encoder outputs
    void addMemoryEncoderOutput(void* features, void* pos_enc, void* temp_code);
    void* getMemory1();  // [1-7, 64, 32, 32]
    void* getMemoryPosEmbed();  // [n*1024+4*m, 1, 64]

};