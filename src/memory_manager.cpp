#include "memory_manager.h"

GPUMemoryManager::GPUMemoryManager() {
    cudaMalloc(&first_frame_obj_ptr_, 256 * sizeof(float));   
    std::cout << "Initializing GPUMemoryManager" << std::endl;

}

GPUMemoryManager::~GPUMemoryManager() {
    if (first_frame_obj_ptr_) cudaFree(first_frame_obj_ptr_);
    
    for (auto ptr : recent_obj_ptrs_) cudaFree(ptr);
    for (auto& mem : mem_encoder_outputs_) {
        cudaFree(mem.features);
        cudaFree(mem.pos_enc);
        cudaFree(mem.temporal_code);
    }
}

// copy obj_ptr to first_frame_obj_ptr
void GPUMemoryManager::setFirstFrameObjPtr(void* obj_ptr) {
    cudaMemcpy(first_frame_obj_ptr_, obj_ptr, 256 * sizeof(float), cudaMemcpyDeviceToDevice);
}
// copy obj_ptr to front of recent_obj_ptrs_
void GPUMemoryManager::addRecentObjPtr(void* obj_ptr) {
    void* new_ptr = nullptr;
    cudaMalloc(&new_ptr, 256 * sizeof(float));
    cudaMemcpy(new_ptr, obj_ptr, 256 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    if(recent_obj_ptrs_.size() == 15)
    {
        // If there are already 15 elements, remove the oldest one and add the new one
        void* old_ptr = recent_obj_ptrs_.front();
        if (old_ptr) {
            std::cout << "Freeing old ptr memory at " << old_ptr << std::endl;
            cudaFree(old_ptr);
            old_ptr = nullptr;
        }
        recent_obj_ptrs_.pop_front();
        
    }
    recent_obj_ptrs_.push_back(new_ptr);
    while(recent_obj_ptrs_.size() < 15){
        void* new_ptr = nullptr;
        cudaMalloc(&new_ptr, 256 * sizeof(float));
        cudaMemcpy(new_ptr, obj_ptr, 256 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        recent_obj_ptrs_.push_back(new_ptr);
    }
}

void* GPUMemoryManager::getMemory0() {
    
    void* memory0;
    cudaMalloc(&memory0, 16 * 256 * sizeof(float));
    // Copy first frame
    cudaMemcpy(memory0, first_frame_obj_ptr_, 256 * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Copy recent frames
    for (size_t i = 0; i < 15 ; ++i) {
        void* dst = (char*)memory0 + (i + 1) * 256 * sizeof(float);
        cudaMemcpy(dst, recent_obj_ptrs_[i], 256 * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    return memory0;
}

void GPUMemoryManager::addMemoryEncoderOutput(void* features, void* pos_enc, void* temp_code) {
    
    MemoryEncoderOutput output;
        
    // 1. 分配并拷贝新数据到设备内存   
    cudaMalloc(&output.features, 1 * 64 * 32 * 32 * sizeof(float));
    cudaMemcpy(output.features, features, 1 * 64 * 32 * 32 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMalloc(&output.pos_enc, 1024 * 1 * 64 * sizeof(float));
    cudaMemcpy(output.pos_enc, pos_enc, 1024 * 1 * 64 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMalloc(&output.temporal_code, 7 * 1 * 1 * 64 * sizeof(float));
    cudaMemcpy(output.temporal_code, temp_code, 7 * 1 * 1 * 64 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    // 2. 如果队列已满(7个)，先释放队头元素
    if (mem_encoder_outputs_.size() == 7) {  // 使用 >= 更安全
        MemoryEncoderOutput& old = mem_encoder_outputs_.front();  // 获取最老元素
        if (old.features){
            cudaFree(old.features);
            old.features = nullptr;
        } 
        if (old.pos_enc) {
            cudaFree(old.pos_enc);
            old.pos_enc = nullptr;
        }
        if (old.temporal_code){
            cudaFree(old.temporal_code);
            old.temporal_code = nullptr;
        } 
        mem_encoder_outputs_.pop_front();  // 移除队头  
    }
    mem_encoder_outputs_.push_back(output);
    while(mem_encoder_outputs_.size() < 7){
        MemoryEncoderOutput output;
        cudaMalloc(&output.features, 1 * 64 * 32 * 32 * sizeof(float));
        cudaMemcpy(output.features, features, 1 * 64 * 32 * 32 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMalloc(&output.pos_enc, 1024 * 1 * 64 * sizeof(float));
        cudaMemcpy(output.pos_enc, pos_enc, 1024 * 1 * 64 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMalloc(&output.temporal_code, 7 * 1 * 1 * 64 * sizeof(float));
        cudaMemcpy(output.temporal_code, temp_code, 7 * 1 * 1 * 64 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        mem_encoder_outputs_.push_front(output);
    }
   
}



void* GPUMemoryManager::getMemory1() {
    void* memory1;
    cudaMalloc(&memory1, 7 * 64 * 32 * 32 * sizeof(float));
    
    // Process each feature
    for (size_t i = 0; i < 7; ++i) {
        void* dst = (char*)memory1 + i * 64 * 32 * 32 * sizeof(float);
        cudaMemcpy(dst, mem_encoder_outputs_[i].features, 
                  64 * 32 * 32 * sizeof(float), cudaMemcpyDeviceToDevice);
                  
    }
    cudaDeviceSynchronize();
    return memory1;
    
}


void* GPUMemoryManager::getMemoryPosEmbed() {
    
    size_t n = 7; // 1-7
    size_t m = 16; // 1-16
    size_t total_size = (n * 1024 + 4 * m) * 64;

    // 分配 GPU 内存
    void* memory_pos_embed = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&memory_pos_embed, total_size * sizeof(float)));

    // 获取最后一个 temporal_code 的指针
    void* temp_code = reinterpret_cast<char*>(mem_encoder_outputs_[n - 1].temporal_code); // [7, 1, 1, 64]

    // 创建逆序索引的指针数组
    void* code_last_gpu = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&code_last_gpu, 7 * 64 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(code_last_gpu, temp_code, 7 * 64 * sizeof(float), cudaMemcpyDeviceToDevice));

    // 处理每个 mem_encoder_output
    size_t offset = 0;
    for (size_t j = 0; j < n; ++j) {
        // 获取当前帧的位置编码和时序编码
        float* pos_enc = reinterpret_cast<float*>(mem_encoder_outputs_[j].pos_enc); // [1024, 64]
        float* dst = reinterpret_cast<float*>(memory_pos_embed) + offset;

        // 获取当前 temporal_code
        const float* temp_code_gpu = reinterpret_cast<float*>(code_last_gpu) + (n - 1 - j) * 64;
        size_t rows = 1024;
        size_t cols = 64;
        launchAddKernel(  
            pos_enc, temp_code_gpu, dst, rows, cols
        );

        // 更新偏移量
        offset += rows * cols;
    }

    // 填充剩余部分为 0
    size_t remaining = total_size - (n * 1024 * 64);
    if (remaining > 0) {
        float* remaining_ptr = reinterpret_cast<float*>(memory_pos_embed) + offset;
        CHECK_CUDA_ERROR(cudaMemset(remaining_ptr, 0, remaining * sizeof(float)));
    }

    // 释放临时分配的 GPU 内存
    CHECK_CUDA_ERROR(cudaFree(code_last_gpu));

    return memory_pos_embed;
}
