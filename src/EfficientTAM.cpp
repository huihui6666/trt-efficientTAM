#include "EfficientTAM.h"

 
EfficientTAM::EfficientTAM(const std::string& model_dir) 
{   
    image_encoder_ = std::make_shared<ImageEncoder>(model_dir + "/image_encoder.onnx",
        model_dir + "/image_encoder.engine");
    std::cout<<"image_encoder engine done!"<<std::endl;
    memory_attention_ = std::make_shared<MemoryAttention>(model_dir + "/memory_attention.onnx",
            model_dir + "/memory_attention.engine");
    std::cout<<"memory attention engine done!"<<std::endl;
    memory_encoder_ = std::make_shared<MemoryEncoder>(model_dir + "/memory_encoder.onnx",
            model_dir + "/memory_encoder.engine");
    std::cout<<"memory_encoder engine done!"<<std::endl;
    
    image_decoder_ = std::make_shared<ImageDecoder>(model_dir + "/image_decoder.onnx",
            model_dir + "/image_decoder.engine");
    std::cout<<"image_decoder engine done!"<<std::endl;
    
    memory_manager_ = std::make_shared<GPUMemoryManager>();
}

EfficientTAM::~EfficientTAM()
{
}
void EfficientTAM::inference(const void* input_image,
                                  const void* point_coords,
                                  const void* point_labels,
                                  void** pred_mask,int frame_idx) {
    using namespace std::chrono;
    // 1. Run image encoder  
    void *pix_feat=nullptr, *vision_feat=nullptr, *vision_pos_embed=nullptr;
    image_encoder_->infer(input_image, &pix_feat, &vision_feat, &vision_pos_embed);
    cudaDeviceSynchronize();
    std::cout<<"image encoder done"<<std::endl;
    
    void *image_embed = nullptr;
    if (memory_manager_->isFirstFrame()) {
        // First frame uses vision_feat directly
        image_embed = vision_feat;
        cudaDeviceSynchronize();
    } else {
        // Subsequent frames use memory attention
        void *memory0 = memory_manager_->getMemory0();
        //size_t memory0_size = 16*256;
        cudaDeviceSynchronize();
        
        void *memory1 = memory_manager_->getMemory1();
        
        void *memory_pos_embed = memory_manager_->getMemoryPosEmbed();
        //size_t memory_pos_embed_size = 7232*1*64;
        cudaDeviceSynchronize();
        
        memory_attention_->infer(vision_feat, vision_pos_embed,
                               memory0, memory1, memory_pos_embed, 16, 7, 7232,
                               &image_embed);
        cudaDeviceSynchronize();  
    }
    std::cout<<"memory attention done!"<<std::endl;
    // 2. Run image decoder
    
    void *obj_ptr=nullptr, *mask_for_mem=nullptr;
    image_decoder_->infer(point_coords, point_labels, 
                         image_embed, &obj_ptr, &mask_for_mem, pred_mask);
                    
    cudaDeviceSynchronize();
    
    // 3. Store obj_ptr
    if (memory_manager_->isFirstFrame()) {
        memory_manager_->setFirstFrameObjPtr(obj_ptr);
        cudaDeviceSynchronize();
        
    } else {
        
        memory_manager_->addRecentObjPtr(obj_ptr);
        cudaDeviceSynchronize();
    }
   
    // 4. Run memory encoder (except first frame)
    
    void *maskmem_features=nullptr, *maskmem_pos_enc=nullptr, *temporal_code=nullptr;
    memory_encoder_->infer(mask_for_mem, pix_feat,
                            &maskmem_features, &maskmem_pos_enc, &temporal_code);
    
    memory_manager_->addMemoryEncoderOutput(maskmem_features, maskmem_pos_enc, temporal_code);
    cudaDeviceSynchronize();

    //std::cout<<"memory encoder done!"<<std::endl;
    memory_manager_->incrementFrame();
    cudaDeviceSynchronize();
    
    
}