
#include <tensorrt_base.h>

namespace fs = std::filesystem;

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;



// 带参数的构造函数
TensorRTModelBase::TensorRTModelBase(const std::string& onnx_path, const std::string& engine_path) {
    try {
        buildEngine(onnx_path, engine_path);
        std::cout << "Engine built/loaded successfully" << std::endl;
        // Check if this is a memory_attention model
    } catch (const std::exception& e) {
        std::cerr << "Exception in TensorRTModelBase constructor: " << e.what() << std::endl;
        throw;
    }
}

TensorRTModelBase:: ~TensorRTModelBase() {
    // First destroy context
    context_.reset();
    // Then destroy engine
    engine_.reset();
        // Free device buffers
        for (auto& buf : device_buffers_) {
            if (buf) {
                cudaFree(buf);
                buf = nullptr;
            }
        }   
        // Free host buffers
        for (auto& buf : host_buffers_) {
            if (buf) {
                free(buf);
                buf = nullptr;
            }
        }
    }
    
// Load or build TensorRT engine from ONNX
void TensorRTModelBase::buildEngine(const std::string& onnx_path, const std::string& engine_path) {
    if (fs::exists(engine_path)) {
        std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;
        loadEngine(engine_path);
        std::cout << "Engine loaded successfully" << std::endl;
    } else {
        buildEngineFromONNX(onnx_path, engine_path);
    }
}

// Build TensorRT engine from ONNX model
void TensorRTModelBase:: buildEngineFromONNX(const std::string& onnx_path, const std::string& engine_path) {
    std::cout << "Building TensorRT engine from ONNX: " << onnx_path << std::endl;
    auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        throw std::runtime_error("Failed to create TensorRT builder");
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        throw std::runtime_error("Failed to create TensorRT network");
    }
    auto parser = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX file");
    }
   
    auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1 << 30); // 1GB
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // Set dynamic batch size
    auto profile = builder->createOptimizationProfile();
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();
        std::cout << "Input " << i << " (" << input->getName() << "): ";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j] << (j < dims.nbDims - 1 ? "x" : "");
        }
        std::cout << std::endl;
        
        if (dims.d[0] == -1) {  // Check if the first dimension is dynamic
            std::cout<<"set dynamic dimension"<<std::endl;
            dims.d[0] = 1;  // Set the minimum batch size to 1
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
            dims.d[0] = 1;  // Set the optimal batch size to 1
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
            dims.d[0] = 1;  // Set the maximum batch size to 1
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
        }
    }
    config->addOptimizationProfile(profile);
    
    // Build engine
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    
    if (!engine_) {
        throw std::runtime_error("Failed to build TensorRT engine");
    }
    
    // Save engine
    saveEngine(engine_path);
    std::cout<<"Engine save"<<std::endl;
    // Initialize after building
    initialize();

    
}

// Save TensorRT engine to file
void TensorRTModelBase:: saveEngine(const std::string& engine_path) {
    std::cout << "Saving TensorRT engine to: " << engine_path << std::endl;
    auto serialized_engine = std::shared_ptr<nvinfer1::IHostMemory>(engine_->serialize());
    std::ofstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        throw std::runtime_error("Failed to open engine file for writing");
    }
    engine_file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    engine_file.close();
}
    
// Load TensorRT engine from file
void TensorRTModelBase::loadEngine(const std::string& engine_path) {
    try {
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file.is_open()) {
            throw std::runtime_error("Failed to open engine file");
        }
        
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        
        if (engine_size == 0) {
            throw std::runtime_error("Engine file is empty");
        }
        
        std::vector<char> engine_data(engine_size);
        engine_file.read(engine_data.data(), engine_size);
        engine_file.close();
        
        runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_data.data(), engine_size));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine");
        }
        
        initialize();
        std::cout << "Engine initialized successfully" << std::endl;
        // Check if this is a memory_attention model
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading engine: " << e.what() << std::endl;
        throw; // rethrow the exception
    }

}
    
// Initialize after engine is built or loaded
void TensorRTModelBase::initialize() {
    // 添加检查
    if (!engine_) {
        throw std::runtime_error("Engine not initialized");
    }
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }
    
    // Get input/output dimensions and names
    input_dims_.clear();
    output_dims_.clear();
    input_names_.clear();
    output_names_.clear();
    
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = context_->getTensorShape(name);
        
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_dims_.push_back(dims);
            input_names_.push_back(name);
        } else {
            output_dims_.push_back(dims);
            output_names_.push_back(name);
        }
    }

    std::cout<<"Initialized"<<std::endl;
}
    
// Allocate device and host buffers based on engine requirements
void TensorRTModelBase:: allocateBuffers() {
    // Free existing buffers if any
    for (auto& buf : device_buffers_) {
        if (buf) {
            cudaFree(buf);
            buf = nullptr;
        }
    }
    for (auto& buf : host_buffers_) {
        if (buf){
            free(buf);
            buf = nullptr;
        } 
    }
    
    device_buffers_.clear();
    host_buffers_.clear();
    
    // 2. 预分配空间减少vector重分配
    device_buffers_.reserve(engine_->getNbIOTensors());
    host_buffers_.reserve(engine_->getNbIOTensors());

    // 3. 为每个tensor分配内存
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = context_->getTensorShape(name);
        if (dims.d[0] == -1) {
            dims.d[0] = 1;   // 默认MAX维度  
        }
        if (dims.d[1] == -1) {
            dims.d[1] = 2;   // 默认MAX维度
        }
        std::cout << "Tensor: " << name << ", Dimensions: ";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j] << (j < dims.nbDims - 1 ? "x" : "");
        }
        // 计算内存大小
        size_t size = getSizeFromDims(dims);
        std::cout << ", Size: " << size << " bytes" << std::endl;
        if (size == 0) {
            throw std::runtime_error("Zero-sized tensor: " + std::string(name));
        }

        // 分配设备内存
        void* device_buf = nullptr;
       
        CHECK_CUDA_ERROR(cudaMalloc(&device_buf, size));
        CHECK_CUDA_ERROR(cudaMemset(device_buf, 0, size));
        device_buffers_.push_back(device_buf);

        // 为输出分配对齐的主机内存
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            void* host_buf = malloc(size);
            if (!host_buf) {
                throw std::runtime_error("Failed to allocate aligned host memory");
            }
            host_buffers_.push_back(host_buf);
        } else {
            host_buffers_.push_back(nullptr);
        }
    }
    std::cout<<"Allocated"<<std::endl;
}
   
// Utility function to calculate size from dimensions
size_t TensorRTModelBase:: getSizeFromDims(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size * sizeof(float); // Assuming float data type
}
    
    // Get device buffer for a tensor by name
void* TensorRTModelBase:: getDeviceBuffer(const std::string& tensor_name) {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        if (std::string(engine_->getIOTensorName(i)) == tensor_name) {
            return device_buffers_[i];
        }
    }
    throw std::runtime_error("Tensor not found: " + tensor_name);
}
    
    // Inference function to be implemented by derived classes
void TensorRTModelBase:: infer(){
    if (!engine_) {
        throw std::runtime_error("Engine is not initialized");
    }
    if (!context_) {
        throw std::runtime_error("Execution context is not initialized");
    }

    // Set input/output tensor shapes in context
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        std::cout << "Setting tensor address for: " << name << std::endl;
        if (device_buffers_[i] == nullptr) {
            throw std::runtime_error("Device buffer for tensor " + std::string(name) + " is not allocated");
        }
        context_->setTensorAddress(name, device_buffers_[i]);

    }
    context_->executeV2(device_buffers_.data());
}

    
    // Utility function to print model info
void TensorRTModelBase::printModelInfo() {
    std::cout << "Model Information:" << std::endl;
    std::cout << "Inputs:" << std::endl;
    for (size_t i = 0; i < input_names_.size(); ++i) {
        std::cout << "  " << input_names_[i] << ": ";
        for (int j = 0; j < input_dims_[i].nbDims; ++j) {
            std::cout << input_dims_[i].d[j];
            if (j < input_dims_[i].nbDims - 1) std::cout << "x";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Outputs:" << std::endl;
    for (size_t i = 0; i < output_names_.size(); ++i) {
        std::cout << "  " << output_names_[i] << ": ";
        for (int j = 0; j < output_dims_[i].nbDims; ++j) {
            std::cout << output_dims_[i].d[j];
            if (j < output_dims_[i].nbDims - 1) std::cout << "x";
        }
        std::cout << std::endl;
    }
}
