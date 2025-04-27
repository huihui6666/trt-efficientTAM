cmake_minimum_required(VERSION 3.14)
project(trt_efficientTAM)

add_compile_options(-std=c++17)
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# 强制指定编译器
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)


# 查找依赖
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc dnn)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  set(TensorRT_LIBRARIES
    /usr/lib/aarch64-linux-gnu/libnvinfer.so
    /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
    /usr/lib/aarch64-linux-gnu/libnvonnxparser.so
    usr/lib/aarch64-linux-gnu/libnvparsers.so
    /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource.so.8.6.2
    "/usr/local/cuda/targets/aarch64-linux/lib/libcublasLt.so"
  )


else()
  message("embed_platform off")
  set(TensorRT_ROOT_DIRS $ENV{TensorRT_HOME})
  set(TensorRT_LIBRARIES
    ${TensorRT_ROOT_DIRS}/lib/libnvinfer.so
    ${TensorRT_ROOT_DIRS}/lib/libnvinfer_plugin.so
    ${TensorRT_ROOT_DIRS}/lib/libnvonnxparser.so
    ${TensorRT_ROOT_DIRS}/lib/libnvparsers.so
    ${TensorRT_ROOT_DIRS}/lib/libnvinfer_builder_resource.so.8.6.1
    #${TensorRT_ROOT_DIRS}/lib/libnvinfer_builder_resource.so.10.5.0
    "/usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so"
  )

endif()

# 可执行文件
add_executable(trt_efficientTAM
  src/main.cpp
  src/tensorrt_base.cpp
  src/EfficientTAM.cpp
  src/image_decoder.cpp
  src/image_encoder.cpp
  src/memory_attention.cpp
  src/memory_encoder.cpp
  src/memory_manager.cpp
  src/temporal_encoding_kernel.cu
)


# 包含目录
target_include_directories(trt_efficientTAM PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${TensorRT_ROOT_DIRS}/include
  ${OpenCV_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(trt_efficientTAM PRIVATE
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${TensorRT_LIBRARIES}
  CUDA::cudart
)