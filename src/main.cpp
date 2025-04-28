#include "EfficientTAM.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <NvInfer.h>
#include <fstream>

// 自定义 Logger 类
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 根据 severity 输出日志
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
};
std::vector<float> preprocessCpu(cv::Mat& image)
{
    // std::vector<cv::Mat> mats{image};
    // cv::Mat blob = cv::dnn::blobFromImages(mats, 1/255.0,cv::Size(512,512), cv::Scalar(0, 0, 0), true, false);
    // std::vector<float> input_tensor(blob.total());
    // std::memcpy(input_tensor.data(), blob.ptr<float>(), blob.total() * sizeof(float));
    cv::Mat image_resize;
    cv::resize(image, image_resize, cv::Size(512, 512));
    std::vector<float> input_tensor;
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < image_resize.rows; i++)
        {
            for (int j = 0; j < image_resize.cols; j++)
            {
                input_tensor.emplace_back(((float)image_resize.at<cv::Vec3b>(i, j)[k]  / 255.0f));
            }
        }
    }
    return input_tensor;
}
void processMask(void* pred_mask_gpu, cv::Mat& mask_display, int width, int height) {
    // 从GPU拷贝数据到CPU
    cv::Mat mask_float(128, 128, CV_32FC1);
    cudaMemcpy(mask_float.data, pred_mask_gpu, 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost);
    // 调整大小到原始帧大小
    cv::Mat resized_mask_float;
    cv::resize(mask_float, resized_mask_float, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    //std::cout << "Resized mask to " << width << "x" << height << std::endl;
    // 转换为8位灰度图像
    cv::Mat mask_8bit;
    resized_mask_float.convertTo(mask_8bit, CV_8UC1, 255);
    
    // 二值化处理
    cv::threshold(mask_8bit, mask_display, 0, 255, cv::THRESH_BINARY);
    
    // 形态学操作
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask_display, mask_display, cv::MORPH_OPEN, element);
}
int main() {
    const std::string model_dir = "~/trt-efficientTAM/models/etam_fp16";
    
    // 创建一个日志记录器实例
    Logger logger;
    
    // 初始化推理管道
    auto efficientTAM = std::make_unique<EfficientTAM>(model_dir);
    
    // 设置提示信息（点坐标和标签）
    std::vector<float> point_coords;  // [x, y]
    std::vector<float> point_labels;      // 前景点
    
    // 加载视频文件
    std::string video_path = "../demo_video/01_dog.mp4";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open video file " << video_path << std::endl;
        return -1;
    }
    
    // 打印视频信息
    std::cout << "视频宽度: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "视频高度: " << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "视频帧率: " << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "总帧数: " << capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    // 视频分辨率
    int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::vector<float> frame_size = {frame_width, frame_height}; 
    point_coords = {420.0 / frame_width * 512,380.0f/ frame_height * 512,660.0f / frame_width * 512,600.0f/ frame_height * 512}; //x1y1x2y2
    std::cout<<"point_coords: "<<point_coords[0]<<" "<<point_coords[1]<<std::endl;
    cv::Mat frame;
    size_t frame_idx = 0;
    void* pred_mask = nullptr;

    cv::VideoWriter video_writer;
    // 保存视频
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // 或使用其他编码器，如'X','2','6','4'
    double fps = 30.0; // 根据实际情况调整
    cv::Size frame_size_(frame_width, frame_height); // 确保与输入帧尺寸一致
    std::string output_filename = "output_video.avi";
    video_writer.open(output_filename, fourcc, fps, frame_size_);

    while (true) {
        // 读取视频帧
        if (!capture.read(frame) || frame.empty()) {
            break;
        }
        std::cout << "start infer: " << frame_idx << std::endl;
        // 调整帧大小并预处理（假设需要512x512输入）
        std::vector<float> input = preprocessCpu(frame);
        point_labels = {1};
         if(frame_idx==0){
            point_labels = {2,3};
        }else{
            point_labels = {-1,-1};
        }
        // 执行推理
        auto start = std::chrono::high_resolution_clock::now();
        for(auto coord : point_coords){
            std::cout<<"coord: "<<coord<<std::endl;
        }
        std::cout<<"point_labels: "<<point_labels[0]<<std::endl;
        
        efficientTAM->inference(input.data(),
                            point_coords.data(),
                            point_labels.data(),
                            &pred_mask,frame_idx);

        // 处理mask并找到最佳轮廓
        cv::Mat mask_display;
        processMask(pred_mask, mask_display, frame_width, frame_height);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask_display.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);    
        
        // 更新点坐标和显示
        if (contours.size() >= 0) {
            for(int i=0 ; i < contours.size(); i++)
            // 在原始帧上绘制结果
            cv::drawContours(frame, contours,i ,cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
        
        // 显示性能信息
        std::ostringstream oss;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout<<"Inference time: "<<duration<<"ms"<<std::endl;
        oss << "Frame: " << frame_idx << " FPS: " << 1000.0f / duration;
        cv::putText(frame, oss.str(), cv::Point(30, 40), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        // 显示结果
        cv::imshow("Original Frame", frame);
        //cv::imshow("Segmentation Mask", mask_display);
    
        video_writer.write(frame);
        // 按键控制
        int key = cv::waitKey(5);
        if (key == 'q' || key == 27) {  // ESC或'q'退出
            break;
        }
        // if(frame_idx==5){
        //     break;
        // }
        
        frame_idx++;
    }
    
    // 释放资源
    capture.release();
    //cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}
