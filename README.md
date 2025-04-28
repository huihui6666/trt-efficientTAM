使用TensorRT 推理 efficientTAM

支持单目标跟踪

###### 使用

```bash
#export your TensorRT path
export TensorRT_HOME=your_tensorRT_path
mkdir build && cd build
cmake .. && make
./trt-efficientTAM
```



## 参考

EfficientTAM： https://github.com/yformer/EfficientSAM

SAM2Export：https://github.com/Aimol-l/SAM2Export

onnxinference: https://github.com/Aimol-l/OrtInference
