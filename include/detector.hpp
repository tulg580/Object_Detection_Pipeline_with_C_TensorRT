#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class Detector {
public:
    explicit Detector(const std::string& engine_path);
    ~Detector();

    void infer();
    float* getHostInputBuffer() { return h_input_; }
    float* getHostOutputBuffer() { return h_output_; }

private:
    void initBuffers();

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

    cudaStream_t stream_{nullptr};

    // 버퍼 포인터 (Host Pinned & Device)
    float* h_input_{nullptr};
    float* h_output_{nullptr};
    void* d_input_{nullptr};
    void* d_output_{nullptr};
    void* bindings_[2]{nullptr};

    const size_t input_size_ = 1 * 3 * 640 * 640 * sizeof(float);
    const size_t output_size_ = 1 * 84 * 8400 * sizeof(float);
};