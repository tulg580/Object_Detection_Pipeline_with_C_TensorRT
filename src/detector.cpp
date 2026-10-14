#include "detector.hpp"
#include <fstream>
#include <iostream>

Detector::Detector(const std::string& engine_path) {
    // 1. .engine 파일 로드
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Engine file not found: " + engine_path);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // 2. TensorRT 엔진 인스턴스화
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    context_.reset(engine_->createExecutionContext());

    cudaStreamCreate(&stream_);
    initBuffers();
}

Detector::~Detector() {
    cudaStreamDestroy(stream_);
    cudaFreeHost(h_input_);
    cudaFreeHost(h_output_);
    cudaFree(d_input_);
    cudaFree(d_output_);
}

void Detector::initBuffers() {
    // Pinned Host Memory 할당 (PCIe 전송 속도 향상)
    cudaHostAlloc(&h_input_, input_size_, cudaHostAllocDefault);
    cudaHostAlloc(&h_output_, output_size_, cudaHostAllocDefault);

    // Device Memory 할당
    cudaMalloc(&d_input_, input_size_);
    cudaMalloc(&d_output_, output_size_);

    bindings_[0] = d_input_;
    bindings_[1] = d_output_;
}

void Detector::infer() {
    // Host -> Device 복사 (비동기)
    cudaMemcpyAsync(d_input_, h_input_, input_size_, cudaMemcpyHostToDevice, stream_);

    // 추론 수행 (enqueueV2)
    context_->enqueueV2(bindings_, stream_, nullptr);

    // Device -> Host 복사 (비동기)
    cudaMemcpyAsync(h_output_, d_output_, output_size_, cudaMemcpyDeviceToHost, stream_);

    // 동기화 완료 대기
    cudaStreamSynchronize(stream_);
}