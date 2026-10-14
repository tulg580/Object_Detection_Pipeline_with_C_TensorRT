# 🚀 Real-Time Object Detection Pipeline with C++ & TensorRT

[![C++](https://img.shields.io/badge/C++-17-blue.svg?logo=c%2B%2B)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg?logo=opencv)](https://opencv.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-brightgreen.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.x%2F10.x-76B900.svg?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![CMake](https://img.shields.io/badge/CMake-3.20+-red.svg?logo=cmake)](https://cmake.org/)

> **PyTorch 기반 객체 인식 모델(YOLO)을 TensorRT 엔진으로 변환하고, C++ OpenCV 기반의 전·후처리 및 CUDA 비동기 파이프라인을 구축하여 실시간 초저지연(Ultra-Low Latency) 추론을 달성한 프로젝트입니다.**

---

## 📌 Project Overview (개요)

* **배경 및 문제 정의**:
  * Python 기반의 추론 환경(PyTorch/Python-OpenCV)은 높은 개발 편의성을 제공하지만, GIL(Global Interpreter Lock), 인터프리터 오버헤드, Host-Device 간 잦은 데이터 복사로 인해 임베디드 및 엣지 디바이스 환경에서 프레임 저하와 지연(Latency)이 발생합니다.
* **프로젝트 목표**:
  1. Python 추론 환경 대비 **지연 시간 70% 이상 단축 및 Throughput(FPS) 3배 이상 향상**
  2. OpenCV C++ API를 활용한 최적화된 Pre-processing(Letterbox, Normalization) 및 Post-processing(NMS) 모듈 자체 구현
  3. CUDA Stream과 Pinned Memory(Page-locked)를 활용한 비동기 병렬 파이프라인 구축

---

## 🎬 Demo

<!-- 저장소 루트의 assets/demo.gif 경로에 실제 시연 영상을 첨부하세요 -->
![Demo GIF](./assets/demo.gif)

---

## 📊 Benchmark & Performance (성능 측정 결과)

* **테스트 하드웨어**: Ubuntu 22.04 LTS, Intel i7-12700 CPU, NVIDIA GeForce RTX 3060 (12GB)
* **소프트웨어 스택**: CUDA 12.2, cuDNN 8.9, TensorRT 8.6, OpenCV 4.8.0
* **평가 모델**: YOLOv8n (Input: 640x640, Test Video: 1080p 60FPS Video Stream)

| 파이프라인 구성 | 런타임 엔진 | 정밀도 (Precision) | 전처리 (Pre) | 순수 추론 (Inference) | 후처리 (Post) | End-to-End 지연시간 | 처리량 (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Python Baseline | PyTorch (CUDA) | FP32 | 3.2 ms | 18.5 ms | 6.7 ms | 28.4 ms | ~35.2 FPS |
| C++ Intermediate | ONNXRuntime | FP32 | 2.1 ms | 10.2 ms | 1.9 ms | 14.2 ms | ~70.4 FPS |
| **Proposed (Ours)** | **TensorRT** | **FP32** | **1.3 ms** | **4.2 ms** | **1.3 ms** | **6.8 ms** | **~147.0 FPS** |
| **Optimized (Ours)** | **TensorRT** | **FP16** | **1.3 ms** | **1.4 ms** | **1.2 ms** | **3.9 ms** | **~256.4 FPS** |

> 💡 **최종 성과**:
> * Python Baseline 대비 **End-to-End Latency 약 86.3% 감축 (28.4ms → 3.9ms)**
> * 처리량(Throughput) **약 7.3배 향상 (35.2 FPS → 256.4 FPS)**으로 고해상도 다채널 영상 실시간 처리 가능

---

## 🏗️ System Architecture

전체 파이프라인은 CPU(OpenCV)와 GPU(CUDA/TensorRT) 간의 전송 병목을 최소화하도록 설계되었습니다.

```text
[ Input Frame (Webcam / RTSP / Video) ]
                    │
                    ▼
   [ OpenCV C++ Pre-processing Module ]
     - Aspect-ratio Preserving Resize (Letterbox Padding)
     - Color Space Conversion (BGR to RGB)
     - HWC to CHW Layout Conversion & Normalization (/255.0)
                    │
   [ CUDA Stream Host-to-Device Copy (Pinned Memory) ]
                    │
                    ▼
     [ TensorRT Engine Inference (GPU) ]
     - FP16 Quantized Kernel Execution
     - Dynamic / Static Tensor Allocation
                    │
   [ CUDA Stream Device-to-Host Copy ]
                    │
                    ▼
   [ OpenCV C++ Post-processing Module ]
     - Confidence & Class Thresholding
     - Non-Maximum Suppression (NMS)
     - Bounding Box Coordinate Restoration (Letterbox Inversion)
                    │
                    ▼
[ Output Visualization (cv::imshow / VideoWriter / RTSP) ]
```

---

## 🛠️ Key Engineering Challenges & Troubleshooting (핵심 문제 해결)

### 1. OpenCV Pre-processing CPU 병목 최적화
* **문제 상황**: 고해상도(1080p) 입력 프레임을 640x640 크기로 조정하고, HWC 데이터를 딥러닝 입력 형식인 NCHW Float형 포인터로 변환하는 과정에서 CPU 연산 병목 발생 (전처리만 약 6ms 소요).
* **해결 방법**:
  * `cv::Mat`의 데이터 연속성(`isContinuous()`)을 보장하여 메모리 재할당 오버헤드 제거.
  * 중첩 반복문(`for`) 기반의 정규화 및 CHW 복사 로직을 `cv::split` 및 `cv::Mat::convertTo` 벡터 연산으로 교체하여 전처리 시간을 **6ms $\rightarrow$ 1.3ms**로 단축.

### 2. 동기식 `cudaMemcpy`로 인한 GPU Idle Time 제거
* **문제 상황**: 프레임마다 동기식 `cudaMemcpy`를 호출하여 CPU(전/후처리)와 GPU(추론)가 순차적으로 대기하는 유휴 시간(Idle time) 발생.
* **해결 방법**:
  * 호스트 메모리를 일반 가상 메모리가 아닌 `cudaHostAlloc`을 통한 **Pinned Memory(Page-locked)**로 할당하여 PCIe 전송 대역폭 극대화.
  * 2개 이상의 `cudaStream_t`를 활용하여 **"N번 프레임 추론(GPU)"**과 **"N+1번 프레임 전처리(CPU/Memcpy)"**를 중첩 실행하는 비동기 이중 버퍼링(Double Buffering) 구조 설계.

### 3. 종횡비 왜곡 방지(Letterbox) 및 정확한 바운딩 박스 역변환
* **문제 상황**: 원본 영상(16:9)을 모델 입력 크기(1:1, 640x640)로 단순 Resize할 경우 물체의 형태 왜곡이 발생하여 탐지율이 저하됨.
* **해결 방법**:
  * 원본의 종횡비를 보존하며 최소 패딩을 채워 넣는 Letterbox 알고리즘을 C++로 직접 구현.
  * 모델 출력 좌표($x_1, y_1, x_2, y_2$)에서 패딩 오프셋을 빼고 스케일 팩터로 나누어 원본 영상 좌표계로 정밀하게 역투영(Inverse Transform)하는 후처리 로직 완성.

---

## 💻 Tech Stack & Environment

* **Language**: C++17
* **Build System**: CMake (>= 3.20)
* **Core Libraries**:
  * **OpenCV**: 4.8.0+
  * **CUDA**: 12.0+
  * **cuDNN**: 8.9+
  * **TensorRT**: 8.6+ / 10.x
* **Tested OS**: Ubuntu 20.04 / 22.04 LTS

---

## 🚀 Getting Started

### 1. Prerequisite (Python Environment for Export)
```bash
# PyTorch 모델을 ONNX 형식으로 변환 (배치 사이즈 1 고정)
pip install ultralytics onnx
yolo export model=yolov8n.pt format=onnx opset=12 dynamic=False
```

### 2. Build TensorRT Engine via CLI
```bash
# TensorRT trtexec CLI를 이용해 FP16 최적화 엔진 빌드
trtexec --onnx=yolov8n.onnx \
        --saveEngine=yolov8n.engine \
        --fp16 \
        --verbose
```

### 3. Build C++ Project
```bash
# 레포지토리 클론
git clone https://github.com/your-username/realtime-tensorrt-cpp.git
cd realtime-tensorrt-cpp

# 빌드 디렉토리 생성 및 컴파일
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 4. Run Inference
```bash
# 웹캠 입력 실시간 추론 (Device ID: 0)
./yolo_trt --engine ../models/yolov8n.engine --source 0

# 동영상 파일 추론
./yolo_trt --engine ../models/yolov8n.engine --source ../data/test_video.mp4
```

---

## 📂 Project Directory Structure

```text
├── CMakeLists.txt              # CMake 빌드 스크립트 (CUDA, TensorRT, OpenCV 링크)
├── include
│   ├── detector.hpp            # TensorRT IExecutionContext 래퍼 클래스
│   ├── preprocessor.hpp        # Letterbox, Normalization, HWC->CHW 유틸리티
│   ├── postprocessor.hpp       # Confidence Filter, NMS, BBox 복원 모듈
│   └── types.hpp               # Bounding Box, Detection 결과 구조체 정의
├── src
│   ├── main.cpp                # 비디오 I/O 파이프라인 및 FPS 메트릭 루프
│   ├── detector.cpp            # 엔진 로드, 텐서 바인딩, 비동기 추론 구현체
│   ├── preprocessor.cpp        # OpenCV 기반 최적화 전처리 구현체
│   └── postprocessor.cpp       # NMS 및 좌표 계산 구현체
├── models/                     # .onnx 및 .engine 파일 보관 (.gitignore 적용 권장)
├── data/                       # 테스트용 샘플 이미지/비디오 디렉토리
└── assets/                     # README용 Demo GIF 및 아키텍처 다이어그램
```

---

## 👤 Author

* **작성자**: [tulg580]
* **Email**: [solkk6584@gmail.com]
* **GitHub**: [@tulg580](https://github.com/tulg580))]
