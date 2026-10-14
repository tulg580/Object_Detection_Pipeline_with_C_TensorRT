cmake_minimum_required(VERSION 3.20)
project(yolo_trt_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. OpenCV 탐색
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

# 2. CUDA 탐색
find_package(CUDAToolkit REQUIRED)

# 3. TensorRT 설정 (설치 경로가 다를 경우 TENSORRT_DIR 수정)
set(TENSORRT_DIR "/usr" CACHE PATH "Path to TensorRT root")
find_path(NVINFER_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR} PATH_SUFFIXES include
)
find_library(NVINFER_LIBRARY nvinfer
    HINTS ${TENSORRT_DIR} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

if (NOT NVINFER_INCLUDE_DIR OR NOT NVINFER_LIBRARY)
    message(FATAL_ERROR "TensorRT (nvinfer) not found! Please check TENSORRT_DIR.")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVINFER_INCLUDE_DIR}
)

# 소스 파일 추가
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/preprocessor.cpp
    src/postprocessor.cpp
)

add_executable(yolo_trt ${SOURCES})

target_link_libraries(yolo_trt PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVINFER_LIBRARY}
)