#pragma once
#include <opencv2/opencv.hpp>
#include <string>

struct Detection {
    int class_id{0};
    float confidence{0.0f};
    cv::Rect box;
};

// 원본 복원을 위한 전처리 정보
struct PreprocessInfo {
    float scale{1.0f};
    int pad_w{0};
    int pad_h{0};
};