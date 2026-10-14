#pragma once
#include "types.hpp"
#include <opencv2/opencv.hpp>

class Preprocessor {
public:
    Preprocessor(int input_w = 640, int input_h = 640);
    PreprocessInfo process(const cv::Mat& src, float* dst_buffer);

private:
    int input_w_;
    int input_h_;
};