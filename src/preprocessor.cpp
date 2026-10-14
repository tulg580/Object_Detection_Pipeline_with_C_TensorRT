#include "preprocessor.hpp"

Preprocessor::Preprocessor(int input_w, int input_h)
    : input_w_(input_w), input_h_(input_h) {}

PreprocessInfo Preprocessor::process(const cv::Mat& src, float* dst_buffer) {
    PreprocessInfo info;

    // 1. 종횡비 유지를 위한 스케일 계산
    info.scale = std::min(static_cast<float>(input_w_) / src.cols,
                          static_cast<float>(input_h_) / src.rows);
    int new_w = std::round(src.cols * info.scale);
    int new_h = std::round(src.rows * info.scale);

    info.pad_w = (input_w_ - new_w) / 2;
    info.pad_h = (input_h_ - new_h) / 2;

    // 2. Resize 수행
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 3. 남는 공간 패딩 (114, 114, 114는 YOLO 표준 패딩값)
    cv::Mat letterboxed;
    cv::copyMakeBorder(resized, letterboxed,
                       info.pad_h, input_h_ - new_h - info.pad_h,
                       info.pad_w, input_w_ - new_w - info.pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 4. BGR -> RGB 색공간 변환
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    // 5. HWC -> CHW 변환 및 0.0 ~ 1.0 정규화 (cv::Mat::convertTo 벡터 연산 사용)
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    int channel_size = input_w_ * input_h_;
    for (int c = 0; c < 3; ++c) {
        channels[c].convertTo(
            cv::Mat(input_h_, input_w_, CV_32FC1, dst_buffer + c * channel_size),
            CV_32FC1,
            1.0f / 255.0f
        );
    }

    return info;
}