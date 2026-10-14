#pragma once
#include "types.hpp"
#include <vector>

class Postprocessor {
public:
    Postprocessor(float conf_thresh = 0.25f, float nms_thresh = 0.45f, int num_classes = 80);
    std::vector<Detection> process(const float* output_buffer,
                                  const PreprocessInfo& info,
                                  const cv::Size& orig_size);

private:
    float conf_thresh_;
    float nms_thresh_;
    int num_classes_;
};