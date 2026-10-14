#include "postprocessor.hpp"

Postprocessor::Postprocessor(float conf_thresh, float nms_thresh, int num_classes)
    : conf_thresh_(conf_thresh), nms_thresh_(nms_thresh), num_classes_(num_classes) {}

std::vector<Detection> Postprocessor::process(const float* output,
                                              const PreprocessInfo& info,
                                              const cv::Size& orig_size) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // YOLOv8 출력 크기: 84 * 8400 (행: x, y, w, h + 80 클래스, 열: 8400개 앵커)
    constexpr int num_anchors = 8400;

    for (int i = 0; i < num_anchors; ++i) {
        // 최대 신뢰도를 가지는 클래스 탐색
        float max_score = 0.0f;
        int best_cls = -1;

        for (int c = 0; c < num_classes_; ++c) {
            float score = output[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                best_cls = c;
            }
        }

        if (max_score >= conf_thresh_) {
            // 박스 중심(cx, cy), 너비(w), 높이(h)
            float cx = output[0 * num_anchors + i];
            float cy = output[1 * num_anchors + i];
            float w  = output[2 * num_anchors + i];
            float h  = output[3 * num_anchors + i];

            // Letterbox 패딩 제거 및 원본 크기 역변환
            float x1 = (cx - w * 0.5f - info.pad_w) / info.scale;
            float y1 = (cy - h * 0.5f - info.pad_h) / info.scale;
            float box_w = w / info.scale;
            float box_h = h / info.scale;

            // 영상 경계 클리핑
            int ix = std::clamp(static_cast<int>(x1), 0, orig_size.width);
            int iy = std::clamp(static_cast<int>(y1), 0, orig_size.height);
            int iw = std::clamp(static_cast<int>(box_w), 0, orig_size.width - ix);
            int ih = std::clamp(static_cast<int>(box_h), 0, orig_size.height - iy);

            boxes.emplace_back(ix, iy, iw, ih);
            confidences.push_back(max_score);
            class_ids.push_back(best_cls);
        }
    }

    // NMS 적용
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, indices);

    std::vector<Detection> results;
    for (int idx : indices) {
        results.push_back({class_ids[idx], confidences[idx], boxes[idx]});
    }

    return results;
}