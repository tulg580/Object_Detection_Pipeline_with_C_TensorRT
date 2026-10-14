#include "detector.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <video_path or camera_id>\n";
        std::cerr << "Example: " << argv[0] << " ../models/yolov8n.engine 0\n";
        return -1;
    }

    std::string engine_path = argv[1];
    std::string source = argv[2];

    try {
        Detector detector(engine_path);
        Preprocessor preprocessor(640, 640);
        Postprocessor postprocessor(0.3f, 0.45f);

        cv::VideoCapture cap;
        if (source.size() == 1 && std::isdigit(source[0])) {
            cap.open(std::stoi(source)); // 웹캠
        } else {
            cap.open(source); // 동영상 파일
        }

        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video stream: " << source << "\n";
            return -1;
        }

        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) break;

            auto start_time = std::chrono::high_resolution_clock::now();

            // 1. 전처리 (CPU)
            PreprocessInfo info = preprocessor.process(frame, detector.getHostInputBuffer());

            // 2. 추론 (GPU via TensorRT)
            detector.infer();

            // 3. 후처리 (CPU NMS)
            std::vector<Detection> detections = postprocessor.process(
                detector.getHostOutputBuffer(), info, frame.size());

            auto end_time = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            float fps = 1000.0f / latency;

            // 4. 시각화 (바운딩 박스 & 텍스트 렌더링)
            for (const auto& det : detections) {
                cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
                std::string label = "Class " + std::to_string(det.class_id) + ": " +
                                    cv::format("%.2f", det.confidence);
                cv::putText(frame, label, cv::Point(det.box.x, std::max(20, det.box.y - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }

            // FPS 및 지연시간 표시
            std::string perf_text = cv::format("FPS: %.1f | Latency: %.1f ms", fps, latency);
            cv::putText(frame, perf_text, cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

            cv::imshow("YOLOv8 TensorRT Pipeline", frame);
            if (cv::waitKey(1) == 27) { // ESC 누르면 종료
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}