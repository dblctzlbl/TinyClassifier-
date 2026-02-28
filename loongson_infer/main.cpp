#include <chrono>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

#include "LQ_NCNN.hpp"

int main() {
    try {
        LQ_NCNN classifier;
        if (!classifier.init()) {
            std::cerr << "Init failed: check model param/bin path in LQ_NCNN.cpp" << std::endl;
            return 1;
        }

        const std::string image_path = "test.jpg";
        const cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Cannot read image: " << image_path << std::endl;
            return 1;
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        const std::string cls = classifier.infer(image);
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Class: " << cls << std::endl;
        std::cout << "Infer time: " << ms << " ms" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
