#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <ncnn/net.h>

class LQ_NCNN {
public:
    LQ_NCNN();

    bool init();
    std::string infer(const cv::Mat& bgr_image) const;

private:
    static int argmax(const ncnn::Mat& logits);

private:
    ncnn::Net net_;
    std::vector<std::string> labels_;
    bool initialized_ = false;
};
