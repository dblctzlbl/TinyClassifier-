#include "LQ_NCNN.hpp"

#include <limits>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace {
// ===================== 可修改配置 =====================
const std::string kModelParamPath = "../artifacts/tiny_classifier_fp32.ncnn.param";
const std::string kModelBinPath = "../artifacts/tiny_classifier_fp32.ncnn.bin";

// 类别名，顺序必须与训练时类别索引顺序一致
const std::vector<std::string> kUserLabels = {
    "supplies",
    "vehicle",
    "weapon"
};

const int kInputWidth = 96;
const int kInputHeight = 96;

// 训练时使用的是 ImageNet 标准归一化
const float kMeanVals[3] = {123.675f, 116.28f, 103.53f};
const float kNormVals[3] = {0.01712475f, 0.017507f, 0.01742919f};

// 按当前导出网络默认输入/输出名
const char* kInputBlobName = "in0";
const char* kOutputBlobName = "out0";
// ======================================================================
}

LQ_NCNN::LQ_NCNN() = default;

bool LQ_NCNN::init() {
    net_.opt.use_vulkan_compute = false;
    net_.opt.num_threads = 1;

    if (net_.load_param(kModelParamPath.c_str()) != 0) {
        return false;
    }
    if (net_.load_model(kModelBinPath.c_str()) != 0) {
        return false;
    }

    labels_ = kUserLabels;
    initialized_ = true;
    return true;
}

std::string LQ_NCNN::infer(const cv::Mat& bgr_image) const {
    if (!initialized_) {
        throw std::runtime_error("LQ_NCNN not initialized. Call init() first.");
    }
    if (bgr_image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }

    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(kInputWidth, kInputHeight));

    ncnn::Mat input = ncnn::Mat::from_pixels(
        resized.data,
        ncnn::Mat::PIXEL_BGR,
        kInputWidth,
        kInputHeight
    );
    input.substract_mean_normalize(kMeanVals, kNormVals);

    ncnn::Extractor ex = net_.create_extractor();
    ex.input(kInputBlobName, input);

    ncnn::Mat logits;
    ex.extract(kOutputBlobName, logits);

    const int class_id = argmax(logits);
    if (class_id < 0) {
        throw std::runtime_error("Failed to get class id from output logits.");
    }

    if (class_id >= 0 && class_id < static_cast<int>(labels_.size())) {
        return labels_[class_id];
    }
    return std::to_string(class_id);
}

int LQ_NCNN::argmax(const ncnn::Mat& logits) {
    if (logits.w <= 0) {
        return -1;
    }

    int best_index = 0;
    float best_value = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < logits.w; ++i) {
        const float value = logits[i];
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    return best_index;
}
