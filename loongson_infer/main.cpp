#include <algorithm>
#include <chrono>
#include <cctype>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sys/stat.h>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <ncnn/net.h>

static std::vector<std::string> read_lines(const std::string& path)
{
    std::ifstream fin(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fin, line))
    {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

static std::string trim_line(std::string s)
{
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t'))
    {
        s.pop_back();
    }
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t'))
    {
        start++;
    }
    if (start > 0) s = s.substr(start);

    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF)
    {
        s = s.substr(3);
    }
    return s;
}

static std::string trim_spaces(std::string s)
{
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t'))
    {
        start++;
    }
    size_t end = s.size();
    while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t'))
    {
        end--;
    }
    return s.substr(start, end - start);
}

static std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static std::string file_ext(const std::string& p)
{
    const size_t pos = p.find_last_of('.');
    if (pos == std::string::npos) return "";
    return to_lower(p.substr(pos));
}

static bool has_image_ext(const std::string& p)
{
    std::string ext = file_ext(p);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

static bool is_dir(const std::string& path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

static std::string join_path(const std::string& a, const std::string& b)
{
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

static std::string parent_name(const std::string& path)
{
    size_t end = path.size();
    while (end > 0 && (path[end - 1] == '/' || path[end - 1] == '\\')) --end;
    if (end == 0) return "";

    size_t slash = path.find_last_of("/\\", end - 1);
    if (slash == std::string::npos) return "";

    size_t parent_end = slash;
    while (parent_end > 0 && (path[parent_end - 1] == '/' || path[parent_end - 1] == '\\')) --parent_end;
    if (parent_end == 0) return "";

    size_t parent_slash = path.find_last_of("/\\", parent_end - 1);
    if (parent_slash == std::string::npos) return path.substr(0, parent_end);
    return path.substr(parent_slash + 1, parent_end - parent_slash - 1);
}

static std::string normalize_slash(std::string s)
{
    for (auto& c : s)
    {
        if (c == '\\') c = '/';
    }
    return s;
}

static std::string label_from_path(const std::string& full_path, const std::string& root)
{
    std::string p = normalize_slash(full_path);
    std::string r = normalize_slash(root);
    if (!r.empty() && r.back() == '/') r.pop_back();

    if (!r.empty() && p.size() > r.size() && p.compare(0, r.size(), r) == 0)
    {
        std::string rest = p.substr(r.size());
        while (!rest.empty() && rest.front() == '/') rest.erase(rest.begin());
        if (!rest.empty())
        {
            size_t slash = rest.find('/');
            if (slash == std::string::npos) return rest;
            if (slash > 0) return rest.substr(0, slash);
        }
    }
    return parent_name(full_path);
}

static std::string label_from_segments(const std::string& full_path, const std::vector<std::string>& labels)
{
    std::string p = normalize_slash(full_path);
    size_t start = 0;
    while (start < p.size())
    {
        size_t slash = p.find('/', start);
        size_t end = (slash == std::string::npos) ? p.size() : slash;
        if (end > start)
        {
            std::string seg = p.substr(start, end - start);
            for (const auto& l : labels)
            {
                if (seg == l) return seg;
            }
        }
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
    return "";
}

static void collect_images_recursive_impl(const std::string& root, std::vector<std::string>& out)
{
    DIR* dir = opendir(root.c_str());
    if (!dir) return;

    struct dirent* ent = nullptr;
    while ((ent = readdir(dir)) != nullptr)
    {
        const std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        const std::string full = join_path(root, name);

        if (is_dir(full))
        {
            collect_images_recursive_impl(full, out);
        }
        else if (has_image_ext(full))
        {
            out.push_back(full);
        }
    }

    closedir(dir);
}

static std::vector<std::string> collect_images_recursive(const std::string& root)
{
    std::vector<std::string> paths;
    collect_images_recursive_impl(root, paths);
    std::sort(paths.begin(), paths.end());
    return paths;
}

static int infer_one(
    ncnn::Net& net,
    const cv::Mat& bgr,
    int target_size,
    const std::vector<std::string>& labels,
    int& out_class,
    float& out_prob)
{
    if (bgr.empty()) return -1;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(target_size, target_size));

    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR2RGB, target_size, target_size);

    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {0.0171247538f, 0.0175070028f, 0.0174291939f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    int ret = ex.extract("out0", out);
    if (ret != 0) return ret;

    out_class = -1;
    out_prob = -1.f;
    for (int i = 0; i < out.w; ++i)
    {
        const float v = out[i];
        if (v > out_prob)
        {
            out_prob = v;
            out_class = i;
        }
    }
    if (out_class < 0 || out_class >= (int)labels.size()) return -2;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::cout << "Usage:\n"
                  << "  loongson_cls_bench <model.param> <model.bin> <labels.txt> <test_root> [loops]\n"
                  << "Example:\n"
                  << "  loongson_cls_bench tiny_classifier_96.opt.param tiny_classifier_96.opt.bin labels.txt test_images 10\n";
        return 1;
    }

    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const std::string labels_path = argv[3];
    const std::string test_root = argv[4];
    const int loops = argc > 5 ? std::max(1, std::stoi(argv[5])) : 10;
    const int target_size = 96;

    std::vector<std::string> labels = read_lines(labels_path);
    for (auto& l : labels)
    {
        l = trim_line(l);
    }
    if (labels.empty())
    {
        std::cerr << "labels.txt is empty.\n";
        return 2;
    }

    if (!is_dir(test_root))
    {
        std::cerr << "test_root not found: " << test_root << "\n";
        return 3;
    }

    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = false;

    if (net.load_param(param_path.c_str()) != 0)
    {
        std::cerr << "load param failed\n";
        return 4;
    }
    if (net.load_model(bin_path.c_str()) != 0)
    {
        std::cerr << "load bin failed\n";
        return 5;
    }

    std::vector<std::string> images = collect_images_recursive(test_root);
    if (images.empty())
    {
        std::cerr << "no images found in: " << test_root << "\n";
        return 6;
    }

    std::vector<int> class_total(labels.size(), 0);
    std::vector<int> class_correct(labels.size(), 0);
    int matched_labels = 0;
    int unlabeled_images = 0;
    std::map<std::string, int> unlabeled_buckets;
    std::vector<std::string> unlabeled_samples;

    double sum_ms = 0.0;
    int total_runs = 0;

    for (int warmup = 0; warmup < 5; ++warmup)
    {
        cv::Mat bgr = cv::imread(images[warmup % images.size()], cv::IMREAD_COLOR);
        int pred = -1;
        float prob = 0.f;
        infer_one(net, bgr, target_size, labels, pred, prob);
    }

    for (const auto& img_path : images)
    {
        std::string parent = label_from_segments(img_path, labels);
        if (parent.empty())
        {
            parent = trim_spaces(label_from_path(img_path, test_root));
        }
        auto it = std::find(labels.begin(), labels.end(), parent);
        int gt = it == labels.end() ? -1 : (int)std::distance(labels.begin(), it);

        cv::Mat bgr = cv::imread(img_path, cv::IMREAD_COLOR);
        if (bgr.empty()) continue;

        int pred = -1;
        float prob = 0.f;
        double one_img_ms = 0.0;

        for (int k = 0; k < loops; ++k)
        {
            auto t0 = std::chrono::steady_clock::now();
            int ret = infer_one(net, bgr, target_size, labels, pred, prob);
            auto t1 = std::chrono::steady_clock::now();
            if (ret != 0)
            {
                std::cerr << "infer failed: " << img_path << " ret=" << ret << "\n";
                continue;
            }
            one_img_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        one_img_ms /= loops;
        sum_ms += one_img_ms;
        total_runs += 1;

        if (gt >= 0)
        {
            class_total[gt] += 1;
            if (pred == gt) class_correct[gt] += 1;
            matched_labels += 1;
        }
        else
        {
            unlabeled_images += 1;
            if (parent.empty()) parent = "UNKNOWN";
            unlabeled_buckets[parent] += 1;
            if (unlabeled_samples.size() < 5)
            {
                unlabeled_samples.push_back(img_path);
            }
        }
    }

    int total = 0;
    int correct = 0;
    for (size_t i = 0; i < labels.size(); ++i)
    {
        total += class_total[i];
        correct += class_correct[i];
    }

    unlabeled_images = (int)images.size() - matched_labels;
    const double acc = total > 0 ? (double)correct / total : 0.0;
    const double avg_ms = total_runs > 0 ? sum_ms / total_runs : 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "===== NCNN Bench Result =====\n";
    std::cout << "Total images: " << images.size() << "\n";
    std::cout << "Labeled images: " << matched_labels << "\n";
    if (unlabeled_images > 0)
    {
        std::cout << "Unlabeled images: " << unlabeled_images << " (check folder names vs labels.txt)\n";
        std::cout << "Unlabeled buckets:\n";
        for (const auto& kv : unlabeled_buckets)
        {
            std::cout << "  [" << kv.first << "]: " << kv.second << "\n";
        }
        if (!unlabeled_samples.empty())
        {
            std::cout << "Unlabeled samples:\n";
            for (const auto& s : unlabeled_samples)
            {
                std::cout << "  " << s << "\n";
            }
        }
    }
    std::cout << "Accuracy: " << acc << "\n";

    for (size_t i = 0; i < labels.size(); ++i)
    {
        double cacc = class_total[i] > 0 ? (double)class_correct[i] / class_total[i] : 0.0;
        std::cout << "[" << labels[i] << "] " << class_correct[i] << "/" << class_total[i]
                  << " = " << cacc << "\n";
    }

    return 0;
}
