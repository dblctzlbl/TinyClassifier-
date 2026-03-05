# TinyClassifier

轻量级图像分类模型训练框架，支持一键训练并导出 NCNN 模型，适用于嵌入式板卡部署。

---

## 1. 环境准备

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 2. 数据集格式

数据集需为 ImageFolder 结构：

```text
dataset/
  class_a/
    xxx.jpg
  class_b/
    yyy.jpg
  class_c/
    zzz.jpg
```

要求：
- 每个子目录代表一个类别
- 支持图片后缀：jpg / jpeg / png / bmp
- 至少两个类别

---

## 3. 一键训练

修改 `run_all.ps1` 顶部的配置项：

```powershell
# ========================= User Config (edit here) =========================
$PythonExe = ""           # Python路径，留空则使用系统默认
$DataRoot = "dataset"     # 数据集目录
$OutDir = "artifacts"     # 输出目录
$TargetAcc = 0.90         # 目标准确率，达到后提前停止

$Train = @{
    ImgSize = 96          # 输入图像尺寸
    BatchSize = 64        # 批大小
    Epochs = 120          # 最大训练轮数
    Lr = 1e-3             # 学习率
    WidthMult = 0.6       # 模型宽度系数
}
# ===========================================================================
```

运行脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

训练完成后输出：
- `artifacts/best_model.pt` - PyTorch 模型
- `artifacts/tiny_classifier_fp32.onnx` - ONNX 模型
- `artifacts/tiny_classifier_fp32.ncnn.param` - NCNN 参数文件
- `artifacts/tiny_classifier_fp32.ncnn.bin` - NCNN 权重文件
- `artifacts/labels.txt` - 类别标签顺序

---

## 4. 板卡推理

`Board_Card_Reasoning/` 目录下的推理代码适用于所有支持 NCNN 的嵌入式板卡，包括但不限于：
- 龙芯 2K0300 久久派
- 树莓派
- Jetson 系列
- 其他 ARM/x86 嵌入式平台

使用方法：
1. 将 `tiny_classifier_fp32.ncnn.param` 和 `tiny_classifier_fp32.ncnn.bin` 复制到板卡
2. 修改 `main.cpp` 中的 `labels` 为你的类别名称
3. 编译并运行

---

## 5. 常见问题

| 问题 | 解决方案 |
|------|----------|
| `No module named ...` | 执行 `pip install -r requirements.txt` |
| `Neither onnx2ncnn.exe nor pnnx.exe was found.` | 将 `onnx2ncnn.exe` 或 `pnnx.exe` 放入 PATH 或项目目录 |
| 精度不理想 | 扩充数据量、增加 `Epochs`、适当增大 `WidthMult` |
| 板卡推理精度差 | 确认输入图像已从 BGR 转换为 RGB（代码已内置转换）|
