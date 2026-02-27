# TinyClassifier 通用训练与导出指南

本工程支持“任意多类图像分类数据集”的快速替换与复用，核心流程：

1. 准备数据集（ImageFolder 结构）
2. 训练并评估模型
3. 导出 ONNX / NCNN
4. （可选）抽样测试集用于部署验证

---

## 1. 关键脚本

- `train_tiny_classifier.py`：训练 + 验证 + 测试 + 导出 ONNX + 生成 `metrics.json` 与 `labels.txt`
- `evaluate_local_accuracy.py`：本地复核测试集与全量精度
- `convert_to_ncnn.ps1`：ONNX 转 NCNN（自动探测 `onnx2ncnn` / `pnnx`）
- `prepare_board_testset.py`：按类别随机抽样测试图片

默认输入目录：`dataset/`  
默认输出目录：`artifacts/`

说明：所有脚本都支持通过 `--data-root` 指定自定义数据集路径（绝对路径或相对路径）。

---

## 2. 环境准备

建议使用你的 Conda 环境 `loong`：

```powershell
(C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate C:\Users\a1523\.conda\envs\loong)
```

安装依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如只想安装最小核心包，也可使用：

```powershell
python -m pip install torch torchvision onnx onnxsim numpy pillow
```

---

## 3. 如何更换训练集（重点）

### 3.1 数据组织格式

把新数据集放成以下结构（类别名可任意）：

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
- 图片格式支持：jpg / jpeg / png / bmp
- 每类建议至少 80~100 张（越多越稳）

### 3.2 类别顺序规则

类别索引由 `ImageFolder` 按“文件夹名字典序”自动生成。  
也就是说：模型输出顺序、`metrics.json` 中 `classes` 顺序、`labels.txt` 顺序必须一致。训练脚本会自动写出该顺序的 `labels.txt`。

### 3.3 切换到新数据集的最短步骤

1) 清空或替换 `dataset/` 内容为你的新类别目录  
2) 直接重新训练（命令见第 4 节）  
3) 训练完成后，直接使用 `artifacts/labels.txt`（已自动生成）作为最终标签顺序

自定义路径示例：

```powershell
python train_tiny_classifier.py --data-root "D:/datasets/my_cls"
python evaluate_local_accuracy.py --data-root "D:/datasets/my_cls" --checkpoint artifacts/best_model.pt
python prepare_board_testset.py --data-root "D:/datasets/my_cls" --out-dir board_pack/test_images --per-class 30 --seed 42
```

---

## 4. 训练命令

默认训练：

```powershell
python train_tiny_classifier.py
```

推荐显式参数：

```powershell
python train_tiny_classifier.py `
  --data-root dataset `
  --img-size 96 `
  --batch-size 64 `
  --epochs 30 `
  --lr 1e-3 `
  --weight-decay 2e-4 `
  --num-workers 2 `
  --seed 42 `
  --width-mult 0.6 `
  --patience 8 `
  --out-dir artifacts
```

训练后关键产物：
- `artifacts/best_model.pt`
- `artifacts/tiny_classifier_96.onnx`
- `artifacts/labels.txt`（自动生成，按模型输出顺序）
- `artifacts/metrics.json`

---

## 5. 本地精度评估

```powershell
python evaluate_local_accuracy.py `
  --data-root dataset `
  --checkpoint artifacts/best_model.pt `
  --img-size 96 `
  --batch-size 128 `
  --num-workers 2 `
  --seed 42
```

输出包括：
- Test split accuracy（与训练同划分）
- Full dataset accuracy（全量数据）
- 各类别精度

---

## 6. 导出 NCNN

执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\convert_to_ncnn.ps1
```

默认产物：
- `artifacts/tiny_classifier_96.ncnn.param`
- `artifacts/tiny_classifier_96.ncnn.bin`
- 若可优化则附带：
  - `artifacts/tiny_classifier_96.opt.param`
  - `artifacts/tiny_classifier_96.opt.bin`

---

## 7. 生成通用抽样测试集（可选）

```powershell
python prepare_board_testset.py `
  --data-root dataset `
  --out-dir board_pack/test_images `
  --per-class 30 `
  --seed 42
```

如果你的原始类别名和目标部署标签名不同，可用：

```powershell
python prepare_board_testset.py --data-root dataset --out-dir board_pack/test_images --per-class 30 --seed 42 --label-map "old1=new1,old2=new2"
```

---

## 8. 常见问题

- `No module named ...`：缺少依赖，先 `pip install`
- `onnx2ncnn/pnnx not found`：把工具加入 PATH 或放到工程目录
- 精度不达标：优先增加数据量、提高 `epochs`、适当调大 `width-mult`
- 速度不达标：减小 `width-mult`、保持输入 96×96、使用优化后的 `.opt` 模型

---

## 9. 一套最短通用命令

```powershell
python train_tiny_classifier.py --data-root dataset
python evaluate_local_accuracy.py --data-root dataset --checkpoint artifacts/best_model.pt
powershell -ExecutionPolicy Bypass -File .\convert_to_ncnn.ps1
python prepare_board_testset.py --data-root dataset --out-dir board_pack/test_images --per-class 30 --seed 42
```

---

## 10. 一键脚本（推荐）

已提供一键脚本：`run_all.ps1`，会串行执行：

1. 训练（`train_tiny_classifier.py`）
2. 评估（`evaluate_local_accuracy.py`）
3. 导出 NCNN（`convert_to_ncnn.ps1`）

使用方法：

```powershell
.\run_all.ps1
```

也可以使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

自定义配置方式：

- 打开 `run_all.ps1`
- 修改顶部“User Config (edit here)”即可（例如 `PythonExe`、`DataRoot`、`Epochs`、`BatchSize`、`WidthMult`）
- 支持把 `DataRoot` 改成任意路径，如：`D:/datasets/my_cls`
- 脚本默认使用“当前终端环境”的 `python`，并打印实际 `sys.executable` 以便确认环境

---

## 11. run_all.ps1 日志说明（对照输出看）

### 11.1 环境与依赖

- `Using Python: ...`：脚本选中的 Python 可执行文件路径
- `Python executable from runtime: ...`：Python 运行时实际路径（应与上面一致）
- `===== Check Dependencies =====` + `ok`：`torch/torchvision/onnx/onnxsim` 依赖检查通过

### 11.2 训练阶段（Train Model）

- `Using device: cpu`：本次用 CPU 训练（若有可用 GPU 会显示 cuda）
- `Data root: dataset`：本次读取的数据集根目录
- `Classes: [...]`：类别顺序（即模型输出顺序）
- `Train/Val/Test: a/b/c`：按 8:1:1 分层划分后的样本数
- `Epoch xx/xx ...`：每轮训练/验证损失与精度
- `Early stopping ...`：验证集长期无提升，提前停止
- `Best val_acc=...`：最佳验证精度及对应轮次
- `Test loss/test acc`：在测试集上的最终结果
- `Torch CPU(1 thread) avg latency`：PyTorch 单线程推理时延参考
- `Saved: ...`：本轮训练输出文件（含自动生成的 `labels.txt`）

### 11.3 评估阶段（Evaluate Accuracy）

- `Test split accuracy`：与训练划分一致的测试集精度
- `[Test] 类别: x/y`：各类别在测试集上的精度
- `Full dataset accuracy`：整个数据集上的精度
- `[Full] 类别: x/y`：各类别在全量数据上的精度

### 11.4 导出阶段（Export NCNN）

- `pnnxparam/ncnnparam/...`：导出过程中的中间文件与目标文件路径
- `pass_level...`：pnnx/onnx 图优化与转换流程日志
- `Conversion completed: ...`：NCNN 转换成功（至少有 `.ncnn.param/.ncnn.bin`）

### 11.5 最终汇总

- `All steps finished. Output dir: artifacts`：一键流程全部完成
- `Model / Label order / Metrics / ONNX`：关键产物路径汇总

### 11.6 两个常见 Warning 含义

- `torch.load ... FutureWarning (weights_only=False)`：PyTorch 的未来行为提醒，不是当前错误，流程可正常完成
- `onnxruntime ... Unknown CPU vendor`：CPU 信息识别告警，常见于某些平台，不影响本次导出成功
