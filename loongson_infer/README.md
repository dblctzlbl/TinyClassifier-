# 龙芯板卡最小分类推理（NCNN）

## 1) 板卡上的目录结构

建议在板卡放成如下结构：

```text
deploy_cls/
  loongson_cls_bench
  tiny_classifier_96.opt.param
  tiny_classifier_96.opt.bin
  labels.txt
  test_images/
    vehicle/   (30 images)
    weapon/    (30 images)
    supplies/  (30 images)
```

说明：
- 图片格式支持：jpg、jpeg、png、bmp
- 输入会在程序内自动 resize 到 96x96
- 支持任意类别数（由 `labels.txt` 行数决定）
- `test_images` 子目录名需与 `labels.txt` 内容一致

## 2) 本机准备每类30张测试图

在当前工程目录执行：

```bash
C:/Users/a1523/.conda/envs/loong/python.exe prepare_board_testset.py --data-root dataset --out-dir board_pack/test_images --per-class 30 --seed 42
```

然后把以下文件拷到板卡 `deploy_cls/`：
- `artifacts/tiny_classifier_96.opt.param`
- `artifacts/tiny_classifier_96.opt.bin`
- `artifacts/labels.txt`（训练脚本自动生成，表示模型输出顺序）
- `board_pack/test_images/`

## 3) 板卡编译

在板卡上执行（假设已安装 ncnn 和 opencv）：

```bash
mkdir -p build && cd build
cmake ../loongson_infer
make -j4
```

生成可执行文件：`loongson_cls_bench`

## 4) 板卡测试

在 `deploy_cls/` 目录执行：

```bash
./loongson_cls_bench tiny_classifier_96.opt.param tiny_classifier_96.opt.bin labels.txt test_images 10
```

参数 `10` 表示每张图重复推理10次后求均值，输出会给出：
- 总体 Accuracy
- 每类 Accuracy
- 平均时延（ms，单线程）

目标：平均时延 < 10ms。
