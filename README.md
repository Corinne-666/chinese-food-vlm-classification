# 中餐食物VLM分类（食物卡路里识别 - 前置子任务）

基于视觉语言模型（Chinese-CLIP）的中餐食物开放词汇分类，作为"食物卡路里识别"项目的关键前置任务。依次实现并对比了 **Zero-shot → Few-shot → LoRA微调 → LoRA+数据增强** 四条技术路线，并对数据集标注质量做了系统性复核与清洗。

## 目录

- [数据集](#数据集)
- [方法与结果总览](#方法与结果总览)
- [快速复现](#快速复现)
- [目录结构](#目录结构)
- [数据清洗](#数据清洗)
- [详细分析报告](#详细分析报告)

## 数据集

- 30类常见中餐食物，覆盖川菜、粤菜、鲁菜、苏菜、湘菜、沪菜、浙菜、东北菜、徽菜、客家菜、朝鲜族腌菜、陕西小吃等菜系
- 共4503张图像，训练集 : 测试集 = 约 7 : 2（另有验证集）
- 标注格式：`labels.csv`（列：`image_path`, `class`, `split`），`selected_classes.json`（pinyin类别名 ↔ 中文菜名映射）
- 数据集经过人工复核清洗（详见[数据清洗](#数据清洗)一节），`data/labels_cleaned.csv` 为清洗后的最终版本

> 原始图片文件体积较大，未包含在本仓库中，请参考 `data/README.md` 中的获取方式，并按 `labels_cleaned.csv` 中的相对路径组织好本地目录（默认约定为 `food_dataset/split/<pinyin类名>/xxx.jpg`）。

## 方法与结果总览

| 方法 | 说明 | 清洗前 Top-1 | 清洗后 Top-1 | 清洗前 Top-5 | 清洗后 Top-5 |
|------|------|------|------|------|------|
| Zero-shot | Chinese-CLIP，最佳模板"一道美味的{}" | 84.28% | 85.93% | 96.98% | 97.62% |
| Few-shot v1 | 纯图像原型法，N=10 | 88.30% | 90.03% | 98.60% | 99.18% |
| Few-shot v2 | 图文融合原型，alpha=0.5 | 90.38% | 92.07% | 99.13% | 99.59% |
| Few-shot v2b | alpha消融最佳（alpha=0.6） | 90.63% | 92.32% | 99.07% | 99.52% |
| LoRA微调 | r=8, alpha=16, dropout=0.1，5 epoch | 93.76% | 95.60% | 99.56% | 99.82% |
| **LoRA微调 + 数据增强（最终版本）** | 旋转/裁剪/颜色抖动/翻转 | 94.12% | **96.33%** | 99.60% | **99.86%** |

**关键发现**：数据清洗对六种方法均带来提升（+1.65~+2.21个百分点），其中LoRA+数据增强这一最终版本的提升幅度（+2.21pp）明显高于其余五种方法（+1.65~1.84pp），推测与数据增强会放大标注错误样本的影响有关。清洗前的标注噪声是系统性的、与具体分类方法无关，此前汇报的准确率均被这部分噪声系统性拖低。详见 [`reports/analysis_report.md`](reports/analysis_report.md) 第六节。

## 快速复现

```bash
pip install -r requirements.txt
```

按顺序执行（假设已按约定组织好 `food_dataset/` 目录）：

```bash
# 1. Zero-shot：对比8种prompt模板，选出最佳模板
python src/zero_shot/prompt_comparison.py

# 2. Few-shot：纯图像原型 → 图文融合 → alpha消融
python src/few_shot/few_shot_v1_image_only.py
python src/few_shot/few_shot_v2_fusion.py
python src/few_shot/few_shot_v2b_alpha_ablation.py

# 3. LoRA微调（可选：不带/带数据增强两个版本）
python src/lora/lora_finetune.py
python src/lora/lora_finetune_augmented.py

# 4. 导出逐样本预测结果（用于后续分析，无需重新训练，直接加载已保存权重）
python src/eval/export_predictions_lora_augmented.py

# 5. 生成混淆矩阵 / 失败案例 / 分菜系分析 / 每类指标
python src/eval/analysis.py
```

## 目录结构

```
chinese-food-vlm-classification/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── labels_cleaned.csv          # 清洗后的标注（最终版本）
│   ├── selected_classes.json       # 类别定义 / 中英文映射
│   └── README.md                   # 数据来源、采集方式、清洗流程说明
│
├── src/
│   ├── zero_shot/
│   │   └── prompt_comparison.py
│   ├── few_shot/
│   │   ├── few_shot_v1_image_only.py
│   │   ├── few_shot_v2_fusion.py
│   │   └── few_shot_v2b_alpha_ablation.py
│   ├── lora/
│   │   ├── lora_finetune.py
│   │   └── lora_finetune_augmented.py
│   ├── eval/
│   │   ├── export_predictions_lora_augmented.py
│   │   └── analysis.py
│   └── data_cleaning/
│       └── clean_labels_workflow.py
│
├── results/
│   ├── prompt_comparison.csv
│   ├── fewshot_v1_result.csv
│   ├── fewshot_v2_result.csv
│   ├── alpha_ablation.csv
│   ├── results_lora_augmented.csv
│   ├── per_class_report.csv
│   └── figures/
│       ├── confusion_matrix.png
│       ├── cuisine_accuracy.png
│       └── failure_cases.png
│
└── reports/
    └── analysis_report.md          # 完整分析报告（混淆矩阵/失败案例/菜系分层/清洗效果）
```

## 数据清洗

对LoRA+数据增强版本在测试集上置信度最高的失败案例做了逐一人工复核，发现约60%的"失败案例"实际是数据集标注错误，模型判断本身是对的。清洗流程：

1. `src/eval/export_predictions_lora_augmented.py` 导出逐样本预测结果（含置信度）
2. `src/data_cleaning/clean_labels_workflow.py` 的 `export_wrong_samples_for_review()`，把错判样本按 `image_path` 关联回 `labels.csv`，导出人工复核清单
3. 人工核对每条样本，标注错误的填正确类别、不属于任何类别的填 `DELETE`
4. `apply_corrections()` 批量回写，生成清洗后的 `labels_cleaned.csv`
5. 用清洗后的标注重跑全部实验，对比清洗前后的准确率变化（见上方结果总览表）

## 详细分析报告

完整的混淆矩阵分析、按菜系分层分析、失败案例三分类复核（标注错误 / 真实视觉难点 / 模型真实识别错误）、数据清洗效果验证，见 [`reports/analysis_report.md`](reports/analysis_report.md)。
