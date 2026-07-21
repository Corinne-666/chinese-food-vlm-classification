"""
export_predictions_lora_augmented.py
====================================================
用途：加载已经训练好并保存的 "LoRA + 数据增强" 权重
     (food_dataset/lora_weights_augmented)，在测试集上跑一遍推理，
     把每张图片的 (image_path, true_label, pred_label, confidence)
     导出为 CSV，供 task4_analysis.py 做混淆矩阵/失败案例/菜系分析。

不需要重新训练——权重已经存在磁盘上了，这里只做加载+推理+导出。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"

import csv
import json
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from peft import PeftModel

# ========== 配置区（与训练脚本保持一致）==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS_CSV = "food_dataset/labels_cleaned.csv"
SELECTED_CLASSES_JSON = "selected_classes.json"
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
LORA_WEIGHTS_PATH = "food_dataset/lora_weights_augmented"  # 已保存的权重
BEST_TEMPLATE = "一道美味的{}"
OUTPUT_CSV = "results_lora_augmented.csv"


# ---------------------------------------------------
# 数据准备（与训练脚本一致）
# ---------------------------------------------------
def load_samples(labels_csv):
    train_samples, test_samples = [], []
    with open(labels_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "train":
                train_samples.append(row)
            elif row["split"] == "test":
                test_samples.append(row)
    return train_samples, test_samples


def get_class_list(selected_classes_json):
    with open(selected_classes_json, "r", encoding="utf-8") as f:
        selected_classes = json.load(f)
    pinyin_to_cn = {info["name"]: info["cn_name"] for info in selected_classes.values()}
    class_names = sorted(pinyin_to_cn.keys())
    return class_names, pinyin_to_cn


# ---------------------------------------------------
# 特征提取（与训练脚本一致）
# ---------------------------------------------------
def extract_features(output):
    if isinstance(output, torch.Tensor):
        return output
    elif hasattr(output, "text_embeds"):
        return output.text_embeds
    elif hasattr(output, "image_embeds"):
        return output.image_embeds
    elif hasattr(output, "pooler_output"):
        return output.pooler_output
    else:
        return output[0]


def encode_images_eval(model, processor, image_paths, device, batch_size=32):
    all_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            raw_output = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = extract_features(raw_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def encode_texts_eval(model, processor, texts, device):
    model.eval()
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        raw_output = model.get_text_features(**inputs)
        feats = extract_features(raw_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()


# ---------------------------------------------------
# 评估 + 导出预测结果（新增部分）
# ---------------------------------------------------
def evaluate_and_export(model, processor, test_samples, class_names, pinyin_to_cn,
                         template, device, save_csv_path):
    texts = [template.format(pinyin_to_cn[c]) for c in class_names]
    text_feats = encode_texts_eval(model, processor, texts, device)

    image_paths = [s["image_path"] for s in test_samples]
    test_feats = encode_images_eval(model, processor, image_paths, device)

    similarities = test_feats @ text_feats.T
    logit_scale = model.logit_scale.exp().item()   # 如果报错试试 model.base_model.logit_scale.exp().item()
    probs = (similarities * logit_scale).softmax(dim=-1)
    print("logit_scale exp值:", model.logit_scale.exp().item())

    records = []
    top1_correct, top5_correct = 0, 0
    for idx, s in enumerate(test_samples):
        sorted_idx = probs[idx].argsort(descending=True).tolist()
        pred_top1_pinyin = class_names[sorted_idx[0]]
        pred_top5_pinyin = [class_names[i] for i in sorted_idx[:5]]
        confidence = probs[idx][sorted_idx[0]].item()

        top1_correct += (pred_top1_pinyin == s["class"])
        top5_correct += (s["class"] in pred_top5_pinyin)

        records.append({
            "image_path": s["image_path"],
            "true_label": pinyin_to_cn[s["class"]],
            "pred_label": pinyin_to_cn[pred_top1_pinyin],
            "confidence": round(confidence, 4),
        })

    with open(save_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "true_label", "pred_label", "confidence"])
        writer.writeheader()
        writer.writerows(records)

    n = len(test_samples)
    top1_acc, top5_acc = top1_correct / n, top5_correct / n
    print(f"预测结果已保存至: {save_csv_path}  (共 {n} 条)")
    print(f"复核准确率: Top-1 {top1_acc:.2%}  Top-5 {top5_acc:.2%}")
    return top1_acc, top5_acc


# ---------------------------------------------------
# 主流程
# ---------------------------------------------------
def main():
    class_names, pinyin_to_cn = get_class_list(SELECTED_CLASSES_JSON)
    _, test_samples = load_samples(LABELS_CSV)
    print(f"测试集: {len(test_samples)} 张")

    processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)

    print("加载基础模型 + LoRA权重...")
    base_model = ChineseCLIPModel.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
    model = model.to(DEVICE)
    model.eval()

    print("开始评估并导出逐样本预测结果...")
    evaluate_and_export(
        model, processor, test_samples, class_names, pinyin_to_cn,
        BEST_TEMPLATE, DEVICE, OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()