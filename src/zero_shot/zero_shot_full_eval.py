"""
zero_shot_full_eval.py
在全部测试集上跑 zero-shot 分类，对比8种prompt模板效果
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import csv
import json
import time
import torch
from PIL import Image
from collections import defaultdict
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

#os.environ["HF_HUB_OFFLINE"] = "1"  # 已下载过模型，离线加载更快

# ========== 配置 ==========
LABELS_CSV = "food_dataset/labels_cleaned.csv"
SELECTED_CLASSES_JSON = "selected_classes.json"
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
BATCH_SIZE = 32  # 每批处理多少张图
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


PROMPT_TEMPLATES = {
    "template_A": "一张{}的照片",
    "template_B": "这是{}",
    "template_C": "中式菜肴：{}",
    "template_D": "一道美味的{}",
    "template_E": "{}",                    # 极简版，只有菜名本身，作为对照基线
    "template_F": "一张中国菜{}的图片",
    "template_G": "美食：{}",
    "template_H": "餐桌上的{}",
}


def load_test_samples(labels_csv):
    """从labels.csv中筛选出test集的样本"""
    samples = []
    with open(labels_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test":
                samples.append(row)
    print(f"测试集共 {len(samples)} 张图片")
    return samples


def get_class_list(selected_classes_json):
    """返回 (拼音名列表, 拼音名->中文名字典)，按固定顺序，保证索引一致"""
    with open(selected_classes_json, "r", encoding="utf-8") as f:
        selected_classes = json.load(f)
    pinyin_to_cn = {info["name"]: info["cn_name"] for info in selected_classes.values()}
    class_names = sorted(pinyin_to_cn.keys())  # 固定顺序
    return class_names, pinyin_to_cn


def extract_features(output):
    """兼容不同transformers版本的返回格式，统一取出特征张量"""
    if isinstance(output, torch.Tensor):
        return output
    elif hasattr(output, "image_embeds"):
        return output.image_embeds
    elif hasattr(output, "text_embeds"):
        return output.text_embeds
    elif hasattr(output, "pooler_output"):
        return output.pooler_output
    else:
        return output[0]  # 兜底，取第一个元素
    

def encode_all_images(model, processor, samples, batch_size,device):
    all_features = []
    valid_samples = []

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            images = []
            for s in batch:
                try:
                    img = Image.open(s["image_path"]).convert("RGB")
                    images.append(img)
                    valid_samples.append(s)
                except Exception as e:
                    print(f"跳过损坏图片: {s['image_path']} ({e})")

            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}      # 加这一行
            raw_output = model.get_image_features(**inputs)
            image_features = extract_features(raw_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_features.append(image_features.cpu())

            if (i // batch_size) % 10 == 0:
                print(f"  已编码 {i + len(batch)}/{len(samples)} 张图片...")

    all_features = torch.cat(all_features, dim=0)
    print(f"图像编码完成，共 {all_features.shape[0]} 张，特征维度 {all_features.shape[1]}")
    return all_features, valid_samples


def encode_texts(model, processor, class_names, pinyin_to_cn, template,device):
    texts = [template.format(pinyin_to_cn[name]) for name in class_names]
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}          # 加这一行
        raw_output = model.get_text_features(**inputs)
        text_features = extract_features(raw_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu() 


def evaluate_template(image_features, text_features, valid_samples, class_names,
                        logit_scale, pinyin_to_cn):
    """给定图像特征和文本特征，计算Top-1/Top-5准确率，并返回逐条预测结果"""
    # 相似度矩阵 [N, C]，模拟CLIP内部的logit计算方式
    logits = logit_scale * image_features @ text_features.T
    probs = logits.softmax(dim=-1)

    top1_correct = 0
    top5_correct = 0
    per_sample_results = []

    for idx, sample in enumerate(valid_samples):
        true_class = sample["class"]
        sorted_indices = probs[idx].argsort(descending=True).tolist()
        pred_top1 = class_names[sorted_indices[0]]
        pred_top5 = [class_names[i] for i in sorted_indices[:5]]

        is_top1 = (pred_top1 == true_class)
        is_top5 = (true_class in pred_top5)
        top1_correct += is_top1
        top5_correct += is_top5

        per_sample_results.append({
            "image_path": sample["image_path"],
            "true_class": true_class,
            "true_cn_name": pinyin_to_cn.get(true_class, true_class),
            "pred_top1": pred_top1,
            "pred_top1_cn": pinyin_to_cn.get(pred_top1, pred_top1),
            "correct_top1": is_top1,
            "correct_top5": is_top5,
            "confidence": probs[idx][sorted_indices[0]].item(),
        })

    n = len(valid_samples)
    top1_acc = top1_correct / n
    top5_acc = top5_correct / n
    return top1_acc, top5_acc, per_sample_results


def main():
    class_names, pinyin_to_cn = get_class_list(SELECTED_CLASSES_JSON)
    samples = load_test_samples(LABELS_CSV)

    print("正在加载模型...")
    model = ChineseCLIPModel.from_pretrained(MODEL_NAME)
    model=model.to(DEVICE)
    processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    logit_scale = model.logit_scale.exp().item()  # CLIP内部用于缩放相似度的可学习参数

    print("\n开始编码全部测试图片(只需一次，后续复用)...")
    start = time.time()
    image_features, valid_samples = encode_all_images(model, processor, samples, BATCH_SIZE,DEVICE)
    print(f"图像编码耗时: {time.time() - start:.1f} 秒\n")

    # 对每种prompt模板分别评估
    summary = []
    all_template_results = {}

    for template_name, template_str in PROMPT_TEMPLATES.items():
        print(f"=== 评估模板 [{template_name}]: \"{template_str}\" ===")
        text_features = encode_texts(model, processor, class_names, pinyin_to_cn, template_str,DEVICE)
        top1_acc, top5_acc, per_sample_results = evaluate_template(
            image_features, text_features, valid_samples, class_names,
            logit_scale, pinyin_to_cn
        )
        print(f"Top-1: {top1_acc:.2%}  Top-5: {top5_acc:.2%}\n")

        summary.append({
            "template_name": template_name,
            "template_str": template_str,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
        })
        all_template_results[template_name] = per_sample_results

    # 打印对比表（按Top-1准确率从高到低排序）
    print("=" * 60)
    print("Prompt模板对比结果 (按Top-1准确率排序):")
    print("=" * 60)
    sorted_summary = sorted(summary, key=lambda x: -x["top1_acc"])
    for rank, s in enumerate(sorted_summary, 1):
        print(f"{rank}. {s['template_name']:12s} \"{s['template_str']}\"  "
              f"Top-1: {s['top1_acc']:.2%}   Top-5: {s['top5_acc']:.2%}")
        
    # 找出表现最好的模板
    best = max(summary, key=lambda x: x["top1_acc"])
    print(f"\n🏆 最佳模板: {best['template_name']} (\"{best['template_str']}\")")

    # 保存模板对比结果
    with open("food_dataset/prompt_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["template_name", "template_str", "top1_acc", "top5_acc"])
        writer.writeheader()
        writer.writerows(summary)
    print("\n✅ 模板对比结果已保存: food_dataset/prompt_comparison.csv")

    # 保存最佳模板的逐条预测结果（供Day7画混淆矩阵、找失败案例用）
    best_results = all_template_results[best["template_name"]]
    with open("food_dataset/best_template_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["image_path", "true_class", "true_cn_name", "pred_top1",
                      "pred_top1_cn", "correct_top1", "correct_top5", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_results)
    print("✅ 最佳模板逐条预测结果已保存: food_dataset/best_template_predictions.csv")


if __name__ == "__main__":
    main()