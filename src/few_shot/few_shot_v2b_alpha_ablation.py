"""
few_shot_v2b_alpha_ablation.py
====================================================
在 v2(图文融合) 基础上的消融实验：对比不同 alpha 值的效果

背景：
v2 用 alpha=0.5 拿到了 Top-1 90.38%，但0.5只是一个直觉上的中间值，
没有验证过是不是最优的融合比例。这里系统性地测试一组alpha，
找到图像信息和文本信息的最佳配比，同时也能画出一条效果曲线，
直观展示"融合比例"这个超参数如何影响最终效果。

优化点：
图像原型(未融合)和文本特征，在所有alpha值下都是同一份数据，
只有"加权融合"这一步的计算量随alpha变化。所以这里只编码一次
图像和文本，之后循环对多个alpha做融合+评估，避免重复跑最耗时的
图像编码步骤。
====================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"

import csv
import json
import random
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# ========== 配置区 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS_CSV = "food_dataset/labels_cleaned.csv"
SELECTED_CLASSES_JSON = "selected_classes.json"
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
BEST_TEMPLATE = "一道美味的{}"
N_SHOTS = 10
BATCH_SIZE = 32
SEED = 42

# 要对比的alpha取值：0.0=纯文本(约等于zero-shot)，1.0=纯图像(等于v1)
ALPHA_LIST = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

random.seed(SEED)


# ---------------------------------------------------
# 数据准备（与v2一致）
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


def sample_few_shot(train_samples, class_names, n_shots):
    by_class = {c: [] for c in class_names}
    for s in train_samples:
        if s["class"] in by_class:
            by_class[s["class"]].append(s)
    few_shot_samples = {}
    for c in class_names:
        pool = by_class[c]
        few_shot_samples[c] = random.sample(pool, min(n_shots, len(pool)))
    return few_shot_samples


# ---------------------------------------------------
# 编码部分（与v2一致：图像手动调底层模块，文本用get_text_features兜底）
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


def encode_images(model, processor, image_paths, device, batch_size=32):
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_output = vision_outputs.pooler_output
            image_embeds = model.visual_projection(pooled_output)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            all_feats.append(image_embeds.cpu())
    return torch.cat(all_feats, dim=0)


def encode_texts(model, processor, texts, device):
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        raw_output = model.get_text_features(**inputs)
        text_embeds = extract_features(raw_output)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds.cpu()


# ---------------------------------------------------
# 核心：先分别算出"图像原型"和"文本特征"，不做融合
# ---------------------------------------------------

def compute_image_prototypes(model, processor, few_shot_samples, class_names, device):
    """
    只计算每类的图像原型(未融合)，供后续多个alpha复用，
    避免每个alpha都重新跑一次图像编码。
    """
    prototypes = []
    for c in class_names:
        paths = [s["image_path"] for s in few_shot_samples[c]]
        img_feats = encode_images(model, processor, paths, device)
        prototype = img_feats.mean(dim=0)
        prototype = prototype / prototype.norm()
        prototypes.append(prototype)
    return torch.stack(prototypes, dim=0)  # [C, D]


def fuse_prototypes(image_prototypes, text_features, alpha):
    """
    给定已经算好的图像原型和文本特征，按指定alpha做融合。
    这一步计算量很小(只是加权+归一化)，可以对每个alpha快速重复调用。
    """
    fused = alpha * image_prototypes + (1 - alpha) * text_features
    fused = fused / fused.norm(dim=-1, keepdim=True)
    return fused


# ---------------------------------------------------
# 评估部分（与v2一致）
# ---------------------------------------------------

def evaluate(test_feats, prototypes, test_samples, class_names):
    """
    这里传入的test_feats是提前算好的测试图特征(所有alpha共用，同样避免重复编码)
    """
    similarities = test_feats @ prototypes.transpose(0, 1)
    probs = similarities.softmax(dim=-1)

    top1_correct, top5_correct = 0, 0
    for idx, s in enumerate(test_samples):
        sorted_idx = probs[idx].argsort(descending=True).tolist()
        pred_top1 = class_names[sorted_idx[0]]
        pred_top5 = [class_names[i] for i in sorted_idx[:5]]
        top1_correct += (pred_top1 == s["class"])
        top5_correct += (s["class"] in pred_top5)

    n = len(test_samples)
    return top1_correct / n, top5_correct / n


# ---------------------------------------------------
# 主流程
# ---------------------------------------------------

def main():
    class_names, pinyin_to_cn = get_class_list(SELECTED_CLASSES_JSON)
    train_samples, test_samples = load_samples(LABELS_CSV)
    print(f"训练集: {len(train_samples)} 张 | 测试集: {len(test_samples)} 张")

    print("加载模型...")
    model = ChineseCLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    few_shot_samples = sample_few_shot(train_samples, class_names, N_SHOTS)
    print(f"每类抽取 {N_SHOTS} 张作为 few-shot 支持集\n")

    # 关键：这三步只算一次，后面循环所有alpha时直接复用
    print("计算图像原型(未融合)...")
    image_prototypes = compute_image_prototypes(model, processor, few_shot_samples, class_names, DEVICE)

    print("计算文本特征...")
    texts = [BEST_TEMPLATE.format(pinyin_to_cn[c]) for c in class_names]
    text_features = encode_texts(model, processor, texts, DEVICE)

    print("编码全部测试图片(只需一次)...")
    test_image_paths = [s["image_path"] for s in test_samples]
    test_feats = encode_images(model, processor, test_image_paths, DEVICE)
    print(f"测试图编码完成，形状: {test_feats.shape}\n")

    # 对每个alpha分别融合+评估
    results = []
    print(f"{'alpha':>8} | {'Top-1':>8} | {'Top-5':>8}")
    print("-" * 32)
    for alpha in ALPHA_LIST:
        prototypes = fuse_prototypes(image_prototypes, text_features, alpha)
        top1_acc, top5_acc = evaluate(test_feats, prototypes, test_samples, class_names)
        results.append({"alpha": alpha, "top1_acc": top1_acc, "top5_acc": top5_acc})
        print(f"{alpha:>8.1f} | {top1_acc:>7.2%} | {top5_acc:>7.2%}")

    # 找出最佳alpha
    best = max(results, key=lambda x: x["top1_acc"])
    print(f"\n🏆 最佳 alpha = {best['alpha']}  (Top-1: {best['top1_acc']:.2%}, Top-5: {best['top5_acc']:.2%})")

    # 保存消融结果，供Day7画图/写报告使用
    with open("food_dataset/alpha_ablation.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "top1_acc", "top5_acc"])
        writer.writeheader()
        writer.writerows(results)
    print("\n✅ 消融结果已保存: food_dataset/alpha_ablation.csv")


if __name__ == "__main__":
    main()