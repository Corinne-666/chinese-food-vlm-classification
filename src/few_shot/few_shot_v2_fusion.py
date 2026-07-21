"""
few_shot_v2_fusion.py
====================================================
在 v1(纯图像原型) 基础上的改进版：图文融合原型 (Image-Text Fusion Prototype)

背景：
v1 用 few-shot 图像的平均特征作为类别原型，Top-1达到88.30%，
比zero-shot(84.28%)提升了4.02个百分点，但还没有稳定超过5个百分点的目标线。

改进思路：
纯图像原型完全依赖那 N_SHOTS 张抽样图片，如果这几张图片本身有噪声
(比如角度奇怪、光线差、被遮挡)，平均出来的原型就会不够准确。
文本描述("一道美味的{类别名}")虽然不如图像精确，但它是稳定的、
不会因为抽样运气不好而跑偏。

所以这一版把两种信息按权重融合：
    最终原型 = alpha × 图像原型 + (1-alpha) × 文本特征
用图像信息提供"具体视觉细节"，用文本信息提供"稳定的语义锚点"，
两者取长补短。
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
BEST_TEMPLATE = "一道美味的{}"   # Day4实验中表现最好的prompt模板
N_SHOTS = 10                     # 和v1保持一致，方便直接对比融合带来的提升
ALPHA = 0.5                      # 图像原型权重，(1-ALPHA)是文本特征权重
BATCH_SIZE = 32
SEED = 42

random.seed(SEED)


# ---------------------------------------------------
# 数据准备部分（和v1完全一致）
# ---------------------------------------------------

def load_samples(labels_csv):
    """从labels.csv里分别取出train和test两部分样本"""
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
    """返回排序后的类别拼音名列表，以及拼音名->中文名字典"""
    with open(selected_classes_json, "r", encoding="utf-8") as f:
        selected_classes = json.load(f)
    pinyin_to_cn = {info["name"]: info["cn_name"] for info in selected_classes.values()}
    class_names = sorted(pinyin_to_cn.keys())
    return class_names, pinyin_to_cn


def sample_few_shot(train_samples, class_names, n_shots):
    """每类随机抽n_shots张，作为few-shot支持集"""
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
# 编码部分：图像和文本，都用手动调底层模块的稳定写法
# ---------------------------------------------------

def encode_images(model, processor, image_paths, device, batch_size=32):
    """
    批量编码图片。写法和v1一致：手动调用 vision_model + visual_projection，
    不用行为不稳定的 get_image_features()，保证向量维度始终一致。
    返回L2归一化后的特征矩阵 [N, D]
    """
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
    """
    批量编码文本。这里沿用Day4(zero_shot_full_eval.py)中验证过可行的写法：
    调用 model.get_text_features()，再用 extract_features() 做兼容性兜底。
    
    (注：图像编码那边get_image_features()不稳定，所以v1/v2都改成了手动调用
    vision_model+visual_projection；但文本编码在Day4已验证get_text_features()
    是可靠的，这里不需要额外改动，直接复用即可)
    """
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        raw_output = model.get_text_features(**inputs)
        text_embeds = extract_features(raw_output)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    return text_embeds.cpu()


def extract_features(output):
    """兼容不同transformers版本的返回格式，统一取出特征张量"""
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


# ---------------------------------------------------
# 核心：构建图文融合原型
# ---------------------------------------------------

def build_fusion_prototypes(model, processor, few_shot_samples, class_names,
                              pinyin_to_cn, template, device, alpha):
    """
    为每个类别构建"图文融合原型"：
        1. 用该类few-shot图片，算出图像原型(做法同v1)
        2. 用该类最佳prompt模板，算出文本特征
        3. 按 alpha : (1-alpha) 加权融合两者，再归一化

    alpha越接近1，越依赖图像信息(越像v1)；
    alpha越接近0，越依赖文本信息(越像zero-shot)。
    """
    # 第一步：批量算出所有类别的文本特征(一次性算完，比图像编码快很多)
    texts = [template.format(pinyin_to_cn[c]) for c in class_names]
    text_feats = encode_texts(model, processor, texts, device)  # [C, D]

    prototypes = []
    for i, c in enumerate(class_names):
        # 图像原型：和v1做法完全一致
        paths = [s["image_path"] for s in few_shot_samples[c]]
        img_feats = encode_images(model, processor, paths, device)
        img_prototype = img_feats.mean(dim=0)
        img_prototype = img_prototype / img_prototype.norm()

        # 该类别对应的文本特征
        text_prototype = text_feats[i]

        # 加权融合：核心改进点
        fused = alpha * img_prototype + (1 - alpha) * text_prototype
        fused = fused / fused.norm()  # 融合后向量长度会变化，重新归一化保证是单位向量

        prototypes.append(fused)

    prototypes = torch.stack(prototypes, dim=0)
    print(f"[检查] 融合原型矩阵形状: {prototypes.shape}  (应为 [类别数={len(class_names)}, 向量维度])")
    return prototypes


# ---------------------------------------------------
# 评估部分（和v1逻辑一致，只是喂入的prototypes不同）
# ---------------------------------------------------

def evaluate(model, processor, test_samples, prototypes, class_names, device):
    image_paths = [s["image_path"] for s in test_samples]
    test_feats = encode_images(model, processor, image_paths, device)

    print(f"[检查] 测试图特征形状: {test_feats.shape}")
    print(f"[检查] 原型矩阵形状: {prototypes.shape}")
    assert test_feats.shape[1] == prototypes.shape[1], \
        f"维度不匹配！测试图向量维度={test_feats.shape[1]}，原型向量维度={prototypes.shape[1]}"

    similarities = test_feats @ prototypes.transpose(0, 1)
    probs = similarities.softmax(dim=-1)

    top1_correct, top5_correct = 0, 0
    per_sample_results = []

    for idx, s in enumerate(test_samples):
        sorted_idx = probs[idx].argsort(descending=True).tolist()
        pred_top1 = class_names[sorted_idx[0]]
        pred_top5 = [class_names[i] for i in sorted_idx[:5]]

        is_top1 = (pred_top1 == s["class"])
        is_top5 = (s["class"] in pred_top5)
        top1_correct += is_top1
        top5_correct += is_top5

        per_sample_results.append({
            "image_path": s["image_path"],
            "true_class": s["class"],
            "pred_top1": pred_top1,
            "correct_top1": is_top1,
            "correct_top5": is_top5,
        })

    n = len(test_samples)
    return top1_correct / n, top5_correct / n, per_sample_results


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
    print(f"每类抽取 {N_SHOTS} 张作为 few-shot 支持集")
    print(f"融合权重 alpha={ALPHA} (图像:{ALPHA} / 文本:{1-ALPHA})\n")

    print("构建图文融合原型...")
    prototypes = build_fusion_prototypes(
        model, processor, few_shot_samples, class_names,
        pinyin_to_cn, BEST_TEMPLATE, DEVICE, ALPHA
    )

    print("\n在测试集上评估...")
    top1_acc, top5_acc, results = evaluate(
        model, processor, test_samples, prototypes, class_names, DEVICE
    )

    print(f"\n{'=' * 50}")
    print(f"📊 Few-shot v2 (图文融合原型法) 结果:")
    print(f"N_SHOTS = {N_SHOTS}  |  alpha = {ALPHA}")
    print(f"Top-1: {top1_acc:.2%}  Top-5: {top5_acc:.2%}")
    print(f"{'=' * 50}")
    print(f"\n对比 Day4 zero-shot:        Top-1 85.12%  Top-5 97.85%")
    print(f"对比 Day5 v1(纯图像原型):    Top-1 90.03%  Top-5 99.18%")
    print(f"v2 相比 zero-shot 提升: {(top1_acc - 0.8512) * 100:+.2f} 个百分点")
    print(f"v2 相比 v1     提升: {(top1_acc - 0.9003) * 100:+.2f} 个百分点")

    with open("food_dataset/fewshot_v2_result.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "true_class", "pred_top1",
                                                  "correct_top1", "correct_top5"])
        writer.writeheader()
        writer.writerows(results)
    print("\n✅ 逐条结果已保存: food_dataset/fewshot_v2_result.csv")


if __name__ == "__main__":
    main()