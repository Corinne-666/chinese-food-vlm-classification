"""
few_shot_v1_image_only.py
====================================================
最朴素的 Few-shot 方法：纯图像原型法 (Image Prototype)

核心思路：
1. 对每个类别，从训练集里随机抽 N_SHOTS 张图片
2. 用 CLIP 的图像编码器把这些图片转成向量(embedding)
3. 把这 N_SHOTS 个向量取平均，得到这个类别的"代表向量"(原型 prototype)
4. 测试时，把测试图片编码成向量，和每个类别的原型比较余弦相似度
   相似度最高的类别，就是预测结果

====================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 避免Windows下OpenMP库冲突报错
os.environ["HF_HUB_OFFLINE"] = "1"            # 模型已下载过，离线加载更快

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
N_SHOTS = 10        # 每类用几张图构建原型
BATCH_SIZE = 32     # 每批处理多少张图(显存/内存不够可以调小)
SEED = 42           # 固定随机种子，保证每次抽样结果一致，实验可复现

random.seed(SEED)


# ---------------------------------------------------
# 数据准备部分
# ---------------------------------------------------

def load_samples(labels_csv):
    """
    从 labels.csv 里按 split 字段，分别取出 train 和 test 两部分样本。
    train 用来抽取 few-shot 支持集(构建原型)，test 用来最终评估效果。
    """
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
    """
    读取类别信息文件，返回：
    - class_names: 排好序的类别拼音名列表(固定顺序，保证后面索引对应关系不会错乱)
    - pinyin_to_cn: 拼音名 -> 中文名 的字典，方便打印结果时显示中文
    """
    with open(selected_classes_json, "r", encoding="utf-8") as f:
        selected_classes = json.load(f)
    pinyin_to_cn = {info["name"]: info["cn_name"] for info in selected_classes.values()}
    class_names = sorted(pinyin_to_cn.keys())
    return class_names, pinyin_to_cn


def sample_few_shot(train_samples, class_names, n_shots):
    """
    从训练集里，给每个类别随机抽 n_shots 张图片，组成"few-shot支持集"。
    这些图片就是模型"学习"每个类别长什么样时唯一能看到的样本。
    """
    # 先把训练样本按类别分组，方便后面按类抽样
    by_class = {c: [] for c in class_names}
    for s in train_samples:
        if s["class"] in by_class:
            by_class[s["class"]].append(s)

    few_shot_samples = {}
    for c in class_names:
        pool = by_class[c]
        # min(n_shots, len(pool)) 是防御性写法：
        # 万一某个类别训练图片总数不足n_shots张，就有多少抽多少，不会报错
        few_shot_samples[c] = random.sample(pool, min(n_shots, len(pool)))
    return few_shot_samples


# ---------------------------------------------------
# 核心：图像编码部分（这是本次修复的重点）
# ---------------------------------------------------

def encode_images(model, processor, image_paths, device, batch_size=32):
    """
    批量把一组图片编码成CLIP的图像特征向量。

    重要说明：这里没有直接调用 model.get_image_features()，
    而是手动拆成两步：
        1) model.vision_model(...)   得到图像编码器的池化输出(pooler_output)
        2) model.visual_projection(...) 把它投影到和文本共享的向量空间

    这样写的原因：
    在某些transformers版本里，get_image_features()对Chinese-CLIP的封装不稳定，
    有时返回的是"投影后的向量"(我们想要的)，有时返回的是"投影前的原始向量"，
    维度会不一致(512 vs 768)，导致后面图像向量和文本/其他图像向量维度对不上。
    直接调用底层的 vision_model + visual_projection，可以保证每次拿到的
    都是同样处理流程、同样维度的向量，避免这种不稳定性。

    返回：L2归一化后的特征矩阵，形状 [图片总数, 向量维度]
    """
    all_feats = []
    with torch.no_grad():  # 推理阶段不需要计算梯度，节省显存、加快速度
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            # processor 负责把PIL图片转成模型需要的张量格式(resize、归一化像素值等)
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}  # 搬到GPU/CPU

            # 第1步：过图像编码器，得到池化后的特征(pooler_output)
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_output = vision_outputs.pooler_output  # 形状 [batch, hidden_dim]

            # 第2步：过投影层，映射到和文本共享的向量空间
            image_embeds = model.visual_projection(pooled_output)  # 形状 [batch, projection_dim]

            # L2归一化：把每个向量的长度缩放为1，这样点积就等于余弦相似度
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            all_feats.append(image_embeds.cpu())  # 挪回CPU，减少显存占用

    return torch.cat(all_feats, dim=0)


def build_image_prototypes(model, processor, few_shot_samples, class_names, device):
    """
    核心逻辑：为每个类别构建"图像原型"。

    做法：把该类别 few-shot support set 里所有图片的向量取平均，
    得到一个能"代表这个类别长什么样"的向量。

    直觉理解：如果给模型看10张宫保鸡丁的照片，虽然每张角度、光线不同，
    但它们的平均向量会更稳定地落在"宫保鸡丁"在向量空间里的大致位置，
    比单独一张照片更能代表这个类别的典型特征。
    """
    prototypes = []
    for c in class_names:
        paths = [s["image_path"] for s in few_shot_samples[c]]
        img_feats = encode_images(model, processor, paths, device)  # [N_SHOTS, D]

        prototype = img_feats.mean(dim=0)          # 按样本维度取平均 -> [D]
        prototype = prototype / prototype.norm()    # 平均之后向量长度会变化，重新归一化

        prototypes.append(prototype)

    # 把所有类别的原型向量堆叠成一个矩阵，形状固定为 [类别数, 向量维度]
    prototypes = torch.stack(prototypes, dim=0)

    # 防御性检查：打印形状，确认是预期的二维矩阵，方便及时发现类似本次遇到的维度问题
    print(f"[检查] 原型矩阵形状: {prototypes.shape}  (应为 [类别数={len(class_names)}, 向量维度])")

    return prototypes


# ---------------------------------------------------
# 评估部分
# ---------------------------------------------------

def evaluate(model, processor, test_samples, prototypes, class_names, device):
    """
    在测试集上评估：每张测试图和所有类别原型比相似度，取最相似的作为预测结果。
    """
    image_paths = [s["image_path"] for s in test_samples]
    test_feats = encode_images(model, processor, image_paths, device)  # [N, D]

    # 防御性检查：确认测试图特征和原型矩阵的向量维度(D)一致，
    # 这一步就是上次报错的根源所在，提前打印出来能第一时间发现问题
    print(f"[检查] 测试图特征形状: {test_feats.shape}")
    print(f"[检查] 原型矩阵形状: {prototypes.shape}")
    assert test_feats.shape[1] == prototypes.shape[1], \
        f"维度不匹配！测试图向量维度={test_feats.shape[1]}，原型向量维度={prototypes.shape[1]}"

    # 因为两边向量都已经L2归一化，矩阵乘法的结果就等价于余弦相似度
    # test_feats: [N, D]，prototypes: [C, D]，转置后 [D, C]，乘出来是 [N, C]
    similarities = test_feats @ prototypes.transpose(0, 1)
    probs = similarities.softmax(dim=-1)

    top1_correct, top5_correct = 0, 0
    per_sample_results = []

    for idx, s in enumerate(test_samples):
        # 按相似度从高到低排序，得到这张图对所有类别的预测排名
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
    model.eval()  # 切换到推理模式(关闭dropout等训练专用行为)

    few_shot_samples = sample_few_shot(train_samples, class_names, N_SHOTS)
    print(f"每类抽取 {N_SHOTS} 张作为 few-shot 支持集\n")

    print("构建纯图像原型...")
    prototypes = build_image_prototypes(model, processor, few_shot_samples, class_names, DEVICE)

    print("\n在测试集上评估...")
    top1_acc, top5_acc, results = evaluate(model, processor, test_samples, prototypes, class_names, DEVICE)

    print(f"\n{'=' * 50}")
    print(f"📊 Few-shot v1 (纯图像原型法) 结果:")
    print(f"N_SHOTS = {N_SHOTS}")
    print(f"Top-1: {top1_acc:.2%}  Top-5: {top5_acc:.2%}")
    print(f"{'=' * 50}")
    print(f"\n对比 Day4 zero-shot 最佳结果: Top-1 85.12%  Top-5 97.85%")
    print(f"Top-1 变化: {(top1_acc - 0.8512) * 100:+.2f} 个百分点")
    print(f"Top-5 变化: {(top5_acc - 0.9785) * 100:+.2f} 个百分点")

    # 保存逐条预测结果，方便后续和 zero-shot、v2融合版本做对比分析
    with open("food_dataset/fewshot_v1_result.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "true_class", "pred_top1",
                                                  "correct_top1", "correct_top5"])
        writer.writeheader()
        writer.writerows(results)
    print("\n✅ 逐条结果已保存: food_dataset/fewshot_v1_result.csv")


if __name__ == "__main__":
    main()