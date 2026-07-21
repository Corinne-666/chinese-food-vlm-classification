"""
zero_shot_demo.py
用 Chinese-CLIP 在小样本(5类×5张)上跑通 zero-shot 分类流程，验证推理逻辑
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"
import random
import json
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# ========== 配置 ==========
SPLIT_TEST_DIR = "food_dataset/split/test"
SELECTED_CLASSES_JSON = "selected_classes.json"
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
N_CLASSES = 3
N_PER_CLASS = 10
SEED = 42
PROMPT_TEMPLATE = "一张{}的照片"


def load_class_mapping(json_path):
    """读取类别拼音名 -> 中文名 的映射"""
    with open(json_path, "r", encoding="utf-8") as f:
        selected_classes = json.load(f)
    return {info["name"]: info["cn_name"] for info in selected_classes.values()}


def sample_test_data(test_dir, pinyin_to_cn, n_classes, n_per_class, seed):
    """从测试集里随机抽 n_classes 个类别，每类抽 n_per_class 张图"""
    random.seed(seed)

    all_classes = [d for d in os.listdir(test_dir)
                   if os.path.isdir(os.path.join(test_dir, d))]
    sample_classes =["lajiaochaorou","meicaikourou","tangculiji"]
    print("抽中的类别:", [pinyin_to_cn.get(c, c) for c in sample_classes])

    test_samples = []
    for cls in sample_classes:
        cls_dir = os.path.join(test_dir, cls)
        imgs = os.listdir(cls_dir)
        sampled_imgs = random.sample(imgs, min(n_per_class, len(imgs)))
        for img_name in sampled_imgs:
            test_samples.append({
                "image_path": os.path.join(cls_dir, img_name),
                "true_class": cls,
                "true_cn_name": pinyin_to_cn.get(cls, cls)
            })

    print(f"共抽取 {len(test_samples)} 张测试图片\n")
    return sample_classes, test_samples


def load_model(model_name):
    """加载 Chinese-CLIP 模型和处理器"""
    print("正在加载模型...")
    model = ChineseCLIPModel.from_pretrained(model_name)
    processor = ChineseCLIPProcessor.from_pretrained(model_name)
    model.eval()
    print("模型加载完成\n")
    return model, processor


def run_zero_shot_inference(model, processor, test_samples, sample_classes,
                              pinyin_to_cn, prompt_template):
    """对每张测试图跑 zero-shot 推理，返回结果列表"""
    candidate_cn_names = [pinyin_to_cn.get(c, c) for c in sample_classes]
    candidate_texts = [prompt_template.format(name) for name in candidate_cn_names]

    print("候选类别文本:")
    for t in candidate_texts:
        print(" -", t)
    print()

    results = []
    with torch.no_grad():
        for sample in test_samples:
            image = Image.open(sample["image_path"]).convert("RGB")

            inputs = processor(text=candidate_texts, images=image,
                                return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]

            sorted_indices = probs.argsort(descending=True).tolist()
            top1_pred = sample_classes[sorted_indices[0]]
            top5_preds = [sample_classes[i] for i in sorted_indices]

            is_correct_top1 = (top1_pred == sample["true_class"])
            is_correct_top5 = (sample["true_class"] in top5_preds[:5])

            results.append({
                "image_path": sample["image_path"],
                "true_class": sample["true_cn_name"],
                "pred_top1": pinyin_to_cn.get(top1_pred, top1_pred),
                "correct_top1": is_correct_top1,
                "correct_top5": is_correct_top5,
            })

            status = "✅" if is_correct_top1 else "❌"
            print(f"{status} 真实:{sample['true_cn_name']:8s} "
                  f"预测:{pinyin_to_cn.get(top1_pred, top1_pred):8s} "
                  f"置信度:{probs[sorted_indices[0]]:.3f}")

    return results


def compute_accuracy(results):
    """计算 Top-1 / Top-5 准确率"""
    top1_acc = sum(r["correct_top1"] for r in results) / len(results)
    top5_acc = sum(r["correct_top5"] for r in results) / len(results)
    print(f"\n📊 小样本测试结果 (仅{len(results)}张，仅供验证流程):")
    print(f"Top-1 准确率: {top1_acc:.2%}")
    print(f"Top-5 准确率: {top5_acc:.2%}")
    return top1_acc, top5_acc


def main():
    pinyin_to_cn = load_class_mapping(SELECTED_CLASSES_JSON)
    sample_classes, test_samples = sample_test_data(
        SPLIT_TEST_DIR, pinyin_to_cn, N_CLASSES, N_PER_CLASS, SEED
    )
    model, processor = load_model(MODEL_NAME)
    results = run_zero_shot_inference(
        model, processor, test_samples, sample_classes,
        pinyin_to_cn, PROMPT_TEMPLATE
    )
    compute_accuracy(results)


if __name__ == "__main__":
    main()