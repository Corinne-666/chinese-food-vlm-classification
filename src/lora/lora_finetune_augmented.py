"""
lora_finetune_augmented.py
====================================================
在LoRA微调基础上，加入训练图像数据增强，对比对泛化性能的影响

设计原则：
除了训练数据的读取方式(增加了增强变换)，其余部分(模型结构、LoRA配置、
训练超参数、评估逻辑)与 lora_finetune.py 完全一致，这样才能干净地
对比出"数据增强"这一个变量带来的效果差异，而不是多个改动混杂在一起。

对比对象: lora_finetune.py 的结果 (微调后 Top-1 93.76%)
====================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"

import csv
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from peft import LoraConfig, get_peft_model

# ========== 配置区 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS_CSV = "food_dataset/labels_cleaned.csv"
SELECTED_CLASSES_JSON = "selected_classes.json"
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
BEST_TEMPLATE = "一道美味的{}"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj", "query", "value"]

EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------
# 数据增强定义（本次实验的核心新增部分）
# ---------------------------------------------------

# 训练时使用的增强流水线：多种变换随机组合，每次读取同一张图都可能产生
# 略微不同的版本，相当于用有限的图片"制造"出更丰富的视觉变体，
# 帮助模型学到更鲁棒、不那么依赖特定角度/光线/构图的特征。
train_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=15),                          # 模拟拍摄角度倾斜
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),       # 模拟远近/构图差异
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 模拟光线差异
    transforms.RandomHorizontalFlip(p=0.5),                          # 随机水平翻转
])


# ---------------------------------------------------
# 数据集定义（相比原版，__getitem__ 里加了增强这一步）
# ---------------------------------------------------

class FoodDatasetAugmented(Dataset):
    """
    与 lora_finetune.py 中的 FoodDataset 唯一区别：
    每次取图片时，先过一遍 train_augmentation 增强流水线，
    再交给 processor 做模型需要的标准化处理。
    """
    def __init__(self, samples, pinyin_to_cn, template, augmentation):
        self.samples = samples
        self.pinyin_to_cn = pinyin_to_cn
        self.template = template
        self.augmentation = augmentation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        image = self.augmentation(image)          # 关键新增：先做增强
        text = self.template.format(self.pinyin_to_cn[s["class"]])
        return image, text


def collate_fn(batch, processor):
    images, texts = zip(*batch)
    inputs = processor(
        images=list(images),
        text=list(texts),
        return_tensors="pt",
        padding=True
    )
    return inputs


# ---------------------------------------------------
# 数据准备（与之前一致）
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
# LoRA 模型构建（与之前一致）
# ---------------------------------------------------

def setup_lora_model(model_name, target_modules, r, alpha, dropout):
    base_model = ChineseCLIPModel.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=target_modules,
        lora_dropout=dropout, bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model


def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    logits_per_image = logit_scale * image_embeds @ text_embeds.T
    logits_per_text = logits_per_image.T
    labels = torch.arange(image_embeds.shape[0], device=image_embeds.device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    for step, inputs in enumerate(dataloader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, return_loss=False)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        logit_scale = model.logit_scale.exp()
        loss = clip_contrastive_loss(image_embeds, text_embeds, logit_scale)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        if step % 20 == 0:
            print(f"  Epoch {epoch} Step {step}/{len(dataloader)}  Loss: {loss.item():.4f}")
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} 平均Loss: {avg_loss:.4f}")
    return avg_loss


# ---------------------------------------------------
# 评估部分（与之前一致）
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
            images = [Image.open(p).convert("RGB") for p in batch_paths]  # 评估时不做增强，用原图
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


def evaluate_zero_shot(model, processor, test_samples, class_names, pinyin_to_cn, template, device):
    texts = [template.format(pinyin_to_cn[c]) for c in class_names]
    text_feats = encode_texts_eval(model, processor, texts, device)
    image_paths = [s["image_path"] for s in test_samples]
    test_feats = encode_images_eval(model, processor, image_paths, device)
    similarities = test_feats @ text_feats.T
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
    print("本次训练启用数据增强: 随机旋转+随机裁剪+颜色抖动+随机水平翻转\n")

    processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)

    print("构建LoRA模型...")
    model = setup_lora_model(MODEL_NAME, TARGET_MODULES, LORA_R, LORA_ALPHA, LORA_DROPOUT)
    model = model.to(DEVICE)

    print("\n[微调前] 评估当前模型...")
    top1_before, top5_before = evaluate_zero_shot(
        model, processor, test_samples, class_names, pinyin_to_cn, BEST_TEMPLATE, DEVICE
    )
    print(f"微调前: Top-1 {top1_before:.2%}  Top-5 {top5_before:.2%}")

    # 关键区别：这里用带增强的Dataset
    train_dataset = FoodDatasetAugmented(train_samples, pinyin_to_cn, BEST_TEMPLATE, train_augmentation)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    print(f"\n开始LoRA训练(带数据增强)，共 {EPOCHS} 个epoch...")
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)

    print("\n[微调后] 评估最终模型...")
    top1_after, top5_after = evaluate_zero_shot(
        model, processor, test_samples, class_names, pinyin_to_cn, BEST_TEMPLATE, DEVICE
    )

    print(f"\n{'=' * 60}")
    print(f"📊 LoRA + 数据增强 结果:")
    print(f"微调前: Top-1 {top1_before:.2%}  Top-5 {top5_before:.2%}")
    print(f"微调后: Top-1 {top1_after:.2%}  Top-5 {top5_after:.2%}")
    print(f"{'=' * 60}")
    print(f"\n对比 无增强的LoRA(上一轮实验): Top-1 93.76%  Top-5 99.56%")
    print(f"数据增强带来的额外变化: {(top1_after - 0.9376) * 100:+.2f} 个百分点")

    model.save_pretrained("food_dataset/lora_weights_augmented")
    print("\n✅ LoRA权重已保存: food_dataset/lora_weights_augmented")


if __name__ == "__main__":
    main()