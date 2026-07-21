"""
lora_finetune.py
====================================================
用 LoRA 对 Chinese-CLIP 做轻量微调

思路：
- 冻结原模型全部参数
- 只在注意力层的 Query/Value 投影矩阵旁插入小的低秩矩阵(LoRA)
- 训练目标：让模型在你的30类中餐训练集上，图文匹配得更准
  (对比学习loss，和CLIP预训练时的目标一致，只是数据换成了你的中餐数据)
- 训练完在测试集上评估，对比 zero-shot 基线
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
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from peft import LoraConfig, get_peft_model

# ========== 配置区 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS_CSV = "food_dataset/labels_cleaned.csv"
SELECTED_CLASSES_JSON = "selected_classes.json"
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
BEST_TEMPLATE = "一道美味的{}"

# LoRA超参数
LORA_R = 8              # 低秩矩阵的秩，越大参数越多、表达力越强，也越容易过拟合
LORA_ALPHA = 16          # 缩放系数，习惯上设为 2倍的r
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj", "query", "value"]  # 视觉+文本，各自的Q/V层

# 训练超参数
EPOCHS = 5
BATCH_SIZE = 16          # 训练比推理更吃显存，GPU不够可以调小
LEARNING_RATE = 1e-4
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------
# 数据集定义
# ---------------------------------------------------

class FoodDataset(Dataset):
    """
    每个样本返回：(图片, 对应的文本描述)
    训练目标：让模型学会把"这张图"和"这句话"在向量空间里拉近
    """
    def __init__(self, samples, pinyin_to_cn, template):
        self.samples = samples
        self.pinyin_to_cn = pinyin_to_cn
        self.template = template

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        text = self.template.format(self.pinyin_to_cn[s["class"]])
        return image, text


def collate_fn(batch, processor):
    """
    把一个batch的(图片,文本)对，统一处理成模型需要的张量格式。
    自定义collate_fn是因为PIL图片对象不能直接被默认的collate方法拼接，
    需要先交给processor统一转换。
    """
    images, texts = zip(*batch)
    inputs = processor(
        images=list(images),
        text=list(texts),
        return_tensors="pt",
        padding=True
    )
    return inputs


# ---------------------------------------------------
# 数据准备
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
# LoRA 训练核心
# ---------------------------------------------------

def setup_lora_model(model_name, target_modules, r, alpha, dropout):
    """
    加载原始Chinese-CLIP，包装成LoRA模型：
    - 原模型全部参数冻结(peft库自动处理)
    - 只在target_modules指定的层旁插入可训练的低秩矩阵
    """
    base_model = ChineseCLIPModel.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )

    lora_model = get_peft_model(base_model, lora_config)

    # 打印可训练参数占比，直观感受LoRA"轻量"在哪里
    lora_model.print_trainable_parameters()

    return lora_model


def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    """
    CLIP的对比学习损失函数：
    - 一个batch里有N对图文，构建N×N的相似度矩阵
    - 对角线(匹配的图文对)是正样本，应该相似度最高
    - 非对角线(不匹配的图文对)是负样本，应该相似度低
    - 分别从"图像看文本"和"文本看图像"两个方向计算交叉熵损失，取平均
    这和CLIP预训练时用的损失函数完全一致，只是这里数据换成了你的中餐数据。
    """
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_embeds @ text_embeds.T
    logits_per_text = logits_per_image.T

    # 正确答案就是对角线：第i张图对应第i句话
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
        # ChineseCLIPModel forward会自动算出 image_embeds 和 text_embeds
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
# 评估部分（复用之前验证过的稳定写法）
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

            # 注意：peft包装后的模型，底层模块访问需要加 .base_model.model 前缀
            # 这里改用get_image_features，配合extract_features兜底，更简洁稳妥
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
    """用微调后的模型，重新跑一遍zero-shot评估，检验LoRA微调的效果"""
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
    print(f"训练集: {len(train_samples)} 张 | 测试集: {len(test_samples)} 张\n")

    processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)

    print("构建LoRA模型...")
    model = setup_lora_model(MODEL_NAME, TARGET_MODULES, LORA_R, LORA_ALPHA, LORA_DROPOUT)
    model = model.to(DEVICE)

    # 训练前先跑一次zero-shot评估，作为"微调前"基线对照（应该和Day4结果接近）
    print("\n[微调前] 评估当前模型(应接近zero-shot基线)...")
    top1_before, top5_before = evaluate_zero_shot(
        model, processor, test_samples, class_names, pinyin_to_cn, BEST_TEMPLATE, DEVICE
    )
    print(f"微调前: Top-1 {top1_before:.2%}  Top-5 {top5_before:.2%}")

    # 准备训练数据
    train_dataset = FoodDataset(train_samples, pinyin_to_cn, BEST_TEMPLATE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化LoRA新增的参数
        lr=LEARNING_RATE
    )

    print(f"\n开始LoRA训练，共 {EPOCHS} 个epoch...")
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)

    print("\n[微调后] 评估最终模型...")
    top1_after, top5_after = evaluate_zero_shot(
        model, processor, test_samples, class_names, pinyin_to_cn, BEST_TEMPLATE, DEVICE
    )
    print(f"\n{'=' * 50}")
    print(f"📊 LoRA 微调结果对比:")
    print(f"微调前: Top-1 {top1_before:.2%}  Top-5 {top5_before:.2%}")
    print(f"微调后: Top-1 {top1_after:.2%}  Top-5 {top5_after:.2%}")
    print(f"Top-1 变化: {(top1_after - top1_before) * 100:+.2f} 个百分点")
    print(f"{'=' * 50}")

    # 保存LoRA权重(只保存新增的小矩阵，文件很小，不是整个模型)
    model.save_pretrained("food_dataset/lora_weights")
    print("\n✅ LoRA权重已保存: food_dataset/lora_weights")


if __name__ == "__main__":
    main()