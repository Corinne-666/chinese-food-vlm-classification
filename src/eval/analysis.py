"""
Task4: 分类结果分析与可视化
基于 Day3-Day4 实验产出的预测结果，完成：
  1. 混淆矩阵绘制 + 易混淆类别对提取
  2. 高置信度失败案例可视化（≥5组）
  3. 按菜系分层的准确率分析
  4. 每类别 precision/recall/f1 明细表

使用前提：你需要先把某一版模型（建议用效果最好的 LoRA 微调版本）
在测试集上的预测结果保存成一个 CSV，包含以下四列：
    image_path, true_label, pred_label, confidence
"""

import pandas as pd
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 0. 输入数据
# ============================================================
RESULTS_CSV = "results_lora_augmented.csv"  # 改成你实际保存的预测结果路径
results_df = pd.read_csv(RESULTS_CSV)

CLASS_NAMES = sorted(results_df["true_label"].unique().tolist())

# ============================================================
# 1. 菜系映射表（按你数据集里实际的30个类别名补全，示例见下）
# ============================================================
CUISINE_MAP = {
    # 川菜
    "麻婆豆腐": "川菜", "干煸豆角": "川菜", "口水鸡": "川菜", "辣子鸡丁": "川菜","宫保鸡丁":"川菜","毛血旺":"川菜",
    # 粤菜
    "蒸蛋羹": "粤菜", "烧鸭": "粤菜", "白切鸡": "粤菜","梅菜扣肉":"粤菜","菠萝咕噜肉":"粤菜",
    # 鲁菜
    "糖渍番茄": "鲁菜", "拔丝地瓜":"鲁菜","烧腊": "鲁菜", "卤牛肉": "鲁菜",
    # 徽菜
    "甲鱼汤": "徽菜",
    # 东北菜
    "土豆炖豆角":"东北菜","炸蘑菇":"东北菜","锅包肉":"东北菜",
    # 湘菜
    "苦瓜酿肉":"湘菜","辣椒炒肉":"湘菜","剁椒鱼头":"湘菜",
    # 家常菜
    "番茄炒蛋":"家常菜","可乐鸡翅":"家常菜","番茄牛腩":"家常菜",
    # 沪菜
    "红烧肉":"沪菜",
    # 浙菜
    "糖醋里脊":"浙菜",
    # 朝鲜族腌菜
    "辣白菜":"朝鲜族腌菜",
    # 陕西小吃
    "肉夹馍":"陕西小吃",
    # 客家菜
    "酿辣椒":"客家菜"
}

def get_cuisine(dish_name: str) -> str:
    return CUISINE_MAP.get(dish_name, "未分类")

results_df["true_cuisine"] = results_df["true_label"].map(get_cuisine)
results_df["pred_cuisine"] = results_df["pred_label"].map(get_cuisine)

# ============================================================
# 2. 混淆矩阵
# ============================================================
def plot_confusion_matrix(df, class_names, save_path="confusion_matrix.png", normalize=True):
    y_true, y_pred = df["true_label"], df["pred_label"]
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_show, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={"label": "比例" if normalize else "样本数"})
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("中餐食物分类混淆矩阵")
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()
    return cm  # 返回未归一化的原始计数矩阵，供后面统计用

cm = plot_confusion_matrix(results_df, CLASS_NAMES)

def top_confused_pairs(cm, class_names, top_k=10):
    pairs = []
    for i, ci in enumerate(class_names):
        row_total = cm[i].sum()
        for j, cj in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                pairs.append((ci, cj, cm[i, j], cm[i, j] / row_total))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]

confused_pairs = top_confused_pairs(cm, CLASS_NAMES)
print("最容易混淆的类别对 (真实 -> 预测, 样本数, 占该类比例):")
for true_c, pred_c, n, ratio in confused_pairs:
    print(f"  {true_c} -> {pred_c}: {n}张, {ratio:.1%}")

# ============================================================
# 3. 失败案例可视化（挑高置信度的错判——"自信地错"最值得分析）
# ============================================================
def get_failure_cases(df, n=8):
    wrong = df[df["true_label"] != df["pred_label"]].copy()
    return wrong.sort_values("confidence", ascending=False).head(n)

failure_cases = get_failure_cases(results_df, n=15)
print("\n高置信度错误案例:")
print(failure_cases[["image_path", "true_label", "pred_label", "confidence"]].to_string(index=False))

def visualize_failure_cases(cases_df, save_path="failure_cases.png"):
    n = len(cases_df)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    for idx, (_, row) in enumerate(cases_df.iterrows()):
        img = plt.imread(row["image_path"])
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"真实: {row['true_label']}\n预测: {row['pred_label']} ({row['confidence']:.2f})",
            fontsize=10,
        )
        axes[idx].axis("off")
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()

visualize_failure_cases(failure_cases)

# ============================================================
# 4. 按菜系分层分析
# ============================================================
def cuisine_level_report(df):
    rows = []
    for cuisine, group in df.groupby("true_cuisine"):
        acc = (group["true_label"] == group["pred_label"]).mean()
        rows.append({"菜系": cuisine, "样本数": len(group), "Top-1准确率": acc})
    return pd.DataFrame(rows).sort_values("Top-1准确率", ascending=False)

cuisine_df = cuisine_level_report(results_df)
print("\n分菜系准确率:")
print(cuisine_df.to_string(index=False))

def plot_cuisine_accuracy(cuisine_df, save_path="cuisine_accuracy.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cuisine_df["菜系"], cuisine_df["Top-1准确率"], color="#4C72B0")
    ax.set_ylabel("Top-1 准确率"); ax.set_title("各菜系分类准确率对比")
    ax.set_ylim(0, 1)
    for i, v in enumerate(cuisine_df["Top-1准确率"]):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()

plot_cuisine_accuracy(cuisine_df)

# ============================================================
# 5. 每类详细指标
# ============================================================
report = classification_report(
    results_df["true_label"], results_df["pred_label"],
    labels=CLASS_NAMES, output_dict=True, zero_division=0,
)
report_df = pd.DataFrame(report).T
report_df.to_csv("per_class_report.csv", encoding="utf-8-sig")
print("\n每类详细指标已保存至 per_class_report.csv")
print("\n生成的图表: confusion_matrix.png / failure_cases.png / cuisine_accuracy.png")

# ============================================================
# 6.对比正确样本置信度和错误样本置信度
# ============================================================
correct_conf = results_df[results_df["true_label"] == results_df["pred_label"]]["confidence"]
wrong_conf = results_df[results_df["true_label"] != results_df["pred_label"]]["confidence"]
print("正确样本平均置信度:", correct_conf.mean())
print("错误样本平均置信度:", wrong_conf.mean())
print("正确样本数:", len(correct_conf), " 错误样本数:", len(wrong_conf))