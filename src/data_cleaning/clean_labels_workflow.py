"""
clean_labels_workflow.py
====================================================
用途：
  1. 把预测结果里的265个错判样本，和 labels.csv 按 image_path 关联，
     定位到它们在 labels.csv 里的具体行号(row_id)，导出成一份人工复核清单。
  2. 复核完、在清单里填好"修正后的正确类别"之后，用这份清单批量
     回写更新 labels.csv，生成清洗后的新版本。
     如果发现某个样本不属于任何一类(比如拼图、构图异常、内容不相关)，
     在 corrected_class 列填 "DELETE"，该样本会被整行从清洗后的
     labels.csv 里移除(图片文件本身不受影响，仍保留在磁盘上)。

两步分开跑：先 export，人工在Excel/表格软件里把清单填完，再 apply。
====================================================
"""

import pandas as pd

LABELS_CSV = "food_dataset/labels.csv"
RESULTS_CSV = "food_dataset/results_lora_augmented.csv"
REVIEW_CSV = "food_dataset/wrong_samples_for_review.csv"
CLEANED_LABELS_CSV = "food_dataset/labels_cleaned.csv"


# ---------------------------------------------------
# 第一步：导出待复核清单（带 labels.csv 里的行号）
# ---------------------------------------------------
def export_wrong_samples_for_review():
    labels_df = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    results_df = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")

    # 给 labels.csv 加一列行号，方便你在原文件里定位/核对
    labels_df = labels_df.reset_index().rename(columns={"index": "row_id"})

    wrong_df = results_df[results_df["true_label"] != results_df["pred_label"]].copy()
    print(f"错误样本数: {len(wrong_df)}")

    # 按 image_path 关联，拿到每条错判样本在 labels.csv 里的 row_id 和原始 class 字段
    merged = wrong_df.merge(labels_df, on="image_path", how="left")

    # 检查有没有关联不上的样本（image_path 在 labels.csv 里找不到，路径格式不一致导致）
    unmatched = merged[merged["row_id"].isna()]
    if len(unmatched) > 0:
        print(f"⚠️ 警告: 有 {len(unmatched)} 条样本按 image_path 关联不上 labels.csv，"
              f"请检查两个CSV里 image_path 的格式是否一致（比如相对路径 vs 绝对路径、正斜杠 vs 反斜杠）")

    # 加一列空的"修正后类别"，留给你人工填
    merged["corrected_class"] = ""

    out_cols = ["row_id", "image_path", "true_label", "pred_label", "confidence",
                "class", "split", "corrected_class"]
    out_cols = [c for c in out_cols if c in merged.columns]  # 防止某些列名不存在报错
    merged[out_cols].to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    print(f"✅ 待复核清单已保存至: {REVIEW_CSV}  (共 {len(merged)} 条)")
    print("下一步: 打开这份CSV，对每一行判断真实类别是什么，")
    print("  - 如果确认是标注错误，把正确的类别名(pinyin，和 labels.csv 的 class 列格式一致)填进 corrected_class 列；")
    print("  - 如果确认样本不属于任何一类(拼图/构图异常/内容不相关)，corrected_class 填 'DELETE'；")
    print("  - 如果确认是真实视觉难点/模型识别错误，corrected_class 留空即可，后面会自动跳过。")


# ---------------------------------------------------
# 第二步：用填好的复核清单，回写更新 labels.csv
# corrected_class 支持两种填法：
#   - 填具体类别名(pinyin) -> 修正该行的 class 字段
#   - 填 "DELETE"          -> 该样本不属于任何一类，整行从清洗后的labels里删掉
# ---------------------------------------------------
DELETE_MARK = "DELETE"

def apply_corrections():
    labels_df = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")

    # Excel/WPS在中文Windows上默认把CSV另存为GBK编码，
    # 这里做个兼容：先按utf-8-sig读，失败了自动retry用gb18030读
    try:
        review_df = pd.read_csv(REVIEW_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError:
        print("⚠️ 未按UTF-8成功解析review清单，尝试按GBK/GB18030重新读取"
              "（说明这份CSV是被Excel用默认编码保存过的，之后建议另存为时手动选'CSV UTF-8'）")
        review_df = pd.read_csv(REVIEW_CSV, encoding="gb18030")

    to_process = review_df[review_df["corrected_class"].notna() & (review_df["corrected_class"] != "")]
    print(f"待处理样本数: {len(to_process)}")

    fixed_count, deleted_count, not_found = 0, 0, 0
    paths_to_delete = []

    for _, row in to_process.iterrows():
        mask = labels_df["image_path"] == row["image_path"]
        if mask.sum() == 0:
            print(f"⚠️ 未找到对应行，跳过: {row['image_path']}")
            not_found += 1
            continue

        if str(row["corrected_class"]).strip().upper() == DELETE_MARK:
            paths_to_delete.append(row["image_path"])
            deleted_count += 1
        else:
            labels_df.loc[mask, "class"] = row["corrected_class"]
            fixed_count += 1

    if paths_to_delete:
        labels_df = labels_df[~labels_df["image_path"].isin(paths_to_delete)]

    labels_df.to_csv(CLEANED_LABELS_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已修正标注 {fixed_count} 条，删除无效样本 {deleted_count} 条"
          f"{f'，未找到 {not_found} 条' if not_found else ''}")
    print(f"清洗后的新版 labels 已保存至: {CLEANED_LABELS_CSV}")
    print("检查无误后，可以把它改名替换/指向原来的 labels.csv 路径，用于后续重新评估或重新训练。")
    if deleted_count:
        print(f"\n⚠️ 注意: 被删除的 {deleted_count} 条样本只是从 labels.csv 里移除了引用，"
              f"图片文件本身还在磁盘上，不会被自动删除，不影响使用。")


if __name__ == "__main__":
    # 先只运行第一步。填完 corrected_class 列之后，注释掉这行、改跑 apply_corrections()
    #export_wrong_samples_for_review()
    apply_corrections()