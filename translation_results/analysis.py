import os
import re
import csv
from collections import defaultdict

# Root directory
root_dir = "./"

# 存放结果 {task_key: {"bleu": [], "em": [], "codebleu": []}}
results = defaultdict(lambda: {"bleu": [], "em": [], "codebleu": []})

# Simple Regular Expression Matching
bleu_pattern = re.compile(r"BLEU_Smooth:\s*{\s*'bleu': ([0-9.]+)")
em_pattern = re.compile(r"EM:\s*{\s*'em_ratio': ([0-9.]+)")
codebleu_pattern = re.compile(r"CodeBleu:\s*{\s*'codebleu': ([0-9.]+)")

# Iterate through each subfolder
for dirpath, dirnames, filenames in os.walk(root_dir):
    if "output.txt" not in filenames:
        continue

    output_path = os.path.join(dirpath, "output.txt")
    parent_folder = os.path.basename(dirpath)

    # Parsing folder names, such as qwen2.5-7b_Cs2Java_cot_1-shot_0
    parts = parent_folder.rsplit("_", 4)
    if len(parts) < 4:
        continue

    model, task, method, shot = parts[0], parts[1], parts[2], parts[3]

    if model in ("qwen2.5-32b"):
        continue

    task_key = f"{model}_{task}_{method}_{shot}"

    with open(output_path, "r", encoding="utf-8") as f:
        text = f.read()

        # BLEU → Take only the last one
        bleu_vals = bleu_pattern.findall(text)
        if bleu_vals:
            results[task_key]["bleu"].append(float(bleu_vals[-1]))

        # EM → Take only the last one
        em_vals = em_pattern.findall(text)
        if em_vals:
            results[task_key]["em"].append(float(em_vals[-1]))

        # CodeBleu → 只取最后一个
        codebleu_vals = codebleu_pattern.findall(text)
        if codebleu_vals:
            results[task_key]["codebleu"].append(float(codebleu_vals[-1]))

# Save the results to output.txt
# with open("output.txt", "w", encoding="utf-8") as fout:
#     for task_key, metrics in results.items():
#         fout.write(f"Task: {task_key}\n")
#         for metric_name, values in metrics.items():
#             if values:
#                 fout.write(f"  {metric_name}: min={min(values):.4f}, max={max(values):.4f}\n")
#         fout.write("\n")

# Save the results to output.txt
with open("output.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # 先对 task_key 排序
    sorted_keys = sorted(results.keys(), key=lambda x: x.split("_"))

    for task_key in sorted_keys:
        metrics = results[task_key]
        model, task, method, shot = task_key.split("_")
        row = [model, task, method, shot]

        # 按顺序添加三个指标及其 min/max
        for metric_name in ["bleu", "em", "codebleu"]:
            values = metrics.get(metric_name, [])
            if values:
                row.extend([metric_name, f"{min(values):.4f}", f"{max(values):.4f}"])
            else:
                row.extend([metric_name, "", ""])  # 如果没有该指标，留空

        writer.writerow(row)

print("结果已保存到 output.txt 和 output.csv")