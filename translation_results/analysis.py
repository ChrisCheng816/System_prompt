import os
import re
import csv
import json
from collections import defaultdict

# Root directory
root_dir = "./"

# 存放结果 {task_key: {"bleu": [], "em": [], "codebleu": []}}
results = defaultdict(lambda: {"bleu": [], "em": [], "codebleu": []})

# Simple Regular Expression Matching
# bleu_pattern = re.compile(r"BLEU_Smooth:\s*{\s*'bleu': ([0-9.]+)")
# em_pattern = re.compile(r"EM:\s*{\s*'em_ratio': ([0-9.]+)")
# codebleu_pattern = re.compile(r"CodeBleu:\s*{\s*'codebleu': ([0-9.]+)")

# Iterate through each subfolder
for dirpath, dirnames, filenames in os.walk(root_dir):
    if "output_cleaned.json" in filenames:
        output_path = os.path.join(dirpath, "output_cleaned.json")
    else:
        output_path = os.path.join(dirpath, "output.json")
    parent_folder = os.path.basename(dirpath)

    # Parsing folder names, such as qwen2.5-7b_Cs2Java_cot_1-shot_0
    parts = parent_folder.rsplit("_", 4)
    if len(parts) < 4:
        continue

    model, task, method, shot = parts[0], parts[1], parts[2], parts[3]

    task_key = f"{model}_{task}_{method}_{shot}"

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        if "BLEU_Smooth" in data and "bleu" in data["BLEU_Smooth"]:
            bleu_val = round(data["BLEU_Smooth"]["bleu"] * 100, 2)
            results[task_key]["bleu"].append(bleu_val)

        # EM → 乘100并四舍五入到两位
        if "EM" in data and "em_ratio" in data["EM"]:
            em_val = round(data["EM"]["em_ratio"] * 100, 2)
            results[task_key]["em"].append(em_val)

        # CodeBleu → 保持原样
        if "CodeBleu" in data and "codebleu" in data["CodeBleu"]:
            results[task_key]["codebleu"].append(float(data["CodeBleu"]["codebleu"]))
            
# Save the results to output.txt
# with open("output.txt", "w", encoding="utf-8") as fout:
#     for task_key, metrics in results.items():
#         fout.write(f"Task: {task_key}\n")
#         for metric_name, values in metrics.items():
#             if values:
#                 fout.write(f"  {metric_name}: min={min(values):.4f}, max={max(values):.4f}\n")
#         fout.write("\n")

# Save the results to output.csv
with open("output.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model","task","method","shot",
                 "bleu","bleu_min","bleu_max",
                 "em","em_min","em_max",
                 "codebleu","codebleu_min","codebleu_max"])

    sorted_keys = sorted(results.keys(), key=lambda x: x.split("_"))

    for task_key in sorted_keys:
        metrics = results[task_key]
        model, task, method, shot = task_key.split("_")
        row = [model, task, method, shot]

        for metric_name in ["bleu", "em", "codebleu"]:
            values = metrics.get(metric_name, [])
            if values:
                row.extend([metric_name, f"{min(values):.4f}", f"{max(values):.4f}"])
            else:
                row.extend([metric_name, "", ""])

        writer.writerow(row)

print("结果已保存到 output.txt 和 output.csv")