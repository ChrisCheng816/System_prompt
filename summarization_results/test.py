import os
import re
import csv
from collections import defaultdict

# Root directory
root_dir = "./"

# 存放结果 {task_key: {"bleu": []}
results = defaultdict(lambda: {"bleu": []})

# Simple Regular Expression Matching
bleu_pattern = re.compile(r"Bleu:\s*([0-9.]+)")

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

    if model in ("qwen3-30b", "qwen2.5-32b"):
        continue

    task_key = f"{model}_{task}_{method}_{shot}"

    with open(output_path, "r", encoding="utf-8") as f:
        text = f.read()

        # BLEU → Take only the last one
        bleu_vals = bleu_pattern.findall(text)
        if bleu_vals:
            results[task_key]["bleu"].append(float(bleu_vals[-1]))

# Save the results to output.txt
with open("output.txt", "w", encoding="utf-8") as fout:
    for task_key, metrics in results.items():
        fout.write(f"Task: {task_key}\n")
        for metric_name, values in metrics.items():
            if values:
                fout.write(f"  {metric_name}: min={min(values):.4f}, max={max(values):.4f}\n")
        fout.write("\n")

# Save the results to output.txt
with open("output.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "task", "method", "shot", "metric", "min", "max"])

    # 对 task_key 排序
    sorted_keys = sorted(results.keys(), key=lambda x: x.split("_"))

    for task_key in sorted_keys:
        metrics = results[task_key]
        model, task, method, shot = task_key.split("_")
        for metric_name, values in metrics.items():
            if values:
                writer.writerow([
                    model, task, method, shot, metric_name,
                    f"{min(values):.4f}", f"{max(values):.4f}"
                ])

print("Results have been saved to output.txt and output.csv")