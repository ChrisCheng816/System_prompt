import os
import re
import csv
import json
from collections import defaultdict

# Root directory
root_dir = "./"

# 存放结果 {task_key: {"bleu": [], "em": [], "codebleu": []}}
results = defaultdict(lambda: {"pass@1": []})

# Iterate through each subfolder
for dirpath, dirnames, filenames in os.walk(root_dir):
    if "java.jsonl" in filenames or "python.jsonl" in filenames:
        for f in filenames:
            if f not in ["python.jsonl", "java.jsonl", "output.json", "predictions.jsonl"]:
                target_file = f
                break
        output_path = os.path.join(dirpath, target_file)
    parent_folder = os.path.basename(dirpath)

    # Parsing folder names, such as qwen2.5-7b_Cs2Java_cot_1-shot_0
    parts = parent_folder.rsplit("_", 4)
    if len(parts) < 4:
        continue

    model, task, method, shot = parts[0], parts[1], parts[2], parts[3]

    task_key = f"{model}_{task}_{method}_{shot}"

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.readlines()[-1].strip()
        for prefix in ("python", "java"):
            if content.startswith(prefix):
                content = content[len(prefix):]
                break
        print(output_path)
        data = json.loads(content)

        if "accuracy" in data:
            bleu_val = round(data["accuracy"]*100, 2)
            results[task_key]["pass@1"].append(bleu_val)

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
