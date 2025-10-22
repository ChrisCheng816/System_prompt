import os
import re
import csv
import json
from collections import defaultdict

# Root directory
root_dir = "./"

# 存放结果 {task_key: {"bleu": [], "em": [], "codebleu": []}}
results = defaultdict(lambda: {"bleu": [], "F1": [], "METEOR": [], "CodeT5": []})

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
    
    with open(os.path.join(dirpath, "output.json"), "r", encoding="utf-8") as o:
        data_origin = json.load(o)

    if "BLEU_Normal" in data:
        bleu_val = round(float(data["BLEU_Normal"].rstrip("\n")), 2)
        results[task_key]["bleu"].append(bleu_val)
    
    if "summary" in data_origin:
        f1 = round(float(data_origin["summary"]["BERTScore_F1_mean"]), 2)
        results[task_key]["F1"].append(f1)

    if "summary" in data_origin:
        meteor = round(float(data_origin["summary"]["METEOR_mean"]), 2)
        results[task_key]["METEOR"].append(meteor)

    if "summary" in data_origin:
        CodeT5 = round(data_origin["summary"]["CodeT5_plus_cosine_mean"], 2)
        results[task_key]["CodeT5"].append(CodeT5)

# Save the results to output.txt
with open("output.csv", "w", newline="", encoding="utf-8") as csvfile:
    # writer = csv.writer(csvfile)
    # writer.writerow(["model", "task", "method", "shot", "metric", "min", "max"])

    # # 对 task_key 排序
    # sorted_keys = sorted(results.keys(), key=lambda x: x.split("_"))

    # for task_key in sorted_keys:
    #     metrics = results[task_key]
    #     model, task, method, shot = task_key.split("_")
    #     for metric_name, values in metrics.items():
    #         if values:
    #             writer.writerow([
    #                 model, task, method, shot, metric_name,
    #                 f"{min(values):.4f}", f"{max(values):.4f}"
    #             ])
    writer = csv.writer(csvfile)
    writer.writerow(["model","task","method","shot",
                 "bleu","bleu_min","bleu_max",
                 "F1","F1_min","F1_max",
                 "METEOR","METEOR_min","METEOR_max",
                 "CodeT5","CodeT5_min","CodeT5_max"])

    sorted_keys = sorted(results.keys(), key=lambda x: x.split("_"))

    for task_key in sorted_keys:
        metrics = results[task_key]
        model, task, method, shot = task_key.split("_")
        row = [model, task, method, shot]

        for metric_name in ["bleu", "F1", "METEOR","CodeT5"]:
            values = metrics.get(metric_name, [])
            if values:
                row.extend([metric_name, f"{min(values):.4f}", f"{max(values):.4f}"])
            else:
                row.extend([metric_name, "", ""])

        writer.writerow(row)

print("Results have been saved to output.txt and output.csv")
