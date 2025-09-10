import os
import re
import csv
from collections import defaultdict

# 根目录
root_dir = "./"

# 存放结果 {task_key: {"bleu": [], "em": [], "codebleu": []}}
results = defaultdict(lambda: {"bleu": [], "em": [], "codebleu": []})

# 简单正则匹配
bleu_pattern = re.compile(r"BLEU_Smooth:\s*{\s*'bleu': ([0-9.]+)")
em_pattern = re.compile(r"EM:\s*{\s*'em_ratio': ([0-9.]+)")
codebleu_pattern = re.compile(r"CodeBleu:\s*{\s*'codebleu': ([0-9.]+)")

# 遍历每个子文件夹
for dirpath, dirnames, filenames in os.walk(root_dir):
    if "output.txt" not in filenames:
        continue

    output_path = os.path.join(dirpath, "output.txt")
    parent_folder = os.path.basename(dirpath)

    # 解析文件夹名，例如 qwen2.5-7b_Cs2Java_cot_1-shot_0
    parts = parent_folder.rsplit("_", 4)
    if len(parts) < 4:
        continue

    model, task, method, shot = parts[0], parts[1], parts[2], parts[3]

    if model in ("qwen3-30b", "qwen2.5-32b"):
        continue

    task_key = f"{model}_{task}_{method}_{shot}"

    with open(output_path, "r", encoding="utf-8") as f:
        text = f.read()

        # BLEU → 只取最后一个
        bleu_vals = bleu_pattern.findall(text)
        if bleu_vals:
            results[task_key]["bleu"].append(float(bleu_vals[-1]))

        # EM → 只取最后一个
        em_vals = em_pattern.findall(text)
        if em_vals:
            results[task_key]["em"].append(float(em_vals[-1]))

        # CodeBleu → 只取最后一个
        codebleu_vals = codebleu_pattern.findall(text)
        if codebleu_vals:
            results[task_key]["codebleu"].append(float(codebleu_vals[-1]))

# 保存结果到 output.txt
with open("output.txt", "w", encoding="utf-8") as fout:
    for task_key, metrics in results.items():
        fout.write(f"Task: {task_key}\n")
        for metric_name, values in metrics.items():
            if values:
                fout.write(f"  {metric_name}: min={min(values):.4f}, max={max(values):.4f}\n")
        fout.write("\n")

# 保存结果到 output.csv
with open("output.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "task", "method", "shot", "metric", "min", "max"])

    for task_key, metrics in results.items():
        model, task, method, shot = task_key.split("_")
        for metric_name, values in metrics.items():
            if values:
                writer.writerow([model, task, method, shot, metric_name,
                                 f"{min(values):.4f}", f"{max(values):.4f}"])

print("结果已保存到 output.txt 和 output.csv")