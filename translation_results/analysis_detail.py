import sys
sys.path.append("../Prompts")
import os
import re
import csv
from collections import defaultdict
from trans_prompts import trans_prompts_code

root_dir = "./"

# 存放最终结果
# { (model, task, method, shot) : [ {prompt, prompt_len, bleu, em, codebleu} ] }
results = defaultdict(list)

# 收集全局所有出现过的长度
all_prompt_lengths = set()

# 正则匹配指标
bleu_pattern = re.compile(r"BLEU_Smooth:\s*{\s*'bleu': ([0-9.]+)")
em_pattern = re.compile(r"EM:\s*{\s*'em_ratio': ([0-9.]+)")
codebleu_pattern = re.compile(r"CodeBleu:\s*{\s*'codebleu': ([0-9.]+)")

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "output.txt" not in filenames:
        continue

    output_path = os.path.join(dirpath, "output.txt")
    parent_folder = os.path.basename(dirpath)

    # 解析文件夹名
    parts = parent_folder.rsplit("_", 4)
    if len(parts) < 4:
        continue

    model, task, method, shot = parts[0], parts[1], parts[2], parts[3]
    if model in ("qwen2.5-32b",):
        continue

    task_key = f"{model}_{task}_{method}_{shot}"

    with open(output_path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    # 删除前两行空行
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while len(lines) > 1 and lines[1].strip() == "":
        lines.pop(1)

    # 提取 system prompt: 从第二行开始到第一个空行
    system_prompt_lines = []
    for line in lines[1:]:
        if line.strip() == "":
            break
        system_prompt_lines.append(line.strip())
    system_prompt = "\n".join(system_prompt_lines).strip()
    prompt_len = len(system_prompt)

    # 用正则匹配指标，只取最后一个数值
    text = "".join(lines)
    bleu_vals = bleu_pattern.findall(text)
    em_vals = em_pattern.findall(text)
    codebleu_vals = codebleu_pattern.findall(text)

    cur_result = {"prompt": system_prompt, "prompt_len": prompt_len}
    if bleu_vals:
        cur_result["bleu"] = float(bleu_vals[-1])
    if em_vals:
        cur_result["em"] = float(em_vals[-1])
    if codebleu_vals:
        cur_result["codebleu"] = float(codebleu_vals[-1])

    results[task_key].append(cur_result)
    all_prompt_lengths.add(prompt_len)

# 把所有长度统一排序
all_prompt_lengths = sorted(all_prompt_lengths)

# 输出 CSV
csv_path = "output_detail.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # 动态生成表头
    header = ["model", "task", "method", "shot"]
    for metric in ["bleu", "em", "codebleu"]:
        header.append(metric)
        for plen in all_prompt_lengths:
            header.append(f"{metric}_{plen}")
    writer.writerow(header)

    for task_key, result_list in sorted(results.items()):
        model, task, method, shot = task_key.split("_")
        row = [model, task, method, shot]

        # 先按 prompt_len 和指标大小排序（保证同长度时指标大的在前）
        sorted_list = sorted(
            result_list,
            key=lambda x: (x["prompt_len"], x.get("bleu", 0), x.get("em", 0), x.get("codebleu", 0)),
            reverse=True
        )

        # 去重：同样长度只保留第一个（指标大的）
        unique_by_len = {}
        for res in sorted_list:
            plen = res["prompt_len"]
            if plen not in unique_by_len:
                unique_by_len[plen] = res

        # 按照全局 header 输出
        for metric in ["bleu", "em", "codebleu"]:
            row.append(metric)
            for plen in all_prompt_lengths:
                res = unique_by_len.get(plen, None)
                if res is None or metric not in res:
                    row.append("")
                else:
                    row.append(f"{res[metric]:.4f}")

        writer.writerow(row)