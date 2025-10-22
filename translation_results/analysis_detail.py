import sys
sys.path.append("../Prompts")
import os
import re
import csv
import json
from collections import defaultdict
from trans_prompts import trans_prompts_code

root_dir = "./"

# 存放最终结果
# { (model, task, method, shot) : [ {prompt, prompt_len, bleu, em, codebleu} ] }
results = defaultdict(list)

# 收集全局所有出现过的长度
all_prompt_lengths = set()

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "output_cleaned.json" in filenames:
        output_path = os.path.join(dirpath, "output_cleaned.json")
    else:
        output_path = os.path.join(dirpath, "output.json")
    parent_folder = os.path.basename(dirpath)

    # 解析文件夹名
    parts = parent_folder.rsplit("_", 4)
    if len(parts) < 4:
        continue

    model, task, method, shot = parts[0], parts[1], parts[2], parts[3]

    task_key = f"{model}_{task}_{method}_{shot}"

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(dirpath, "output.json"), "r", encoding="utf-8") as o:
        data_origin = json.load(o)
    # 每条记录提取 system_prompt
    system_prompt = data_origin["system_prompt"]
    prompt_len = len(system_prompt)

    cur_result = {"prompt": system_prompt, "prompt_len": prompt_len}

    # 处理 BLEU
    if "BLEU_Smooth" in data and "bleu" in data["BLEU_Smooth"]:
        bleu_val = round(float(data["BLEU_Smooth"]["bleu"]) * 100, 2)
        cur_result["bleu"] = bleu_val

    # 处理 EM
    if "EM" in data and "em_ratio" in data["EM"]:
        em_val = round(float(data["EM"]["em_ratio"]) * 100, 2)
        cur_result["em"] = em_val

    # 处理 CodeBleu
    if "CodeBleu" in data and "codebleu" in data["CodeBleu"]:
        cur_result["codebleu"] = float(data["CodeBleu"]["codebleu"])

    # task_key 需要你根据目录或其他方式确定，这里假设已经有 task_key
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
                    row.append(f"{res[metric]:.2f}")

        writer.writerow(row)

json_path = "output.jsonl"
all_records = []
seen_prompts = set()

# 去重并收集所有记录
for result_list in results.values():
    for res in result_list:
        sp = res["prompt"]
        if sp in seen_prompts:
            continue
        seen_prompts.add(sp)
        all_records.append({
            "system_prompt": sp,
            "prompt_len": res["prompt_len"]
        })

# 按 prompt_len 从小到大排序
all_records.sort(key=lambda x: x["prompt_len"])

# 输出到文件，每条记录独立一行 JSON
with open(json_path, "w", encoding="utf-8") as jf:
    for record in all_records:
        json.dump(record, jf, ensure_ascii=False)
        jf.write("\n")

print("Summarization results have been saved to output.json")