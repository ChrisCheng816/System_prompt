import os
import re
import csv
import json
from collections import defaultdict

root_dir = "./"

# 存放最终结果
# { (model, task, method, shot) : [ {prompt, prompt_len, bleu, em, codebleu} ] }
results = defaultdict(list)
Mapping = {"0":40, "1":185, "2":294, "3":480, "4": 638}
# 收集全局所有出现过的长度
all_prompt_lengths = set()
count = 0
for dirpath, dirnames, filenames in os.walk(root_dir):
    count = count +1 
    output_path = ""
    for f in filenames:
        if f == "output.json":
            target_file = f
            output_path = os.path.join(dirpath, target_file)
            break
    if output_path == "":
        continue
    parent_folder = os.path.basename(dirpath)
    # 解析文件夹名
    parts = parent_folder.rsplit("_", 4)
    if len(parts) < 4:
        continue

    model, task, method, shot, index = parts[0], parts[1], parts[2], parts[3], parts[4]

    task_key = f"{model}_{task}_{method}_{shot}"
    target_value = None
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        try:
            target_value = data["CodeBleu"]["codebleu"]
        except ValueError:
            print(f"Warning: cannot get the date form line -> {data}")

    cur_result = {"prompt": index, "prompt_len": Mapping[index]}

    if target_value is not None:
        cur_result["codebleu"] = target_value

    # task_key 需要你根据目录或其他方式确定，这里假设已经有 task_key
    results[task_key].append(cur_result)

    all_prompt_lengths.add(Mapping[index])

# 把所有长度统一排序
all_prompt_lengths = sorted(all_prompt_lengths)
print(all_prompt_lengths)
# 输出 CSV
csv_path = "output_codebleu.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # 动态生成表头
    header = ["model", "task", "method", "shot"]
    for metric in ["codebleu"]:
        header.append(metric)
        for plen in all_prompt_lengths:
            header.append(f"{metric}_{plen}")
    writer.writerow(header)

    for task_key, result_list in sorted(results.items()):
        print("Processing:", task_key)
        model, task, method, shot = task_key.split("_")
        row = [model, task, method, shot]

        # 先按 prompt_len 和 BLEU 大小排序
        sorted_list = sorted(
            result_list,
            key=lambda x: (x["prompt_len"], x.get("codebleu", 0)),
            reverse=True
        )

        # 去重：同样长度只保留 BLEU 最大的
        unique_by_len = {}
        for res in sorted_list:
            plen = res["prompt_len"]
            if plen not in unique_by_len:
                unique_by_len[plen] = res

        # 按照全局 header 输出
        for metric in ["codebleu"]:
            row.append(metric)
            for plen in all_prompt_lengths:
                res = unique_by_len.get(plen, None)
                if res is None or metric not in res:
                    row.append("")
                else:
                    row.append(f"{res[metric]:.2f}")


        writer.writerow(row)