import re
import gc
import os
import torch
import scann
import json
import subprocess
import numpy as np
import torch.nn.functional as F
from codebleu import calc_codebleu
import evaluate
from exact_match import em_compute, exact_match_no_punct


def find_prediction_output_pairs(root_dir):
    pairs = []  # 保存 (predictions_path, output_path)

    for dirpath, _, filenames in os.walk(root_dir):
        if "python" in dirpath:
            filename = "python.jsonl"
        elif "java" in dirpath:
            filename = "java.jsonl"
        else:
            continue
        if filename in filenames and "output.json" in filenames:
            predictions_path = os.path.join(dirpath, filename)
            output_path = os.path.join(dirpath, "output.json")
            pairs.append((predictions_path, output_path))

    return pairs

def process_predictions(pairs):
    cleaned_pairs = [] 
    for pred_path, out_path in pairs:
        dirpath = os.path.dirname(pred_path)
        
        data_list = []
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data_list.append(json.loads(line))

        with open(pred_path, "w", encoding="utf-8") as f_out:
            for data in data_list:
                if "raw_generation" in data:
                    cleaned_text = clean(data["raw_generation"][0], dirpath)
                    data["raw_generation"][0] = cleaned_text
                # 保留其他字段不变
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")  # 覆盖写入


        cleaned_pairs.append((pred_path, out_path))
    return cleaned_pairs

def compute(pairs):
    for idx, (pred_path, out_path) in enumerate(pairs, start=1):
        filepath = os.path.dirname(pred_path)
        print(filepath)
        print(f"第 {idx} 个文件对") if idx % 10 == 0 else None
        try:
            cmd = f"python3 -u eval_all.py --result_path {filepath} --save_path {filepath}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="../eval")
            print(result.stdout)
        except Exception as e:
            print(f"Error occurred while running evaluation: {e}")

        for filename in os.listdir("../generation_results_mceval/tmp"):
            file_path = os.path.join("../generation_results_mceval/tmp", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)

        # outpath = os.path.dirname(out_path)
        # cleaned_path = os.path.join(outpath, "output_cleaned.json")       
        # append_to_outputs(cleaned_path, result.stdout, bleu_result)

def extract_value(text):
    pattern = r"ngram match:\s*([0-9.]+),\s*weighted ngram match:\s*([0-9.]+)"

    match = re.search(pattern, text)
    if match:
        ngram_match = float(match.group(1))
        weighted_ngram_match = float(match.group(2))
        return ngram_match, weighted_ngram_match
    else:
        print("No match found")

def clean_after_last_brace(text):
    # 允许 ; 和 } 之间有空格
    pattern_end = r"(;\s*}\s*|\;\s*}\s*}\s*|\;\s*}\s*}\s*}\s*)$"
    if re.search(pattern_end, text):
        return text  # 以分号+花括号结尾，直接保留

    # 匹配 ;} 或 ;}} 或 ;}}} 后面跟大写字母（允许空格）
    pattern_mid = r"(;\s*}\s*|\;\s*}\s*}\s*|\;\s*}\s*}\s*}\s*)(?=[A-Z])"
    matches = list(re.finditer(pattern_mid, text))
    if not matches:
        return text  # 没有匹配，直接返回

    # 最后一个匹配的位置
    last_end = matches[-1].end()

    if last_end >= len(text):
        return text  # 后面没有内容

    deleted_content = text[last_end:]
    return text[:last_end]

def remove_lang(text):
    if text.startswith("```java\n"):
        text = text[len("```java\n"):]
    elif text.startswith("```python\n"):
        text = text[len("```python\n"):]
    # 去掉末尾 ```
    if text.endswith("```"):
        text = text[:-3]

    return text
def remove_backticks(text):
    """
    删除文本中所有 ``` 符号
    """
    return text.replace("```", "")

def extract_output(text: str) -> str:
    # 1. 查找 assistantfinal
    if "assistantfinal" in text:
        return text.split("assistantfinal")[-1].strip()
    return text.strip()

def trim_code(text: str) -> str:
    # 注意顺序很重要，长的写前面
    patterns = [
        r"@Override public ",   # @Override 后面紧跟一个空格，然后 public 再跟一个空格
        r"protected internal ", # protected + 一个空格 + internal + 一个空格
        r"public ",             # public + 一个空格
    ]
    
    # 拼成一个大模式，用 | 连接
    combined_pattern = "|".join(patterns)
    
    # 搜索第一个匹配
    match = re.search(combined_pattern, text)
    if match:
        return text[match.start():]
    return text

def clean(text, dirpath):
    # text = remove_backticks(text)
    # text = trim_code(text)
    # text = clean_after_last_brace(text)
    if "openai" in dirpath.lower():
        text = extract_output(text)
    # text = remove_lang(text)
    return text

if __name__ == "__main__":
    root_dir = "../generation_results_mceval"  # 修改这里
    pairs = find_prediction_output_pairs(root_dir)
    cleaned_pairs = process_predictions(pairs)
    print(f"共找到 {len(cleaned_pairs)} 对文件")
    compute(cleaned_pairs)
    