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

bleu_metric = evaluate.load("bleu")


def find_prediction_output_pairs(root_dir):
    pairs = []  # 保存 (predictions_path, output_path)

    for dirpath, _, filenames in os.walk(root_dir):
        if "predictions.jsonl" in filenames and "output.json" in filenames and "references.txt" in filenames:
            predictions_path = os.path.join(dirpath, "predictions.jsonl")
            output_path = os.path.join(dirpath, "output.json")
            references_path = os.path.join(dirpath, "references.txt")
            pairs.append((predictions_path, output_path, references_path))

    return pairs

def append_to_outputs(output_path, Bleu, BLEU_Smooth):
    new_record = {
        "BLEU_Normal": Bleu,
        "Bleu_HF": BLEU_Smooth
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_record, f, indent=4, ensure_ascii=False)

def process_predictions(pairs):
    cleaned_pairs = [] 
    for pred_path, out_path, ref_path in pairs:
        dirpath = os.path.dirname(pred_path)
        cleaned_path_jsonl = os.path.join(dirpath, "predictions_cleaned.jsonl")
        cleaned_path_txt = os.path.join(dirpath, "predictions_cleaned.txt")
        
        data_list = []
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data_list.append(json.loads(line))

        with open(cleaned_path_jsonl, "w", encoding="utf-8") as f_out:  # 覆盖写入
            for i, data in enumerate(data_list):
                line = clean(data["prediction"], dirpath)
                record = {"id": data["id"], "prediction": line}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(cleaned_path_txt, "w", encoding="utf-8") as f_out:  # 覆盖写入
            for i, data in enumerate(data_list):
                line = clean(data["prediction"], dirpath)
                f_out.write(f"{i}\t{line}\n")

        cleaned_pairs.append((cleaned_path_txt, cleaned_path_jsonl, ref_path, out_path))
    return cleaned_pairs

def read_predictions_and_references(pred_path, ref_path):
    """
    Read predictions.txt and references.txt, treating each line as an array element.
    """
    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f_pred:
        data_list = [json.loads(line) for line in f_pred if line.strip()]
        predictions = [item["prediction"] for item in data_list]

    with open(ref_path, "r", encoding="utf-8") as f_ref:
        references = []
        for line in f_ref:
            # 按第一个 tab 或空格切分
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                idx, text = parts
                references.append(text.strip())
            else:
                raise ValueError(f"参考文件格式错误: {line.strip()}")

    return predictions, references

def compute(pairs):
    for idx, (cleaned_path_txt, cleaned_path_jsonl, ref_path, out_path) in enumerate(pairs, start=1):
        print(f"第 {idx} 个文件对") if idx % 10 == 0 else None
        predictions, references = read_predictions_and_references(cleaned_path_jsonl, ref_path)
        try:
            cmd = f"python3 ../evaluator_ct/evaluator.py {ref_path} < {cleaned_path_txt}"
            print(cmd)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(result.stderr)
        except Exception as e:
            print(f"Error occurred while running evaluation: {e}")

        bleu_result = bleu_metric.compute(predictions=predictions, references=references, smooth=True)
        
        outpath = os.path.dirname(out_path)
        cleaned_path = os.path.join(outpath, "output_cleaned.json")       
        append_to_outputs(cleaned_path, result.stdout, bleu_result)

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


def remove_backticks(text):
    """
    删除文本中所有 ``` 符号
    """
    return text.replace("```", "")

def extract_output(text: str) -> str:
    # 1. 查找 assistantfinal
    if "assistantfinal" in text:
        after = text.split("assistantfinal")[-1].strip()
        if after:
            return after

    # # 2. 查找 final output:
    # if "final output:" in text:
    #     return text.split("final output:")[-1].strip()

    # # 3. 查找 output:
    # if "output:" in text:
    #     return text.split("output:")[-1].strip()

    # 4. 都没有，返回原文本（或空字符串）
    return text.strip()

def trim_code(text: str) -> str:
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
    text = remove_backticks(text)
    # text = trim_code(text)
    # text = clean_after_last_brace(text)
    if "openai" in dirpath.lower():
        text = extract_output(text)
    return text

if __name__ == "__main__":
    root_dir = "../summarization_results"  # 修改这里
    pairs = find_prediction_output_pairs(root_dir)
    # cleaned_pairs = process_predictions(pairs)
    print(f"共找到 {len(cleaned_pairs)} 对文件")
    compute(pairs)
    