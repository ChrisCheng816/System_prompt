import re
import gc
import os
import torch
import scann
import subprocess
import numpy as np
import torch.nn.functional as F
from codebleu import calc_codebleu
import evaluate
from exact_match import em_compute, exact_match_no_punct

bleu_metric = evaluate.load("bleu")

def clean_blocks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    block_starts = [i for i, line in enumerate(lines) if line.startswith("BLEU_Smooth:")]

    if not block_starts:
        return

    first_start = block_starts[0]
    last_start = block_starts[-1]
    new_lines = []
    new_lines.extend(lines[:first_start])
    new_lines.extend(lines[first_start:first_start+3])  # 每块固定3行
    new_lines.append("\n")

    if last_start != first_start:  # 避免第一个和最后一个是同一个块
        new_lines.extend(lines[last_start:last_start+3])
        new_lines.append("\n")

    new_lines.extend(lines[last_start+3:])

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def find_prediction_output_pairs(root_dir):
    pairs = []  # 保存 (predictions_path, output_path)

    for dirpath, _, filenames in os.walk(root_dir):
        if "predictions.txt" in filenames and "output.txt" in filenames and "references.txt" in filenames:
            predictions_path = os.path.join(dirpath, "predictions.txt")
            output_path = os.path.join(dirpath, "output.txt")
            references_path = os.path.join(dirpath, "references.txt")
            pairs.append((predictions_path, output_path, references_path))

    return pairs

def append_to_outputs(output_path, BLEU_Smooth, EM, CodeBleu):
    with open(output_path, "a", encoding="utf-8") as f:  # "a" 追加模式
        f.write(f"\nBLEU_Smooth: {BLEU_Smooth}\n")
        f.write(f"EM: {EM}\n")
        f.write(f"CodeBleu: {CodeBleu}\n")
    clean_blocks(output_path)

def process_predictions(pairs):
    cleaned_pairs = [] 
    for pred_path, out_path, ref_path in pairs:
        dirpath = os.path.dirname(pred_path)
        cleaned_path = os.path.join(dirpath, "predictions_cleaned.txt")

        with open(pred_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        with open(cleaned_path, "w", encoding="utf-8") as f_out:  # 覆盖写入
            for line in lines:
                line = line.rstrip("\n")
                line = clean(line)
                f_out.write(line + "\n")
        cleaned_pairs.append((cleaned_path, ref_path, out_path))
    return cleaned_pairs

def read_predictions_and_references(pred_path, ref_path):
    """
    Read predictions.txt and references.txt, treating each line as an array element.
    """
    with open(pred_path, "r", encoding="utf-8") as f_pred:
        predictions = [line.rstrip("\n") for line in f_pred.readlines()]

    with open(ref_path, "r", encoding="utf-8") as f_ref:
        references = [line.rstrip("\n") for line in f_ref.readlines()]

    return predictions, references

def get_lang_from_output(output_path):
    """
    Parse the first line of output.txt and return lang (lowercase)
    """
    with open(output_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if "Direction" in first_line:
        parts = first_line.split("Direction:", 1)[1].strip().split()[0]
        lang = "c_sharp" if "Java2Cs" == parts else "java"

        return lang
    else:
        raise ValueError(f"{output_path} The first line did not find the Direction information.")

def compute(pairs):
    for pred_path, ref_path, out_path in pairs:
        lang = get_lang_from_output(out_path)
        predictions, references = read_predictions_and_references(pred_path, ref_path)
        try:
            # cmd = f"python3 evaluator_cc/evaluator.py -ref {filepath}/references.txt -pre {filepath}/predictions.txt"
            # result1 = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            cmd = f"python3 calc_code_bleu.py --refs {ref_path} --hyp {pred_path} --lang {lang} --params 0.25,0.25,0.25,0.25"
            codebleu = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            BLEU_Smooth = bleu_metric.compute(predictions=predictions, references=references, smooth=True)

            EM = em_compute(predictions, references)

            codebleu_score = calc_codebleu(references, predictions, lang)
            ngram_match, weighted_ngram_match = extract_value(codebleu.stdout)

            CodeBleu_result = ngram_match*25 + weighted_ngram_match*25 + codebleu_score["syntax_match_score"]*25 + codebleu_score["dataflow_match_score"]*25
            CodeBleu = {'codebleu': CodeBleu_result, 'ngram_match_score': ngram_match, 'weighted_ngram_match_score': weighted_ngram_match, 'syntax_match_score': codebleu_score["syntax_match_score"], 'dataflow_match_score': codebleu_score["dataflow_match_score"]}
        except Exception as e:
            print(f"Error occurred while running evaluation: {e}")
        append_to_outputs(out_path, BLEU_Smooth, EM, CodeBleu)

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
    # 先判断是否以 ;} 或 ;}} 或 ;}}} 结尾
    if re.search(r"(;}|;}}|;}}})$", text):
        return text  # 以分号+右花括号结尾，直接保留

    # 否则，匹配 ;} 或 ;}} 或 ;}}} 且后面紧跟大写字母
    matches = list(re.finditer(r"(;}|;}}|;}}})(?=[A-Z])", text))
    if not matches:
        return text  # 没有匹配，直接返回

    # 最后一个匹配的位置
    last_end = matches[-1].end()

    # 检查匹配后是否有内容
    if last_end >= len(text):
        return text  # 后面没有内容，保留原文本

    # 有内容就截断并打印删除部分
    deleted_content = text[last_end:]

    # 返回截断后的文本
    return text[:last_end]


def remove_backticks(text):
    """
    删除文本中所有 ``` 符号
    """
    return text.replace("```", "")


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

def clean(text):
    text = remove_backticks(text)
    text = trim_code(text)
    text = clean_after_last_brace(text)
    return text

if __name__ == "__main__":
    root_dir = "translation_results"  # 修改这里
    pairs = find_prediction_output_pairs(root_dir)
    cleaned_pairs = process_predictions(pairs)
    compute(cleaned_pairs)
    print(f"共找到 {len(cleaned_pairs)} 对文件")