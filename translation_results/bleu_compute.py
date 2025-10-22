import os
import json
import evaluate
from sacrebleu import corpus_bleu

hfsacrebleu = evaluate.load("sacrebleu")
# 根目录，替换成你的路径
root_dir = "./"
output_file = "sacrebleu_results.jsonl"

def read_file_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    return data_list

def read_file_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

results = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    # 判断文件是否存在
    if "predictions_cleaned.jsonl" in filenames:
        pred_file = os.path.join(dirpath, "predictions_cleaned.jsonl")
    elif "predictions.jsonl" in filenames:
        pred_file = os.path.join(dirpath, "predictions.jsonl")
    else:
        continue  # 没有预测文件，跳过

    if "references.txt" not in filenames:
        continue  # 没有 references，跳过

    ref_file = os.path.join(dirpath, "references.txt")

    # 读取内容
    predictions = read_file_json(pred_file)
    references = read_file_txt(ref_file)

    # SacreBLEU 需要 [[ref1, ref2, ...], [ref1, ref2, ...]] 这样的列表格式
    references_list = [references]

    # 计算 BLEU
    print(f"Calculating BLEU for directory: {dirpath}")
    bleu = corpus_bleu(predictions, references_list)
    hfsacrebleu_result = hfsacrebleu.compute(predictions=predictions, references=references)
    # 保存结果
    result = ({
        "directory": dirpath,
        "prediction_file": os.path.basename(pred_file),
        "reference_file": os.path.basename(ref_file),
        "sacrebleu_score": bleu.score,
        "HF_sacrebleu": hfsacrebleu_result["score"]
    })

    output_file = os.path.join(dirpath, "sacrebleu_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
