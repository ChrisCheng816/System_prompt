import os
import json
from sacrebleu import corpus_bleu

# 根目录，替换成你的路径
root_dir = "./"
output_file = "sacrebleu_results.jsonl"

def read_file_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

results = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    # 判断文件是否存在
    if "predictions_cleaned.txt" in filenames:
        pred_file = os.path.join(dirpath, "predictions_cleaned.txt")
    elif "predictions.txt" in filenames:
        pred_file = os.path.join(dirpath, "predictions.txt")
    else:
        continue  # 没有预测文件，跳过

    if "references.txt" not in filenames:
        continue  # 没有 references，跳过

    ref_file = os.path.join(dirpath, "references.txt")

    # 读取内容
    predictions = read_file_lines(pred_file)
    references = read_file_lines(ref_file)
    
    # SacreBLEU 需要 [[ref1, ref2, ...], [ref1, ref2, ...]] 这样的列表格式
    references_list = [references]

    # 计算 BLEU
    bleu = corpus_bleu(predictions, references_list)

    # 保存结果
    results.append({
        "directory": dirpath,
        "prediction_file": os.path.basename(pred_file),
        "reference_file": os.path.basename(ref_file),
        "bleu_score": bleu.score
    })

    output_file = os.path.join(dirpath, "sacrebleu_results.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")

    print(f"Saved BLEU result in {output_file}")
