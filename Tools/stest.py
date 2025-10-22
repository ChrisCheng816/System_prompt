from scipy.stats import friedmanchisquare
import csv
import scikit_posthocs as sp
import numpy as np


# 创建五个数组，每个数组存放对应 prompt 的所有 pass@1 分数
prompt1_scores = []
prompt2_scores = []
prompt3_scores = []
prompt4_scores = []
prompt5_scores = []

# 读取 CSV
with open("../generation_results_codereval/Results/output_detail.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过表头
    for row in reader:
        scores = [float(x) for x in row[5:]]  # 取五个任务的分数
        prompt1_scores.append(scores[0])
        prompt2_scores.append(scores[1])
        prompt3_scores.append(scores[2])
        prompt4_scores.append(scores[3])
        prompt5_scores.append(scores[4])

# 打印检查
print("Prompt1 scores:", prompt1_scores)
print("Prompt2 scores:", prompt2_scores)
print("Prompt3 scores:", prompt3_scores)
print("Prompt4 scores:", prompt4_scores)
print("Prompt5 scores:", prompt5_scores)

stat, p = friedmanchisquare(prompt1_scores, prompt2_scores, prompt3_scores, prompt4_scores, prompt5_scores)
print(stat, p)

# 每行一个任务，每列一个系统提示
data = np.array([
    prompt1_scores,
    prompt2_scores,
    prompt3_scores,
    prompt4_scores,
    prompt5_scores
]).T

# Nemenyi post-hoc test
p_values = sp.posthoc_nemenyi_friedman(data)
print(p_values)