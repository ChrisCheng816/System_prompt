import pandas as pd
import numpy as np

# 读取 CSV 文件
df = pd.read_csv("output_codebleu.csv")

# 要计算标准差的列名
prompt_cols = ["codebleu_40", "codebleu_185", "codebleu_294", "codebleu_480", "codebleu_638"]

# 计算标准差（使用样本标准差 ddof=1）
df["std_codebleu"] = round(df[prompt_cols].std(axis=1, ddof=1), 2)

# 输出结果
print(df[["model", "task", "method", "shot", "std_codebleu"]])

# 如需保存到新文件
df.to_csv("with_std.csv", index=False)