import pandas as pd
import os

df_java = pd.read_csv("gpt-20b_java_zero_0-shot.csv")
df_python = pd.read_csv("gpt-20b_python_naive_3-shot.csv")

base_col = "STRUCT"
col = "REASONING"

a = ((df_java[base_col] == 1) & (df_python[col] == 1)).sum()
b = ((df_java[base_col] == 1) & (df_python[col] == 0)).sum()
c = ((df_java[base_col] == 0) & (df_python[col] == 1)).sum()
d = ((df_java[base_col] == 0) & (df_python[col] == 0)).sum()

print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)