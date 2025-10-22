import pandas as pd

codereval_rq1 = "../generation_results_codereval/Results/rq1.csv"
codereval_rq2 = "../generation_results_codereval/Results/rq2.csv"
mceval_rq1 = "../generation_results_mceval/rq1.csv"
mceval_rq2 = "../generation_results_mceval/rq2.csv"

def compute_final(row):
    if row["task"].lower() == "python":
        return round(row["combined_sum_1"] * 0.8214 + row["combined_sum_2"] * 0.1786, 2)
    elif row["task"].lower() == "java":
        # 避免除以0
        return round(row["combined_sum_1"] * 0.8127 + row["combined_sum_2"] * 0.1873, 2)
    else:
        return None

df1 = pd.read_csv(codereval_rq1)
df2 = pd.read_csv(mceval_rq1)
merged = pd.merge(df1, df2, on=["method", "task", "prompt"], suffixes=("_1", "_2"))
merged["final_score"] = merged.apply(compute_final, axis=1)
merged = merged.drop(columns=["combined_sum_1", "combined_sum_2", "missing_info_1", "missing_info_2"])
merged.to_csv("gen_rq1.csv", index=False)


df1 = pd.read_csv(codereval_rq2)
df2 = pd.read_csv(mceval_rq2)
merged = pd.merge(df1, df2, on=["method", "task", "prompt"], suffixes=("_1", "_2"))
merged["final_score"] = merged.apply(compute_final, axis=1)
merged = merged.drop(columns=["combined_sum_1", "combined_sum_2", "missing_info_1", "missing_info_2"])
merged.to_csv("gen_rq2.csv", index=False)

