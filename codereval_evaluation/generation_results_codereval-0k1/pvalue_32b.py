import pandas as pd
import os
from statsmodels.stats.contingency_tables import mcnemar


all_results = []
root_dir = "./csv_results_python"
for dirpath, dirnames, filenames in os.walk(root_dir):
    output_path = ""
    for f in filenames:
        if f.endswith(".csv"):
            target_file = f
            output_path = os.path.join(dirpath, target_file)
        parent_folder = os.path.splitext(target_file)[0]
        
        parts = parent_folder.rsplit("_", 3)
        if len(parts) < 3:
            continue

        model, task, method, shot = parts[0], parts[1], parts[2], parts[3]
        df = pd.read_csv(output_path)

        base_col = "STRUCT"

        other_cols = ["SELFC"]

        results = []

        for idx, col in enumerate(other_cols, start=1):
            a = ((df[base_col] == 1) & (df[col] == 1)).sum()
            b = ((df[base_col] == 1) & (df[col] == 0)).sum()
            c = ((df[base_col] == 0) & (df[col] == 1)).sum()
            d = ((df[base_col] == 0) & (df[col] == 0)).sum()
            
            table = [[a, b], [c, d]]
            
            
            result = mcnemar(table, exact=False, correction=True)
            
            
            OR = (b + 1) / (c + 1)
            
            all_results.append({
                "ID": idx,
                "Model": model,
                "Task": task,
                "Method": method,
                "Shot": shot,
                "Compare": f"{base_col} vs {col}",
                # "a": a, "b": b, "c": c, "d": d,
                "OR": round(OR, 3),
                "p_value": round(result.pvalue, 5)
            })

output_csv = "mcnemar_results_32.csv"
pd.DataFrame(all_results).to_csv(output_csv, index=False)