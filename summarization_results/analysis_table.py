import pandas as pd

# 读取数据
df = pd.read_csv("output_detail.csv")

# 定义三个维度
metrics = ["Bleu", "F1", "Meteor", "CodeT5+"]

# 提取所有可能的 prompt 后缀
def createCSV(para1, para2, outpath, number):
    prompts = []  # base = 没有后缀的列
    for col in df.columns:
        if "_" in col and col.split("_")[0] in metrics:
            suffix = col.split("_")[1]
            if suffix not in prompts:
                prompts.append(suffix)

    results = []

    # 遍历 method 和 task
    for (method, task), group in df.groupby([para1, para2]):
        for prompt in prompts:
            combined_sum = 0.0
            missing = []

            # 遍历 group 内所有模型
            for _, row in group.iterrows():
                for metric in metrics:
                    colname = f"{metric}_{prompt}"
                    if colname not in row.index:
                        missing.append(f"{row['model']}_{colname}")
                        continue
                    if metric in ["F1", "Meteor", "CodeT5+"]:
                        factor = 100
                    else:
                        factor = 1 
                    value = row[colname]
                    try:
                        combined_sum += float(value*factor)
                    except (ValueError, TypeError):
                        missing.append(f"{row['model']}_{colname}:{value}")

            results.append({
                "method": method,
                "task": task,
                "prompt": prompt,
                "combined_sum": round(combined_sum/number,2),
                "missing_info": ";".join(missing) if missing else ""
            })

    # 保存结果
    out_df = pd.DataFrame(results)
    out_df.to_csv(outpath, index=False)
    print(f"已保存结果到{outpath}")

def rq3_latex():

    # 读入 CSV
    df = pd.read_csv("rq3.csv")

    tasks = ["java", "python"]
    shot_map = {
        "zero": "0-shot",
        "naive": "Naive",
        "retrieval": "Retrieval"
    }

    output_file = "latex_rq3_summ.tex"

    with open(output_file, "w") as f:
        for task in tasks:
            df_task = df[df["task"] == task]

            # 构造表格数据
            table_data = {}
            for model in df_task["model"].unique():
                table_data[model] = {}
                for method_key, shot_name in shot_map.items():
                    row = df_task[(df_task["model"] == model) & (df_task["method"] == method_key)]
                    if not row.empty:
                        cell = f"{row['score_min'].values[0]:.2f}/{row['score_max'].values[0]:.2f}"
                    else:
                        cell = "-"
                    table_data[model][shot_name] = cell

            # 写 LaTeX 表格
            f.write(f"% Table for {task}\n")
            f.write("\\begin{tabular}{|l|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Model & 0-shot & Naive & Retrieval \\\\\n")
            f.write("\\hline\n")
            for model, shot_vals in table_data.items():
                row = [model] + [shot_vals.get(h, "-") for h in ["0-shot", "Naive", "Retrieval"]]
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n\n")

    print(f"已生成 LaTeX 表格文件: {output_file}")

def detail_csv():
    df = pd.read_csv("output_detail.csv")

    prompts = []  # 找出所有后缀
    for col in df.columns:
        if "_" in col and col.split("_")[0] in metrics:
            suffix = col.split("_")[1]
            if suffix not in prompts:
                prompts.append(suffix)

    # 计算综合score（平均）
    for prompt in prompts:
        df[f"score_{prompt}"] = 0
        for metric in metrics:
            if metric in ["F1", "Meteor", "CodeT5+"]:
                # factor = 100
                continue
            else:
                factor = 1
            colname = f"{metric}_{prompt}"
            df[f"score_{prompt}"] += df[colname] * factor
        df[f"score_{prompt}"] = round(df[f"score_{prompt}"], 2)

    # 找到 shot 所在列
    shot_idx = df.columns.get_loc("shot")

    # 只保留 shot 前的列
    keep_cols = list(df.columns[:shot_idx+1])
    df = df[keep_cols + [f"score_{s}" for s in prompts]]

    # 只保留每行的最小值和最大值
    df["score_min"] = df[[f"score_{s}" for s in prompts]].min(axis=1)
    df["score_max"] = df[[f"score_{s}" for s in prompts]].max(axis=1)

    # 删除原始 score 列，只保留 min/max
    df = df[keep_cols + ["score_min", "score_max"]]

    # 保存
    df.to_csv("rq3.csv", index=False)
    print("已保存结果到 rq3.csv")

createCSV("method", "task", "rq1.csv", 24)
createCSV("model", "task", "rq2.csv", 12)
detail_csv()
rq3_latex()