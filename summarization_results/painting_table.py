import pandas as pd

csv_file = "output.csv"
df = pd.read_csv(csv_file)  # 不指定 names

method_order = ["zero", "naive", "cot", "retrieval"]

for (model, task), group in df.groupby(["model", "task"]):
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append(f"\\caption{{{model} on {task} task}}")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\hline")
    latex_lines.append("Method + Shot & BLEU (min/max) \\\\")
    latex_lines.append("\\hline")

    for method in method_order:
        subset = group[group["method"] == method]
        for _, row in subset.iterrows():
            shot = row["shot"]
            bleu_min = row["min"]
            bleu_max = row["max"]

            latex_lines.append(f"{method} {shot} & {bleu_min:.2f} / {bleu_max:.2f} \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}\n")

    output_file = f"./tables/{model}_{task}_table.tex"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"LaTeX table for {model} - {task} saved to {output_file}")