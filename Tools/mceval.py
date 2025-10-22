from datasets import load_dataset
from datasets import DatasetDict

ds_java_train = load_dataset("google/code_x_glue_tc_text_to_code")["train"]
ds_java_train = ds_java_train.rename_column("nl", "instruction")
ds_java_train = ds_java_train.rename_column("code", "output")

ds_python_train = load_dataset("google/code_x_glue_tc_nl_code_search_adv")["train"]
ds_python_train = ds_python_train.rename_column("docstring", "instruction")
ds_python_train = ds_python_train.rename_column("code", "output")

mceval = load_dataset("Multilingual-Multimodal-NLP/McEval", "generation")
mceval_ins = load_dataset("Multilingual-Multimodal-NLP/McEval-Instruct")

mceval_ins_python = mceval_ins.filter(lambda x: x["language"].lower() == "python")
mceval_ins_java = mceval_ins.filter(lambda x: x["language"].lower() == "java")

mceval_python = mceval.filter(lambda x: x["task_id"].lower().startswith("python"))
mceval_java = mceval.filter(
    lambda x: x["task_id"].lower().startswith("java")
    and not x["task_id"].lower().startswith("javascript")
)

mceval_python_all = DatasetDict({
    "train": mceval_ins_python["train"],
    "test": mceval_python["test"]
})

mceval_python_all = DatasetDict({
    "train": ds_python_train,
    "test": mceval_python["test"]
})

# Java
mceval_java_all = DatasetDict({
    "train": ds_java_train,
    "test": mceval_java["test"]
})

mceval_java_mc = DatasetDict({
    "train": mceval_ins_java["train"],
    "test": mceval_java["test"]
})

print(mceval_java_all)
print(mceval_python_all)

mceval_java_all["test"].to_json("./Tools/mceval_java.jsonl", orient="records", lines=True, force_ascii=False)
mceval_python_all["test"].to_json("./Tools/mceval_python.jsonl", orient="records", lines=True, force_ascii=False)