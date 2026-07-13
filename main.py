import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import evaluate
from datasets import load_dataset
from task_evaluation import evaluate_translation, evaluate_summarization, evaluate_generation, generation_test
from Prompts.summary_prompts import summary_prompts as SUM
from Prompts.trans_prompts import trans_prompts as TRAN
from Prompts.gen_prompts import gen_prompts as GEN
from Tools.mceval import mceval_java_all, mceval_python_all, mceval_java_mc
from Tools.codereval import codereval_java, codereval_python

# ---------------------
# Importing data sets and metrics
# ---------------------

dataset_summarization_java = load_dataset("google/code_x_glue_ct_code_to_text", "java")
dataset_summarization_python = load_dataset("google/code_x_glue_ct_code_to_text", "python")

if __name__ == "__main__":
    # ---------------------
    # Executing tasks
    # ---------------------
    models = [
        # "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        # "Qwen/Qwen2.5-Coder-3B-Instruct",
        # "Qwen/Qwen2.5-Coder-7B-Instruct",
        # "Qwen/Qwen2.5-Coder-32B-Instruct",
        # "Qwen/Qwen2.5-Coder-14B-Instruct",
        # "openai/gpt-oss-20b",
        # "unsloth/Meta-Llama-3.1-8B-Instruct"
        # "deepseek-ai/deepseek-coder-33b-instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct"
    ]

    methods = ["zero","naive","retrieval"]
    example_nums = [0, 3]
    max_lengths = {0: 1024, 3: 8196}
    indexs = [0,1,2,3,4]
    generation_runs = [
        # {"output_root": "generation_results_codereval-0k1f", "temperature": 0.0, "num_samples": 1},
        # {"output_root": "generation_results_codereval-1k1f", "temperature": 1.0, "num_samples": 1},
        {"output_root": "generation_results_codereval-1k5f", "temperature": 1.0, "num_samples": 5},
    ]

    for run in generation_runs:
        for model in models:
            for method in methods:
                for example_num in example_nums:
                    if method == "zero" and example_num != 0:
                        continue  # zero 只有 example_num=0 的情况
                    if method == "retrieval" and example_num == 0:
                        continue  # retrieval 只有 example_num=1 和 3 的情况
                    if method == "naive" and example_num == 0:
                        continue  # naive 只有 example_num=1 和 3 的情况
                    if method == "cot" and example_num == 0:
                        continue  # cot 只有 example_num=1 和 3 的情况
                    for index in indexs:
                        if method == "zero" and index in [0, 1]:
                            continue  # deepseek-6.7b 1k5f zero 的 index 0,1 已完成
                        # evaluate_generation(model, method, example_num=example_num, max_length=max_lengths[example_num], system_prompt = GEN[index], dataset_generation = mceval_python_all, datatype = 1)
                        evaluate_generation(model, method, example_num=example_num, max_length=max_lengths[example_num], system_prompt = GEN[index], dataset_generation = codereval_java, datatype = 2, output_root=run["output_root"], temperature=run["temperature"], num_samples=run["num_samples"])
                        evaluate_generation(model, method, example_num=example_num, max_length=max_lengths[example_num], system_prompt = GEN[index], dataset_generation = codereval_python, datatype = 3, output_root=run["output_root"], temperature=run["temperature"], num_samples=run["num_samples"])
