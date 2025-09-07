import evaluate
from datasets import load_dataset
from task_evaluation import evaluate_translation, evaluate_summarization, evaluate_bugfixing
from Prompts.summary_prompts import summary_prompts as SUM
from Prompts.trans_prompts import trans_prompts as TRAN

# ---------------------
# Importing data sets and metrics
# ---------------------

dataset_summarization_java = load_dataset("google/code_x_glue_ct_code_to_text", "java")
dataset_summarization_python = load_dataset("google/code_x_glue_ct_code_to_text", "python")
dataset_summarization_php = load_dataset("google/code_x_glue_ct_code_to_text", "php")
dataset_summarization_js = load_dataset("google/code_x_glue_ct_code_to_text", "javascript")
dataset_summarization_go = load_dataset("google/code_x_glue_ct_code_to_text", "go")
dataset_summarization_ruby = load_dataset("google/code_x_glue_ct_code_to_text", "ruby")

if __name__ == "__main__":
    # ---------------------
    # Executing tasks
    # ---------------------
    # evaluate_summarization("Qwen/Qwen2.5-Coder-7B-Instruct", "zero", example_num=0, test_num=1, max_length=1024, shuffled = False, system_prompt = SUM[5], dataset_summarization=dataset_summarization_python)
    # evaluate_summarization("Qwen/Qwen2.5-Coder-7B-Instruct", "retrieval", example_num=3, max_length=4096, system_prompt = SUM[0], dataset_summarization=dataset_summarization_go)
    # evaluate_summarization("Qwen/Qwen3-Coder-30B-A3B-Instruct", "naive", example_num=3, max_length=4096, system_prompt = SUM[0], dataset_summarization=dataset_summarization_go)


    # evaluate_summarization("Qwen/Qwen2.5-Coder-7B-Instruct", "retrieval", example_num=1, max_length=2048, system_prompt = SUM[0], dataset_summarization=dataset_summarization_python)
    # evaluate_summarization("codellama/CodeLlama-13b-Instruct-hf", "retrieval", example_num=1, max_length=2048, system_prompt = SUM[0], dataset_summarization=dataset_summarization_python)
    # evaluate_summarization("Qwen/Qwen2.5-Coder-7B-Instruct", "retrieval", example_num=3, max_length=4096, system_prompt = SUM[0], dataset_summarization=dataset_summarization_python)
    # evaluate_summarization("codellama/CodeLlama-13b-Instruct-hf", "naive", example_num=3, max_length=4096, system_prompt = SUM[3], dataset_summarization=dataset_summarization_java)
    # evaluate_summarization("codellama/CodeLlama-13b-Instruct-hf", "naive", example_num=3, max_length=4096, system_prompt = SUM[3], dataset_summarization=dataset_summarization_python)
    # evaluate_summarization("codellama/CodeLlama-13b-Instruct-hf", "cot", example_num=3, max_length=4096, system_prompt = SUM[1], dataset_summarization=dataset_summarization_python)
    # evaluate_summarization("codellama/CodeLlama-13b-Instruct-hf", "zero", example_num=0, max_length=2048, system_prompt = SUM[3], dataset_summarization=dataset_summarization_python)
    # evaluate_summarization("codellama/CodeLlama-13b-Instruct-hf", "cot", example_num=3, max_length=4096, system_prompt = SUM[1], dataset_summarization=dataset_summarization_java)
    # evaluate_summarization("codellama/CodeLlama-13b-Instruct-hf", "zero", example_num=0, max_length=2048, system_prompt = SUM[3], dataset_summarization=dataset_summarization_java) 
    
    # evaluate_translation("Qwen/Qwen2.5-Coder-7B-Instruct", "zero", example_num=0, max_length=1024, system_prompt = TRAN[2], order = 0)
    # evaluate_translation("Qwen/Qwen3-Coder-30B-A3B-Instruct", "retrieval", example_num=3, max_length=4096, system_prompt = TRAN[2], order = 1)

    # evaluate_translation("Qwen/Qwen2.5-Coder-7B-Instruct", "naive", example_num=1, test_num=16, max_length=1024, system_prompt = TRAN[1], order = 0)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "zero", example_num=0, max_length=1024, system_prompt = TRAN[1], order = 0)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "cot", example_num=1, max_length=4096, system_prompt = TRAN[1], order = 1)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "naive", example_num=3, max_length=4096, system_prompt = TRAN[2], order = 1)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "naive", example_num=3, max_length=4096, system_prompt = TRAN[2], order = 0)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "cot", example_num=3, max_length=4096, system_prompt = TRAN[2], order = 1)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "cot", example_num=3, max_length=4096, system_prompt = TRAN[2], order = 0)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "cot", example_num=1, max_length=4096, system_prompt = TRAN[2], order = 1)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "cot", example_num=1, max_length=4096, system_prompt = TRAN[2], order = 0)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "retrieval", example_num=1, max_length=2048, system_prompt = TRAN[2], order = 0)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "retrieval", example_num=1, max_length=2048, system_prompt = TRAN[2], order = 1)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "retrieval", example_num=3, max_length=4096, system_prompt = TRAN[2], order = 0)
    # evaluate_translation("codellama/CodeLlama-13b-Instruct-hf", "retrieval", example_num=3, max_length=4096, system_prompt = TRAN[2], order = 1)

    models = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        # "Qwen/Qwen2.5-Coder-32B-Instruct",
        "codellama/CodeLlama-13b-Instruct-hf",
        # "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        # "codellama/CodeLlama-7b-Instruct-hf",
        # "deepseek-ai/deepseek-coder-7b-instruct-v1.5"

        # "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    ]

    methods = ["retrieval"]
    example_nums = [0, 1, 3]
    max_lengths = {0: 1024, 1: 2048, 3: 4096}
    orders = [0, 1]
    indexs = [0, 1, 2, 3, 4]

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
                # for order in orders:
                #     for index in indexs:
                #         evaluate_translation(
                #             model,
                #             method,
                #             example_num=example_num,
                #             max_length=max_lengths[example_num],
                #             system_prompt=TRAN[index],
                #             order=order
                #         )

                # evaluate_summarization(
                #     model,
                #     method,
                #     example_num=example_num,
                #     max_length=max_lengths[example_num],
                #     system_prompt=SUM[3],
                #     dataset_summarization=dataset_summarization_java
                # )
                evaluate_summarization(
                    model,
                    method,
                    example_num=example_num,
                    max_length=max_lengths[example_num],
                    system_prompt=SUM[0],
                    dataset_summarization=dataset_summarization_python
                )
