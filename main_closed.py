from closed_batch_generation import evaluate_generation_openai_batch
from Prompts.gen_prompts import gen_prompts as GEN
from Tools.codereval import codereval_java, codereval_python


if __name__ == "__main__":
    models = [
        "gpt-5.1",
        "gpt-5.1-mini",
    ]

    methods = ["zero", "naive", "retrieval"]
    example_nums = [0, 3]
    max_lengths = {0: 1024, 3: 8196}
    indexs = [0, 1, 2, 3, 4]
    generation_runs = [
        {"output_root": "generation_results_codereval-0k1f", "temperature": 0.0, "num_samples": 1},
        {
            "output_root": "generation_results_codereval-1k5f",
            "temperature": 1.0,
            "num_samples": 5,
            "first_sample_output_root": "generation_results_codereval-1k1f",
        },
    ]

    for run in generation_runs:
        for model in models:
            for method in methods:
                for example_num in example_nums:
                    if method == "zero" and example_num != 0:
                        continue
                    if method == "retrieval" and example_num == 0:
                        continue
                    if method == "naive" and example_num == 0:
                        continue
                    if method == "cot" and example_num == 0:
                        continue
                    for index in indexs:
                        evaluate_generation_openai_batch(
                            model,
                            method,
                            example_num=example_num,
                            max_length=max_lengths[example_num],
                            system_prompt=GEN[index],
                            dataset_generation=codereval_java,
                            datatype=2,
                            output_root=run["output_root"],
                            temperature=run["temperature"],
                            num_samples=run["num_samples"],
                            first_sample_output_root=run.get("first_sample_output_root"),
                        )
                        evaluate_generation_openai_batch(
                            model,
                            method,
                            example_num=example_num,
                            max_length=max_lengths[example_num],
                            system_prompt=GEN[index],
                            dataset_generation=codereval_python,
                            datatype=3,
                            output_root=run["output_root"],
                            temperature=run["temperature"],
                            num_samples=run["num_samples"],
                            first_sample_output_root=run.get("first_sample_output_root"),
                        )
