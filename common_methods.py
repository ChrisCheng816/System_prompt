import re
import gc
import os
import torch
import json
import scann
import subprocess
import numpy as np
import logging
import torch.nn.functional as F
from codebleu import calc_codebleu
import evaluate
from exact_match import em_compute, exact_match_no_punct
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel

bleu_metric = evaluate.load("bleu")
em_metric = evaluate.load("exact_match")
# calc_codebleu = evaluate.load("dvitel/codebleu")
# ---------------------
# Public methods
# ---------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
logging.getLogger("vllm").setLevel(logging.ERROR)

def load_model(model_name, dataset):

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    # model = torch.compile(model)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,   # 三张 GPU
        dtype="auto",
        gpu_memory_utilization=0.95        # FP16 / BF16 自动选择
    )
    # print(f"Model dtype: {next(llm.parameters()).dtype}")
    # print(f"Model loaded into {llm.device} successfully")
    batch_size = 32
    return tokenizer, llm, batch_size

def load_prompt(length, task_description, src_key, tgt_key, test_data, base_prompt, tokenizer, system_prompt, model, max_length=4096):
    references = []
    prompts = []
    counter = 0
    for idx in range(length):
        counter += 1
        if counter % 500 == 0:
            print(f"Processing \033[1;32m{counter}\033[0m instances...")
        input_block = f"{task_description}Input:\n{test_data[src_key][idx]}\nOutput:\n"
        if isinstance(base_prompt, list):
            prompt = base_prompt[idx]
        else:
            prompt = base_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt}{input_block}"}
        ]

        prompts.append(messages)

        if tgt_key == "docstring_tokens":
            references.append(smart_join(test_data[tgt_key][idx]))
        else:
            references.append(test_data[tgt_key][idx].strip())

    full_prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True, return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=max_length)
    return full_prompt, references

# def load_prompt(length, task_description, src_key, tgt_key, test_data, base_prompt, system_prompt):
#     references = []
#     prompts = []
#     counter = 0
#     for idx in range(length):
#         counter += 1
#         if counter % 500 == 0:
#             print(f"Processing \033[1;32m{counter}\033[0m instances...")
#         input_block = f"{task_description}Input:\n{test_data[src_key][idx]}\nOutput:\n"
#         if isinstance(base_prompt, list):
#             prompt = base_prompt[idx]
#         else:
#             prompt = base_prompt
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"{prompt}{input_block}"}
#         ]

#         prompts.append(messages)

#         if tgt_key == "docstring_tokens":
#             references.append(smart_join(test_data[tgt_key][idx]))
#         else:
#             references.append(test_data[tgt_key][idx].strip())
#     return prompts, references

def evaluate_metric_sum(predictions, references, path):
    counter = 0
    while os.path.exists(f"{path}_{counter}"):
        counter += 1
    filepath = f"{path}_{counter}"
    os.makedirs(filepath, exist_ok=True)
    with open(f"{filepath}/predictions.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            f.write(f"{i}\t{item}\n")

    with open(f"{filepath}/references.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(references):
            f.write(f"{i}\t{item}\n")
    try:
        cmd = f"python3 evaluator_ct/evaluator.py {filepath}/references.txt < {filepath}/predictions.txt"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except Exception as e:
        print(f"Error occurred while running evaluation: {e}")

    bleu_result = bleu_metric.compute(predictions=predictions, references=references, smooth=True)

    return filepath, result.stdout, bleu_result

def evaluate_metric_tran(predictions, references, path, lang):
    for i, item in enumerate(predictions):
        predictions[i] = clean_code_blocks(item)
        
    counter = 0
    while os.path.exists(f"{path}_{counter}"):
        counter += 1
    filepath = f"{path}_{counter}"
    os.makedirs(filepath, exist_ok=True)
    with open(f"{filepath}/predictions.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            lines = [ln.strip() for ln in item.splitlines() if ln.strip()]
            s = "".join(lines)
            f.write(f"{s}\n")

    with open(f"{filepath}/references.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(references):
            f.write(f"{item}\n")

    try:
        # cmd = f"python3 evaluator_cc/evaluator.py -ref {filepath}/references.txt -pre {filepath}/predictions.txt"
        # result1 = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        cmd = f"python3 calc_code_bleu.py --refs {filepath}/references.txt --hyp {filepath}/predictions.txt --lang {lang} --params 0.25,0.25,0.25,0.25"
        codebleu = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except Exception as e:
        print(f"Error occurred while running evaluation: {e}")

    BLEU_Smooth = bleu_metric.compute(predictions=predictions, references=references, smooth=True)

    BLEU = bleu_metric.compute(predictions=predictions, references=references, smooth=False)
    print(f"BLEU: {BLEU['bleu']*100}")
    EM = em_compute(predictions, references)

    codebleu_score = calc_codebleu(references, predictions, lang)
    ngram_match, weighted_ngram_match = extract_value(codebleu.stdout)

    CodeBleu_result = ngram_match*25 + weighted_ngram_match*25 + codebleu_score["syntax_match_score"]*25 + codebleu_score["dataflow_match_score"]*25
    CodeBleu = {'codebleu': CodeBleu_result, 'ngram_match_score': ngram_match, 'weighted_ngram_match_score': weighted_ngram_match, 'syntax_match_score': codebleu_score["syntax_match_score"], 'dataflow_match_score': codebleu_score["dataflow_match_score"]}
    
    return filepath, BLEU_Smooth, EM, CodeBleu

def extract_value(text):
    pattern = r"ngram match:\s*([0-9.]+),\s*weighted ngram match:\s*([0-9.]+)"

    match = re.search(pattern, text)
    if match:
        ngram_match = float(match.group(1))
        weighted_ngram_match = float(match.group(2))
        return ngram_match, weighted_ngram_match
    else:
        print("No match found")


def compute_metric(prompts, batch_size, tokenizer, model, references, max_length):
    predictions = []
    counter = 1
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_predictions = run_batch(batch_prompts, tokenizer, model, max_length, 512)
        predictions.extend(batch_predictions)

        if (i // 200) == counter:
            print(f"\033[1;32m{i}\033[0m instances generated successfully")
            counter += 1

        # Strong explicit cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Starting to compute...")

    return predictions

def compute_metric_tran(prompts, batch_size, tokenizer, model, references, max_length, lang):
    predictions = []
    counter = 1
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_predictions = run_batch(batch_prompts, tokenizer, model, max_length, 512)
        predictions.extend(batch_predictions)

        if (i // 200) == counter:
            print(f"\033[1;32m{i}\033[0m instances generated successfully")
            counter += 1
        # Strong explicit cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    for i, item in enumerate(predictions):
        predictions[i] = [ln.strip() for ln in item.splitlines() if ln.strip()]
        predictions[i] = "".join(predictions[i])

    print("Starting to compute...")

    return predictions

def run_batch(batch_prompts, tokenizer, model, input_max_len, output_max_tokens):
    predictions = []
    # Tokenize with padding and truncation
    if tokenizer.pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    # model_inputs = tokenizer.apply_chat_template(
    #     batch_prompts,
    #     return_tensors="pt",
    #     padding=True,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     padding_side="left",
    #     truncation=True,
    #     max_length=input_max_len,
    # )

    # with torch.inference_mode():
    #     outputs = model.generate(
    #         **model_inputs,
    #         max_new_tokens=output_max_tokens,
    #         eos_token_id=tokenizer.eos_token_id,
    #         pad_token_id=pad_token_id,
    #         use_cache=True
    #     )

    sampling_params = SamplingParams(
        max_tokens=output_max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    )

    outputs = model.generate(batch_prompts, sampling_params)
    # Decode input and output to strings

    predictions = [output.outputs[0].text.strip() for output in outputs]

    # decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # input_texts = tokenizer.batch_decode(model_inputs.input_ids, skip_special_tokens=True)
    
    # for prompt, decoded in zip(batch_prompts, decoded_batch):
    #     pred = decoded.outputs[0].text[len(prompt):].strip()
    #     predictions.append(pred)

    # All local variables will be released once function returns
    return predictions

# Construct Prompt
def build_prompt(top_k_examples):
    prompt = ""
    counter = 0
    for ex in top_k_examples:
        counter = counter + 1
        prompt += f"### Example {counter}:\nInput:\n{ex['source_code']}\nOutput:\n{ex['target_code']}\n"
    return prompt

def get_retrieval_prompt(query_code_arr, example_db, k=3):
    retriever = SentenceTransformer("BAAI/bge-code-v1", device="cuda:2")
    support_inputs = [item["source_code"] for item in example_db]
    support_embeddings = retriever.encode(support_inputs, normalize_embeddings=True).astype("float32")

    # searcher = scann.scann_ops_pybind.builder(support_embeddings, k, "dot_product") \
    #     .tree(num_leaves=50, num_leaves_to_search=10, training_sample_size=len(support_embeddings)) \
    #     .score_ah(2, anisotropic_quantization_threshold=0.2) \
    #     .reorder(20) \
    #     .build()
    searcher = scann.scann_ops_pybind.builder(support_embeddings, k, "dot_product") \
        .score_brute_force() \
        .build()

    prompts = []
    counter = 0
    for query_code in query_code_arr:
        counter += 1
        query_emb = retriever.encode([query_code], normalize_embeddings=True).astype("float32")[0]
        neighbors, distances = searcher.search(query_emb)
        retrieved_examples = [example_db[i] for i in neighbors]
        final_prompt = build_prompt(retrieved_examples)
        prompts.append(final_prompt)
        if counter % 200 == 0:
            print(f"Retrieved \033[1;32m{counter}\033[0m queries")

    return prompts

def print_info(model_name, style, example_num, system_prompt, language=None, direction=None):
    var = "Direction" if direction is not None else "Language"
    value = direction if direction is not None else language
    model_base = re.split(r"[\\/]", model_name)[-1]
    print(f"Model:{model_base} Style:{style} Example_Number:{example_num} {var}:{value}")
    print(f"\033[34m{system_prompt}\033[0m")

def smart_join(arr):
    result = ''
    for i, word in enumerate(arr):
        if word in {'.', ',', '?', '!', ';', ':'}:
            result = result.rstrip() + word  # 去掉末尾空格再加标点
        else:
            result += word + ' '
    return result.strip()

def clean_code_blocks(text: str) -> str:
    # 去掉开头的 ```csharp 或 ```java（忽略大小写，允许后面有空格或换行）
    text = re.sub(r"^```(?:csharp|java)\s*\n?", "", text, flags=re.IGNORECASE)
    # 去掉结尾的 ```（可能单独一行，也可能直接接在代码后面）
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()

def check_prompt(number, language, task):
    prompts_file = f"Intermediate_output/{task}/{number}_{language}.jsonl"
    return os.path.exists(prompts_file)

def save_prompt(number, language, task, prompts):
    # Save prompts and references to disk
    os.makedirs(f"Intermediate_output/{task}", exist_ok=True)
    with open(f"Intermediate_output/{task}/{number}_{language}.jsonl", "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"Prompts saved for {language}.")

def extract_prompt(number, language, task, prompt=None):
    with open(f"Intermediate_output/{task}/{number}_{language}.jsonl", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    return prompts
