import random
import time
from common_methods import *
from datasets import load_dataset
import thought_chain as t
from datasets import DatasetDict
from datetime import timedelta
from model_map import model_map
from generate_prompts import generate_translation_prompt, generate_summarization_prompt, generate_bugfixing_prompt

GREEN = "\033[92m"
RESET = "\033[0m"

# ---------------------
# Importing data sets and metrics
# ---------------------
dataset_translation = load_dataset("google/code_x_glue_cc_code_to_code_trans")

# ---------------------
# Code summarization
# ---------------------

def evaluate_summarization(model_name, style, example_num = None, test_num = None, max_length=256, shuffled = False, system_prompt = None, dataset_summarization = None):
    base_prompt = ""
    language = dataset_summarization['test']['language'][0] if isinstance(dataset_summarization, DatasetDict) else dataset_summarization['language'][0]
    print_info(model_name, style, example_num, system_prompt, language = language, direction = None)
    tokenizer, model, batch_size = load_model(model_name, dataset_summarization)
    train_data = dataset_summarization["train"].select(range(example_num)) if example_num else dataset_summarization["train"]
    test_data = dataset_summarization["test"].select(range(test_num)) if test_num else dataset_summarization["test"]
    # Build shared prompt using training examples
    counter = [0]
    if style == "naive" or style == "cot":
        base_prompt = train_data.map(lambda e: generate_summarization_prompt(e, style, counter, t.summarization[language]), load_from_cache_file=False)["prompt"]
        pre_prompt = "The following are a few example(s) for code summarization.\n" if style == "naive" else "The following are a few example(s) with thought steps for code summarization.\n"
        base_prompt = "".join(base_prompt)
        base_prompt = pre_prompt + base_prompt
        print(f"Counter = {counter[0]}")
    elif style == "retrieval":
        if check_prompt(example_num, language, task="summarization") == False:
            example_db = [
                {
                    "source_code": dataset_summarization["train"][i]["code"],
                    "target_code": smart_join(dataset_summarization["train"][i]["docstring_tokens"])  # Join tokens into a single string
                }
                for i in range(5000)
            ]
            print("Starting to integrate example database...")
            query_code_arr = [test_data["code"][i] for i in range(len(test_data["code"]))]
            base_prompt = get_retrieval_prompt(query_code_arr, example_db, example_num)
            pre_prompt = "The following are a few retrieval-based example(s) for code summarization."
            base_prompt = [pre_prompt + prompt for prompt in base_prompt]
            print("Retrieval data integration completed")
            save_prompt(example_num, language, task = "summarization", prompts = base_prompt)
        else:
            base_prompt = extract_prompt(example_num, language, task = "summarization")

    after_description = "Let's think step-by-step to understand this method first, as shown in the example(s) if provided. Please do not output your thought steps if exist, just output the answer directly ###\n" if style == "cot" else "Please output the answer directly as shown in the example(s) if provided.###\n"
    task_description = f"### It is your turn now! Summarizing the follwing code into summary. {after_description}"

    if system_prompt == None:
        system_prompt = "You are a software documentation assistant. Given a code snippet, your task is to generate a concise and informative natural language summary that describes the purpose and behavior of the code."
    src_key, tgt_key = "code", "docstring_tokens"

    predictions= []
    print(f"Loading {len(test_data[src_key])} prompts...")
    prompts, references = load_prompt(len(test_data[src_key]), task_description, src_key, tgt_key, test_data, base_prompt, tokenizer, system_prompt, model, max_length)
    print("Prompts loaded successfully")
    prompts = random.sample(prompts, len(prompts)) if shuffled == True else prompts
    print("Starting to generate...")
    start_time = time.time()
    predictions = compute_metric(prompts, batch_size, tokenizer, model, references, max_length)
    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))

    filepath, result, bleu_result = evaluate_metric_sum(predictions=predictions, references=references, path=f"summarization_results/{model_map[model_name]}_{language}_{style}_{example_num}-shot")
    print("---------BLEU RESULT------------")
    print(f"Normal {result}, HF: {bleu_result}")
    with open(f"{filepath}/output.txt", "a", encoding="utf-8") as f:
        f.write(f"{model_name} Language:{dataset_summarization['test']['language'][0]} {style} {example_num}-shot counter={counter[0]} Time: {elapsed_time}:\n{system_prompt}\n\nBleu: {result}\nHFBleu: {bleu_result}\n")
    del train_data, test_data, base_prompt, prompts, references, predictions, model, tokenizer, result
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ---------------------
# Code translation
# ---------------------

def evaluate_translation(model_name, style, example_num = None, test_num = None, max_length=256, shuffled = False, system_prompt = None, order = None):
    language = "java" if order == 0 else "cs"
    lang = "c_sharp" if order == 0 else "java"
    direction = "Java2Cs" if order == 0 else "Cs2Java"
    print_info(model_name, style, example_num, system_prompt, direction = direction)
    tokenizer, model, batch_size = load_model(model_name, dataset_translation)
    train_data = dataset_translation["train"].select(range(example_num)) if example_num else dataset_translation["train"]
    test_data = dataset_translation["test"].select(range(test_num)) if test_num else dataset_translation["test"]
    counter = [0]
    # Build shared prompt using training examples
    if style == "naive" or style == "cot":
        base_prompt = train_data.map(lambda e: generate_translation_prompt(e, style, counter, t.translation[language], order))["prompt"]
        pre_prompt = "The following are a few example(s) for code translation.\n" if style == "naive" else "The following are a few example(s) with thought steps for code translation.\n"
        base_prompt = "".join(base_prompt)
        base_prompt = pre_prompt + base_prompt
        print(f"Counter = {counter[0]}")
    elif style == "retrieval":
        if check_prompt(example_num, language, task="translation") == False:
            if order == 0:
                example_db = [
                    {
                        "source_code": dataset_translation["train"][i]["java"],
                        "target_code": dataset_translation["train"][i]["cs"]
                    }
                    for i in range(5000)
                ]
                query_code_arr = [test_data["java"][i] for i in range(len(test_data["java"]))]
                base_prompt = get_retrieval_prompt(query_code_arr, example_db, example_num)
                pre_prompt = "The following are a few retrieval-based example(s) for code translation."
                base_prompt = [pre_prompt + prompt for prompt in base_prompt]
            else:
                example_db = [
                    {
                        "source_code": dataset_translation["train"][i]["cs"],
                        "target_code": dataset_translation["train"][i]["java"]
                    }
                    for i in range(3000)
                ]
                query_code_arr = [test_data["cs"][i] for i in range(len(test_data["cs"]))]
                base_prompt = get_retrieval_prompt(query_code_arr, example_db, example_num)
                pre_prompt = "The following are a few retrieval-based example(s) for code translation."
                base_prompt = [pre_prompt + prompt for prompt in base_prompt]
            save_prompt(example_num, language, task = "translation", prompts = base_prompt)
        else:
            base_prompt = extract_prompt(example_num, language, task = "translation")
    else:
        base_prompt = ""

    after_description = "Let's think step-by-step to understand this translation first, as shown in the example(s) if provided. Please do not output your thought steps if exist, just output the answer directly." if style == "cot" else "Please output the answer directly as shown in the example(s) if provided."
    task_description = f"### It is your turn now! {after_description} Translate the following Java code into Csharp code.\n" if order == 0 else f"### It is your turn now! {after_description} Translate the following Csharp code into Java code.\n"
    if system_prompt == None:
        system_prompt = "You are a professional code translator trained to convert source code between programming languages while preserving the original behavior and semantics."
    
    src_key = "java" if order == 0 else "cs"
    tgt_key = "cs" if order == 0 else "java"

    predictions= []
    print(f"Loading {len(test_data[src_key])} prompts...")
    prompts, references = load_prompt(len(test_data[src_key]), task_description, src_key, tgt_key, test_data, base_prompt, tokenizer, system_prompt, model, max_length)
    print(f"{len(prompts)} Prompts loaded successfully")
    prompts = random.sample(prompts, len(prompts)) if shuffled == True else prompts
    print("Starting to generate...")
    start_time = time.time()
    predictions = compute_metric_tran(prompts, batch_size, tokenizer, model, references, max_length, lang)
    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))

    filepath, BLEU_Smooth, EM, CodeBleu = evaluate_metric_tran(predictions=predictions, references=references, path=f"translation_results/{model_map[model_name]}_{direction}_{style}_{example_num}-shot", lang=lang)
    print("----------BLEU RESULT------------")
    print(BLEU_Smooth)

    with open(f"{filepath}/output.txt", "a", encoding="utf-8") as f:
        f.write(f"{model_name} Direction:{direction} {style} {example_num}-shot counter={counter[0]} Time: {elapsed_time}:\n{system_prompt}\n\nBLEU_Smooth:{BLEU_Smooth}\nEM: {EM}\nCodeBleu: {CodeBleu}\n")
    del train_data, test_data, base_prompt, prompts, references, predictions, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ---------------------
# bugfixing
# ---------------------
def evaluate_bugfixing(model_name, style, example_num = None, test_num = None, max_length=256, shuffled = False, system_prompt = None, dataset_refinement = None):
    tokenizer, model, batch_size = load_model(model_name, dataset_refinement)
    train_data = dataset_refinement["train"].select(range(example_num)) if example_num else dataset_refinement["train"]
    test_data = dataset_refinement["test"].select(range(test_num)) if test_num else dataset_refinement["test"]
    # Build shared prompt using training examples
    if style == "naive" or style == "cot":
        base_prompt = train_data.map(lambda e: generate_bugfixing_prompt(e, style))["prompt"]
        base_prompt = "".join(base_prompt)
    elif style == "retrieval":
        example_db = [
            {
                "source_code": dataset_refinement["train"][i]["buggy"],
                "target_code": dataset_refinement["train"][i]["fixed"]
            }
            for i in range(50)
        ]
        query_code_arr = [test_data["buggy"][i] for i in range(len(test_data["buggy"]))]
        base_prompt = get_retrieval_prompt(query_code_arr, example_db, len(example_num))
    else:
        base_prompt = ""
    
    task_description = "Now, fix the following Java code to make it correct.\n"
    if system_prompt == None: 
        system_prompt = "You are a code repair assistant. Your task is to fix buggy code by correcting syntax or logic errors while preserving the original intent."
    src_key, tgt_key, src_label, tgt_label = "buggy", "fixed", "Buggy", "Fixed"

    predictions= []
    
    prompts, references = load_prompt(len(test_data[src_key]), task_description, src_key, tgt_key, src_label, tgt_label, test_data, base_prompt, tokenizer, system_prompt)
    prompts = random.sample(prompts, len(prompts)) if shuffled == True else prompts
    bleu_result = compute_metric(prompts, batch_size, tokenizer, model, references, max_length)

    print("---------Bleu result------------")
    print(bleu_result)
    del train_data, test_data, base_prompt, prompts, references, predictions, model, tokenizer, bleu_result
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()