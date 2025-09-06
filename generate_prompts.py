import thought_chain as t

def generate_summarization_prompt(example, prompt_type, counter, steps):
    if counter is not None:
        counter[0] = counter[0] + 1
    if prompt_type == "naive":
        print(f"Summarization counter to {counter[0]} with naive prompt")
        prompt = f"### Example {counter[0]}:\nInput:\n{example['code']}\nOutput:\n{smart_join(example['docstring_tokens'])}\n\n"
    elif prompt_type == "cot":
        print(f"Summarization counter to {counter[0]} with cot prompt")
        print(steps[counter[0] - 1])
        prompt = (
            f"### Example {counter[0]}:\n Input:\n{example['code']}\n"
            f"Thought steps:\n{steps[counter[0] - 1]}\n"
            f"Output:\n{smart_join(example['docstring_tokens'])}\n\n"
        )
    else:
        raise ValueError("Unsupported prompt type.")
        
    return {"prompt":prompt}

def generate_bugfixing_prompt(example, prompt_type):
    if prompt_type == "naive":
        prompt = f"### Example\nInput:\n{example['buggy']}\nOutput:\n{example['fixed']}\n"
    elif prompt_type == "cot":
        prompt = (
            f"### Example {counter[0]}:\n Input:\n{example['buggy']}.\n"
            f"Thought steps:\n{t.summarization_java_1}\n"
            f"Output:\n{example['fixed']}\n"
        )
    else:
        raise ValueError("Unsupported prompt type.")
        
    return {"prompt":prompt}

def generate_translation_prompt(example, prompt_type, counter, steps, order):
    cur = "java" if order == 0 else "cs"
    tar = "cs" if order == 0 else "java"
    if counter is not None:
        counter[0] = counter[0] + 1
    if prompt_type == "naive":
        prompt = f"### Example\nInput:\n{example[cur]}\nOutput:\n{example[tar]}\n"
    elif prompt_type == "cot":
        print(f"Translation counter to {counter[0]} with cot prompt")
        print(steps[counter[0] - 1])
        prompt = (
            f"### Example {counter[0]}:\n Input:\n{example[cur]}.\n"
            f"Thought steps:\n{steps[counter[0] - 1]}\n"
            f"Output:\n{example[tar]}\n\n"
        )
    else:
        raise ValueError("Unsupported prompt type.")
        
    return {"prompt":prompt}

def smart_join(arr):
    result = ''
    for i, word in enumerate(arr):
        if word in {'.', ',', '?', '!', ';', ':'}:
            result = result.rstrip() + word  # 去掉末尾空格再加标点
        else:
            result += word + ' '
    return result.strip()