import json
import os
import tempfile
import time
from datetime import timedelta

from openai import OpenAI

from model_map import model_map


OPENAI_BATCH_COMPLETION_WINDOW = "24h"
OPENAI_BATCH_POLL_SECONDS = 30
OPENAI_BATCH_ENDPOINT = "/v1/chat/completions"
OPENAI_MAX_COMPLETION_TOKENS = 2048


def generate_generation_prompt(example, prompt_type, counter):
    if counter is not None:
        counter[0] = counter[0] + 1
    if prompt_type == "naive":
        prompt = f"### Example {counter[0]}:\nInput:\n{example['instruction'].strip()}\nOutput:\n{example['output'].strip()}\n\n"
    else:
        raise ValueError("Unsupported prompt type.")

    return {"prompt": prompt}


def build_generation_messages(length, task_description, src_key, test_data, base_prompt, system_prompt):
    src_data = list(test_data[src_key])
    prompts = []
    counter = 0
    for idx in range(length):
        input_block = f"{task_description}Input:\n{src_data[idx]}\nOutput:\n"
        counter += 1
        if counter % 500 == 0:
            print(f"Processing \033[1;32m{counter}\033[0m instances...")

        if isinstance(base_prompt, list):
            prompt = base_prompt[idx]
        else:
            prompt = base_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt}{input_block}"},
        ]

        prompts.append(messages)

    return prompts


def build_prompt(top_k_examples):
    prompt = ""
    counter = 0
    for ex in top_k_examples:
        counter = counter + 1
        prompt += f"### Example {counter}:\nInput:\n{ex['source_code'].strip()}\nOutput:\n{ex['target_code'].strip()}\n\n"
    return prompt


def get_retrieval_prompt(query_code_arr, example_db, k=3):
    import torch
    from sentence_transformers import SentenceTransformer

    retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda:2")

    print("Retriever model loaded.")
    support_inputs = [item["source_code"] for item in example_db]
    query_inputs = query_code_arr

    support_embeddings = retriever.encode(support_inputs, normalize_embeddings=True)
    query_embeddings = retriever.encode(query_inputs, normalize_embeddings=True)

    print("Support set encoded.")
    similarity_matrix = retriever.similarity(query_embeddings, support_embeddings)
    print("Starting retrieval...")
    prompts = []
    top_k_sims_list = []
    for sim in similarity_matrix:
        topk = torch.topk(sim, k=k)
        top_k_idx = topk.indices.cpu().tolist()
        top_k_sims = topk.values.cpu().tolist()

        retrieved_examples = [example_db[idx] for idx in top_k_idx]
        final_prompt = build_prompt(retrieved_examples)
        prompts.append(final_prompt)
        top_k_sims_list.append(top_k_sims)

    return prompts, top_k_sims_list


def save_prompt(number, language, task, prompts, sims):
    os.makedirs(f"Intermediate_output/{task}", exist_ok=True)
    with open(f"Intermediate_output/{task}/{number}_{language}.jsonl", "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    with open(f"Intermediate_output/{task}/{number}_{language}_sims.jsonl", "w", encoding="utf-8") as f:
        json.dump(sims, f, ensure_ascii=False, indent=2)

    print(f"Prompts saved for {language}.")


def evaluate_metric_gen2(predictions, path, test_data=None, lang=None):
    counter = 0
    while os.path.exists(f"{path}_{counter}"):
        counter += 1
    filepath = f"{path}_{counter}"
    os.makedirs(filepath, exist_ok=True)

    ids = test_data["id"]
    with open(f"{filepath}/predictions.jsonl", "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            generate_results = item if isinstance(item, list) else [item]
            record = {"_id": ids[i], "generate_results": generate_results}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return filepath


def _first_sample_predictions(predictions):
    first_predictions = []
    for item in predictions:
        if isinstance(item, list):
            first_predictions.append([item[0] if item else "Unknown"])
        else:
            first_predictions.append([item])
    return first_predictions


def save_result_gen(filepath, model_name, lang, style, example_num, counter, elapsed_time, system_prompt, temperature=None, num_samples=None):
    with open(f"{filepath}/output.json", "a", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "language": lang,
                "style": style,
                "example_num": example_num,
                "counter": counter[0],
                "elapsed_time": elapsed_time,
                "temperature": temperature,
                "num_samples": num_samples,
                "system_prompt": system_prompt,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def generation_data_selector(datatype):
    if datatype == 0:
        source = "instruction"
        prompt = "prompt"
        output = "output"
        lang = "java"
        saving_name = "mceval_java"
    elif datatype == 1:
        source = "instruction"
        prompt = "prompt"
        output = "output"
        lang = "python"
        saving_name = "mceval_python"
    elif datatype == 2:
        source = "instruction"
        prompt = "instruction"
        output = "output"
        lang = "java"
        saving_name = "codereval_java"
    elif datatype == 3:
        source = "instruction"
        prompt = "instruction"
        output = "output"
        lang = "python"
        saving_name = "codereval_python"
    else:
        raise ValueError("Unsupported datatype.")

    return source, prompt, output, lang, saving_name


def _extract_text_from_chat_body(body):
    choices = body.get("choices") or []
    texts = []
    for choice in choices:
        message = (choice or {}).get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content.strip())
        elif isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"text", "output_text"}:
                    text = item.get("text") or item.get("output_text")
                    if text:
                        parts.append(str(text))
            texts.append("".join(parts).strip())
        elif content is None:
            texts.append("")
        else:
            texts.append(str(content).strip())
    return [text for text in texts if text]


def _read_openai_file_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        return bytes(content).decode("utf-8", errors="replace")

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    content_attr = getattr(content, "content", None)
    if isinstance(content_attr, (bytes, bytearray)):
        return bytes(content_attr).decode("utf-8", errors="replace")
    if isinstance(content_attr, str):
        return content_attr

    read_fn = getattr(content, "read", None)
    if callable(read_fn):
        raw = read_fn()
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw).decode("utf-8", errors="replace")

    return str(content)


def _summarize_openai_batch_errors(batch):
    errors = getattr(batch, "errors", None)
    data = getattr(errors, "data", None) if errors is not None else None
    if not data:
        return ""

    parts = []
    for err in data:
        code = getattr(err, "code", None)
        message = getattr(err, "message", None)
        if code and message:
            parts.append(f"code={code} message={message}")
        elif message:
            parts.append(str(message))
        elif code:
            parts.append(f"code={code}")
    return " | ".join(parts)


def _is_openai_batch_token_limit_error(exc):
    text = str(exc or "").lower()
    return "token_limit_exceeded" in text or "enqueued token limit reached" in text


def _is_openai_quota_exhausted_error(exc):
    text = str(exc or "").lower()
    return any(
        marker in text
        for marker in (
            "billing_hard_limit_reached",
            "billing hard limit has been reached",
            "insufficient_quota",
            "exceeded your current quota",
            "you exceeded your current quota",
        )
    )


def _build_batch_body(model_name, messages, temperature, num_samples):
    return {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "n": num_samples,
        "reasoning_effort": "none",
        "max_completion_tokens": OPENAI_MAX_COMPLETION_TOKENS,
    }


def _run_openai_batch(messages_list, model_name, temperature, num_samples, progress_label="instances"):
    if not messages_list:
        return []

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    request_rows = []
    for idx, messages in enumerate(messages_list):
        request_rows.append(
            {
                "custom_id": f"req-{idx}",
                "method": "POST",
                "url": OPENAI_BATCH_ENDPOINT,
                "body": _build_batch_body(model_name, messages, temperature, num_samples),
            }
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmpf:
            temp_path = tmpf.name
            for row in request_rows:
                tmpf.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(temp_path, "rb") as fh:
            input_file = client.files.create(file=fh, purpose="batch")

        batch = client.batches.create(
            input_file_id=input_file.id,
            endpoint=OPENAI_BATCH_ENDPOINT,
            completion_window=OPENAI_BATCH_COMPLETION_WINDOW,
        )
        batch_id = batch.id
        print(f"[INFO] OpenAI batch submitted for {progress_label}: id={batch_id}, size={len(messages_list)}")

        terminal_states = {"completed", "failed", "expired", "cancelled"}
        while True:
            batch = client.batches.retrieve(batch_id)
            status = getattr(batch, "status", None)
            if status in terminal_states:
                break
            print(f"[INFO] OpenAI batch {batch_id} status={status}")
            time.sleep(OPENAI_BATCH_POLL_SECONDS)

        if getattr(batch, "status", None) != "completed":
            error_summary = _summarize_openai_batch_errors(batch)
            if error_summary:
                raise RuntimeError(
                    f"Batch finished with status={getattr(batch, 'status', None)}. {error_summary}"
                )
            raise RuntimeError(f"Batch finished with status={getattr(batch, 'status', None)}")

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            for _ in range(6):
                time.sleep(2)
                batch = client.batches.retrieve(batch_id)
                output_file_id = getattr(batch, "output_file_id", None)
                if output_file_id:
                    break

        if not output_file_id:
            error_file_id = getattr(batch, "error_file_id", None)
            if error_file_id:
                error_content = client.files.content(error_file_id)
                error_text = _read_openai_file_text(error_content)
                first_error = ""
                for raw_line in (error_text or "").splitlines():
                    raw_line = raw_line.strip()
                    if raw_line:
                        first_error = raw_line
                        break
                raise RuntimeError(
                    f"Batch completed without output_file_id (error_file_id={error_file_id}). "
                    f"First error: {first_error or 'N/A'}"
                )
            raise RuntimeError("Batch completed without output_file_id and without error_file_id")

        output_content = client.files.content(output_file_id)
        output_text = _read_openai_file_text(output_content)

        predictions = [["Unknown"] for _ in range(len(messages_list))]
        for raw_line in output_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            custom_id = row.get("custom_id", "")
            if not custom_id.startswith("req-"):
                continue
            try:
                idx = int(custom_id.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if idx < 0 or idx >= len(predictions):
                continue

            if row.get("error"):
                predictions[idx] = ["Unknown"]
                continue

            body = ((row.get("response") or {}).get("body") or {})
            texts = _extract_text_from_chat_body(body)
            predictions[idx] = texts or ["Unknown"]

        return predictions
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _run_openai_batch_resilient(
    messages_list,
    model_name,
    temperature,
    num_samples,
    progress_label="instances",
):
    try:
        return _run_openai_batch(
            messages_list=messages_list,
            model_name=model_name,
            temperature=temperature,
            num_samples=num_samples,
            progress_label=progress_label,
        )
    except Exception as exc:
        if len(messages_list) <= 1 or not _is_openai_batch_token_limit_error(exc):
            raise

        midpoint = len(messages_list) // 2
        if midpoint <= 0 or midpoint >= len(messages_list):
            raise

        print(
            f"[WARN] OpenAI batch exceeded token limit for {progress_label}; "
            f"splitting {len(messages_list)} requests into {midpoint} + {len(messages_list) - midpoint}"
        )
        left = _run_openai_batch_resilient(
            messages_list[:midpoint],
            model_name=model_name,
            temperature=temperature,
            num_samples=num_samples,
            progress_label=f"{progress_label}-part1",
        )
        right = _run_openai_batch_resilient(
            messages_list[midpoint:],
            model_name=model_name,
            temperature=temperature,
            num_samples=num_samples,
            progress_label=f"{progress_label}-part2",
        )
        return left + right


def evaluate_generation_openai_batch(
    model_name,
    style,
    example_num=None,
    max_length=256,
    system_prompt=None,
    dataset_generation=None,
    datatype=None,
    output_root="generation_results_codereval",
    temperature=1.0,
    num_samples=5,
    first_sample_output_root=None,
):
    source, _prompt_input, output, lang, _saving_name = generation_data_selector(datatype)
    base_prompt = ""
    train_data = dataset_generation["train"].select(range(example_num)) if example_num else dataset_generation["train"]
    test_data = dataset_generation["test"]
    counter = [0]

    if style == "naive" or style == "cot":
        base_prompt = train_data.map(
            lambda e: generate_generation_prompt(e, style, counter),
            load_from_cache_file=False,
        )["prompt"]
        pre_prompt = (
            "The following are a few examples for code generation.\n"
            if style == "naive"
            else "The following are a few examples with thought steps for code generation.\n"
        )
        base_prompt = pre_prompt + "".join(base_prompt)
        print(f"Counter = {counter[0]}")
    elif style == "retrieval":
        example_db = [
            {
                "source_code": dataset_generation["train"][i][source],
                "target_code": dataset_generation["train"][i][output],
            }
            for i in range(100000)
        ]
        print("Example database constructed.")
        print("Starting to integrate example database...")
        base_prompt, top_k_sims_list = get_retrieval_prompt(test_data[source], example_db, example_num)
        pre_prompt = "The following are a few retrieval-based examples for code generation.\n"
        base_prompt = [pre_prompt + prompt for prompt in base_prompt]
        print("Retrieval data integration completed")
        save_prompt(example_num, lang, task="generation", prompts=base_prompt, sims=top_k_sims_list)

    after_description = (
        "Please only output the complete method directly as shown in the examples if provided. "
        "Do NOT include any import statements or class declarations. "
        "Do not wrap the output in markdown code fences or quote blocks, output raw code only.###\n"
    )
    task_description = f"### It is your turn now! Generating the code based on the instruction provided. {after_description}"

    print(
        f"Model:{model_name} Style:{style} Example_Number:{example_num} "
        f"Language:{lang} temperature={temperature} n={num_samples}"
    )
    print(f"\033[34m{system_prompt}\033[0m")
    print(f"Loading {len(test_data[source])} prompts...")
    messages_list = build_generation_messages(
        len(test_data[source]),
        task_description,
        source,
        test_data,
        base_prompt,
        system_prompt,
    )

    start_time = time.time()
    try:
        predictions = _run_openai_batch_resilient(
            messages_list=messages_list,
            model_name=model_name,
            temperature=temperature,
            num_samples=num_samples,
            progress_label=f"{model_name}-{lang}-{style}-{example_num}-shot-t{temperature}-n{num_samples}",
        )
    except Exception as exc:
        if _is_openai_quota_exhausted_error(exc):
            print("[ERROR] OpenAI quota or billing limit reached; no sequential fallback will be used.")
        raise
    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))

    filepath = evaluate_metric_gen2(
        predictions=predictions,
        path=f"{output_root}/predictions/{model_map[model_name]}_{lang}_{style}_{example_num}-shot",
        test_data=test_data,
        lang=lang,
    )
    save_result_gen(
        filepath,
        model_name,
        lang,
        style,
        example_num,
        counter,
        elapsed_time,
        system_prompt,
        temperature=temperature,
        num_samples=num_samples,
    )

    if first_sample_output_root is not None:
        first_sample_filepath = evaluate_metric_gen2(
            predictions=_first_sample_predictions(predictions),
            path=f"{first_sample_output_root}/predictions/{model_map[model_name]}_{lang}_{style}_{example_num}-shot",
            test_data=test_data,
            lang=lang,
        )
        save_result_gen(
            first_sample_filepath,
            model_name,
            lang,
            style,
            example_num,
            counter,
            elapsed_time,
            system_prompt,
            temperature=temperature,
            num_samples=1,
        )
        print(f"[INFO] First-sample derived output written to {first_sample_filepath}")

    return filepath
