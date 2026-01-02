import json
import os
import re

root_dir = "../generation_results_codereval/predictions"

def remove_assistant(code):
    if "assistantfinal" in code:
        return code.split("assistantfinal")[-1].strip()
    return code.strip()

def trim_before_return_marker(text: str) -> str:
    marker = "<|return|>"
    if marker in text:
        return text.split(marker)[0]
    return text

pattern = re.compile(r'^[ \t]*import\b[^;\n]*;[ \t]*', re.MULTILINE)

def remove_imports(code: str) -> str:
    while True:
        new_code = pattern.sub('', code)
        if new_code == code:
            return new_code
        code = new_code

def process_code(text: str) -> str:
    """
    Process code strings according to the new rules:
    - Remove leading ``` while preserving subsequent content
    - Remove any ``` appearing in the middle and truncate all content following it
    """
    result = text
    # Remove leading ```
    cut = False
    if result.startswith("```"):
        result = result[3:]
        cut = True

    # Find middle ```
    idx = result.rfind("```")
    if idx != -1 and cut:
        # Truncate ```
        result = result[:idx]
    
    return result.strip()

pattern_public_class_header = re.compile(
    r'public\s+class\s+[^\n{]+'              # public class 类名
    r'(?:\s+(?:extends|implements)\s+[^{]+)?'   # 可选 extends 或 implements
    r'\s*\{',                                   # 左花括号
    re.DOTALL,
)

def delete_last_public_class(code: str) -> str:
    code = code.rstrip()

    # 找最后一个 public class 头
    last_match = None
    for m in pattern_public_class_header.finditer(code):
        last_match = m

    if last_match is None:
        return code

    start = last_match.start()
    header_end = last_match.end()     # 这里已经包括了这个类的 {

    # 从 header_end 开始往后找和这个 { 配对的那个 }
    open_count = 1      # 我们已经看到一個 {
    close_count = 0
    i = header_end
    n = len(code)
    match_brace = None

    while i < n:
        ch = code[i]
        if ch == '{':
            open_count = open_count + 1
        elif ch == '}':
            close_count = close_count + 1
            if open_count == close_count:
                match_brace = i
                break
        i = i + 1

    # 如果没找到配对的 } 就不动
    if match_brace is None:
        return code

    prefix = code[: start]
    body = code[header_end: match_brace]

    return prefix + body

def delete_all_classes(code: str) -> str:
    code = code.rstrip()
    while True:
        new_code = delete_last_public_class(code)
        if new_code == code:
            break
        code = new_code
    return code

def extract_java_python_blocks(text: str) -> str:
    """
    Extract all ```java or ```python code block content from any position within the text and concatenate them in sequence.
    """
    # Match ```java or ```python at the beginning, until the next ``` ends
    pattern1 = r"```(java|python)\n(.*?)```"
    matches1 = re.findall(pattern1, text, re.DOTALL)
    pattern2 = r"```(java|python)\n(.*)"
    matches2 = re.findall(pattern2, text, re.DOTALL)
    if matches1:
        return "".join(code for _, code in matches1)
    elif matches2:
        return "".join(code for _, code in matches2)
    else:
        return text

# def exclude_import_statements(text: str) -> str:
#     # Match import statements at the beginning, until the next ``` ends
#     pattern1 = r"^import\b[\s\S]*?;\\n$"
#     matches1 = re.findall(pattern1, text, re.DOTALL)
#     pattern2 = r"```(java|python)\n(.*)"
#     matches2 = re.findall(pattern2, text, re.DOTALL)
#     if matches1:
#         return "".join(code for _, code in matches1)
#     elif matches2:
#         print("``` without ending found.")
#         return "".join(code for _, code in matches2)
#     else:
#         return text
def braces_balanced(code: str) -> bool:
    stack = []
    for ch in code:
        if ch == "{":
            stack.append(ch)
        elif ch == "}":
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "predictions.jsonl" in filenames:
        jsonl_path = os.path.join(dirpath, "predictions.jsonl")
        lines = []
        output_path = os.path.join(dirpath, "predictions_cleaned.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            print(f"{dirpath} is being cleanning. :-）")
            for line in f:
                data = json.loads(line)
                # Determine whether `generate_results` exists and is not empty.
                if "generate_results" in data and len(data["generate_results"]) > 0:
                    for idx, code in enumerate(data["generate_results"]):
                        if "openai" in dirpath:
                            # code = remove_assistant(code)
                            code = trim_before_return_marker(code)

                        code = extract_java_python_blocks(code)
                        code = process_code(code)
                        code = remove_imports(code)
                        code = delete_all_classes(code)
                        data["generate_results"][idx] = code
                    # code = data["generate_results"][0]
                    # if "openai" in dirpath:
                    #     code = remove_assistant(code)

                    # code = extract_java_python_blocks(code)
                    # code = process_code(code)
                    # data["generate_results"][0] = code
                lines.append(data)
        # Write back to the JSONL file
        with open(output_path, "w", encoding="utf-8") as f:
            for data in lines:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "predictions_cleaned.jsonl" in filenames:
        jsonl_path = os.path.join(dirpath, "predictions_cleaned.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Determine whether `generate_results` exists and is not empty.
                if "generate_results" in data and len(data["generate_results"]) > 0:
                    code = data["generate_results"][0]
                    if "```" in code:
                        print("The input text contains ``` which is not allowed.")
                    if braces_balanced(code):
                        print("shit not good")
            print("no false")
