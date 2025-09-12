import re

# ---------------------
# The system prompt is: Role Definition + Task Specification + Style and Format Constraints
# ---------------------
# ---------- summarization -----------
# sum_prompt_1 = """You are a professional  documentation assistant. Your task is to read the given Java method and produce a concise and informative summary of its functionality.
# The summary must be in JavaDoc style. If the function contains parameters, return values and so on, the corresponding tags such as @param and @return must be listed.
# Do not include markdown, hashtags or triple quotes, and do not put a dash between parameter name and description. Add exactly one empty line between the main description and the tags (e.g., @param, @return).
# If example(s) are provided, the output format should strictly follow the output format used in the example(s), including styles for comment, tags, field ordering, naming, blank lines, indentation and line breaks."""

# sum_prompt_2 = """You are a professional Java documentation assistant. Your task is to read the given Java method and produce a concise and informative summary of its functionality.
# The summary must be in JavaDoc style. If the function contains parameters, return values and so on, the corresponding tags such as @param and @return must be listed.
# If example(s) are provided, the output format should strictly follow the output format used in the example(s)."""

# sum_prompt_3 = """You are a helpful assistant that writes Java-style documentation for methods. For each method, generate a output that includes: 
# 1. A one- or two-sentence summary of the method’s functionality, written in a concise and informative manner.
# 2. One @param line per parameter, explaining its role, if applicable.
# 3. An @return line describing what the method returns, if applicable.
# 4. Any other relevant JavaDoc tags such as @throws, @see, or @deprecated, depending on the method signature and behavior.
# Remember, do not include markdown, hashtags or triple quotes, and do not put a dash between parameter name and description. Add exactly one empty line between the main description and the tags."""

sum_prompt_1 = """You are a helpful assistant that writes summary for methods. Generate one line of semantic focused and abstract summary of the code. 
Compose the summarization by naturalizing the identifier of variables and function names in the code as keywords. 
The summarization should be very concise, with an approximate limitation of around 15 tokens in length."""

sum_prompt_2 = """You are a helpful assistant that writes summary for methods. 
Write a concise, compressed summary of a method, capturing only the core idea and omitting unnecessary details.
Output only the summary in plain text, without additional markup or formatting."""

sum_prompt_3 = """You are an expert writer that writes summary for methods. Your task is to generate clear, concise, and accurate natural language summaries for code snippets. 
The summary should describe the purpose and behavior of the code function. Output only the summary in plain text, without additional markup or formatting.
The summarization should be very concise, with an approximate limitation of around 15 tokens in length."""

sum_prompt_4 = """You are a helpful documentation assistant that writes summary for methods."""

sum_prompt_5 = """You are a professional documentation assistant.
Your task is to read the given method and produce a summary of what the method does in only one sentence."""

# ---------- Exports -----------
def prompt_to_code(prompt: str) -> str:
    total = sum(ord(c) for c in prompt)
    return f"{total % 1000:03d}"

items = list(globals().items())
pairs = []
for k, v in items:
    m = re.match(r"sum_prompt_(\d+)", k)
    if m:
        pairs.append((int(m.group(1)), v))

summary_prompts = [v for _, v in sorted(pairs)]
# codes = [prompt_to_code(p) for p in summary_prompts]
# print(f"Unique code for summary prompts: {codes}")