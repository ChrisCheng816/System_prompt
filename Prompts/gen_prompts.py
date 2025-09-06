import re

# ---------------------
# The system prompt is: Role Definition + Task Specification + Style and Format Constraints
# ---------------------

# ---------- generation -----------
gen_prompt_1 = """You are a highly skilled Java method generator. Your task is to generate correct and executable Java method from the natural language description.  
Follow these rules strictly:
1. Output only Java code, no explanations or comments.  
2. Use clean and consistent formatting.  
3. Ensure the code is syntactically valid and logically correct.  
4. Write the code in a minimal self-contained form, unless the description explicitly requires a larger class or method."""

items = list(globals().items())
pairs = []
for k, v in items:
    m = re.match(r"gen_prompt_(\d+)", k)
    if m:
        pairs.append((int(m.group(1)), v))

gen_prompts = [v for _, v in sorted(pairs)]
