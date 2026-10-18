"""Turn a plain-English question into pandas code using an LLM, then run it safely."""

import os
import re

import anthropic
import pandas as pd
from dotenv import load_dotenv

from modules.sandbox import run_code

load_dotenv()

# Change the model in your .env file with ANTHROPIC_MODEL=... if you want
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are a data analyst who writes pandas code.

Rules:
- The dataframe is already loaded as `df`. pandas as `pd`, numpy as `np`, \
plotly.express as `px` and plotly.graph_objects as `go` are already imported.
- Do NOT read or write files, and do NOT call .show() or print().
- Store the final answer in a variable called `result`.
- `result` can be a number, a string, a DataFrame, a Series, or a Plotly figure.
- If the user asks for a chart, plot or visualization, make a Plotly figure with px.
- Round numbers sensibly. Sort tables so the most relevant rows come first.
- If the question cannot be answered from this data, set `result` to a short \
string explaining why.
- Return ONLY Python code in a single ```python code block. No explanations."""


def _client() -> anthropic.Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Add it to a .env file in the project folder."
        )
    return anthropic.Anthropic()


def build_prompt(df: pd.DataFrame, question: str) -> str:
    dtypes = "\n".join(f"- {col}: {dtype}" for col, dtype in df.dtypes.items())
    sample = df.head(5).to_string()
    return (
        f"Dataframe columns and types:\n{dtypes}\n\n"
        f"First 5 rows:\n{sample}\n\n"
        f"Total rows: {len(df)}\n\n"
        f"Question: {question}"
    )


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def generate_code(df, question, previous_code=None, error=None) -> str:
    messages = [{"role": "user", "content": build_prompt(df, question)}]

    # Self-correction: show the model its own failed code and the error
    if previous_code and error:
        messages.append({"role": "assistant", "content": f"```python\n{previous_code}\n```"})
        messages.append(
            {
                "role": "user",
                "content": f"That code failed with this error:\n{error}\n"
                "Return the corrected code only.",
            }
        )

    response = _client().messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return extract_code(text)


def ask(df: pd.DataFrame, question: str, max_retries: int = 1) -> dict:
    """Question -> code -> safe execution (with one self-correction attempt)."""
    code = generate_code(df, question)
    outcome = run_code(code, df)
    attempts = 1

    while not outcome["ok"] and attempts <= max_retries:
        code = generate_code(df, question, previous_code=code, error=outcome["error"])
        outcome = run_code(code, df)
        attempts += 1

    return {"code": code, "attempts": attempts, **outcome}
