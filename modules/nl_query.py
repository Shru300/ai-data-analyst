"""Turn a plain-English question into pandas code using a FREE LLM, then run it safely.

Supported providers (all free to start, pick one in your .env file):
  - groq    : free API key, no credit card, very fast  (recommended)
  - gemini  : free API key from Google AI Studio
  - ollama  : runs fully on your own computer, no internet or key needed

All three speak the OpenAI-compatible API, so one small client covers them.
"""

import os
import re

import openai
import pandas as pd
from dotenv import load_dotenv

from modules.sandbox import run_code

load_dotenv(override=True)

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "model": "qwen2.5-coder:7b",
    },
}

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


def _settings() -> dict:
    name = os.getenv("LLM_PROVIDER", "groq").lower()
    if name not in PROVIDERS:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{name}'. Use: {', '.join(PROVIDERS)}.")
    cfg = dict(PROVIDERS[name])
    cfg["name"] = name
    cfg["model"] = os.getenv("LLM_MODEL", cfg["model"])
    return cfg


def _client(cfg: dict) -> openai.OpenAI:
    key = "ollama"  # Ollama ignores the key but the client needs something
    if cfg["key_env"]:
        key = os.getenv(cfg["key_env"])
        if not key:
            raise RuntimeError(
                f"{cfg['key_env']} not found. Add it to the .env file in the project folder."
            )
    return openai.OpenAI(base_url=cfg["base_url"], api_key=key)


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
    cfg = _settings()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(df, question)},
    ]

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

    try:
        response = _client(cfg).chat.completions.create(
            model=cfg["model"],
            messages=messages,
            max_tokens=1000,
            temperature=0,
        )
    except openai.AuthenticationError:
        raise RuntimeError(f"Your {cfg['name']} API key was rejected. Check the key in .env.")
    except openai.RateLimitError:
        raise RuntimeError("Free-tier rate limit reached. Wait a minute and try again.")
    except openai.NotFoundError:
        raise RuntimeError(
            f"Model '{cfg['model']}' was not found on {cfg['name']}. "
            "Set LLM_MODEL in .env to a model your account can use."
        )
    except openai.APIConnectionError:
        raise RuntimeError(
            f"Could not connect to {cfg['name']}. "
            + ("Is Ollama running?" if cfg["name"] == "ollama" else "Check your internet.")
        )

    return extract_code(response.choices[0].message.content or "")


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
