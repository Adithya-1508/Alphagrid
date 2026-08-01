from __future__ import annotations

from typing import Any

import ollama

from ..config import load_config


def get_llm_config() -> dict[str, Any]:
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    return {
        "model": llm_cfg.get("model", "gemma4:e2b"),
        "temperature": llm_cfg.get("temperature", 0),
    }


def check_ollama_status() -> bool:
    """
    Startup health check to verify Ollama daemon is active and responsive.
    """
    try:
        models_resp = ollama.list()
        if isinstance(models_resp, dict):
            models_list = models_resp.get("models", [])
        else:
            models_list = getattr(models_resp, "models", [])

        model_names = []
        for m in models_list:
            if isinstance(m, dict):
                name = m.get("name", "")
            else:
                name = getattr(m, "model", getattr(m, "name", ""))
            model_names.append(name)

        cfg = get_llm_config()
        target_model = cfg["model"]
        return any(target_model in name for name in model_names) or len(models_list) > 0
    except Exception:  # noqa: BLE001
        return False


def generate_completion(prompt: str, system_prompt: str | None = None) -> str:
    """
    Sends a prompt to Ollama with temperature=0 for deterministic outputs.
    Catches Ollama OOM, connection, or status errors and returns empty string
    so downstream agents can execute deterministic fallbacks safely.
    """
    cfg = get_llm_config()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = ollama.chat(
            model=cfg["model"],
            messages=messages,
            options={"temperature": cfg["temperature"]},
        )
        return str(response["message"]["content"])
    except Exception as err:  # noqa: BLE001
        print(f"[Ollama Warning] LLM generation unavailable: {err}")
        return ""
