import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI

PROVIDER_PRESETS = {
    "nvidia": {
     "base_url": "https://integrate.api.nvidia.com/v1",
     "default_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
     "strong_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
     "env_key": "NVIDIA_API_KEY",
     "supports_json_mode": False,
 },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "nvidia/nemotron-3.5-lightning:free",
        "strong_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "env_key": "OPENROUTER_API_KEY",
        "supports_json_mode": False,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.5-flash-lite",
        "strong_model": "gemini-3.5-flash-lite",
        "env_key": "GEMINI_API_KEY",
        "supports_json_mode": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "strong_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "supports_json_mode": True,
    },
}


def extract_json(raw_text: str) -> Any:
    """Strips <think> tags/code fences and extracts valid JSON."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from model output:\n{cleaned[:500]}")


class SwappableLLMClient:
    """
    OpenAI-SDK client with 45s fail-fast timeout, live latency logs, and ordered failover:
    Primary (NVIDIA Nemotron) -> Backup 1 (Gemini) -> Backup 2 (OpenAI).
    """

    def __init__(
        self,
        provider: str = "nvidia",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        strong_model: Optional[str] = None,
        backups: Optional[List[Dict[str, str]]] = None,
        min_delay_sec: float = 1.0,
        timeout_sec: float = 45.0,
    ):
        self.provider = provider.lower().strip()
        preset = PROVIDER_PRESETS[self.provider]
        self.model = model or preset["default_model"]
        self.strong_model = strong_model or preset["strong_model"]
        self.timeout_sec = timeout_sec

        primary_key = api_key or os.getenv(preset["env_key"])
        if not primary_key:
            raise ValueError(f"Missing primary API key. Set {preset['env_key']} in .env or pass api_key.")

        self.chain: List[Dict[str, Any]] = [
            self._make_entry(self.provider, primary_key, self.model, self.strong_model, timeout_sec)
        ]

        if backups is None:
            for b_name in ["gemini", "openai"]:
                if b_name != self.provider:
                    b_key = os.getenv(PROVIDER_PRESETS[b_name]["env_key"])
                    if b_key:
                        self.chain.append(self._make_entry(b_name, b_key, timeout_sec=timeout_sec))
        else:
            for b in backups:
                b_name = b["provider"].lower().strip()
                b_key = b.get("api_key") or os.getenv(PROVIDER_PRESETS[b_name]["env_key"])
                if b_key:
                    self.chain.append(
                        self._make_entry(b_name, b_key, b.get("model"), b.get("strong_model"), timeout_sec)
                    )

        self.min_delay_sec = min_delay_sec
        self._last_call_ts = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.failover_events: List[str] = []

    @staticmethod
    def _make_entry(
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        strong_model: Optional[str] = None,
        timeout_sec: float = 45.0,
    ) -> Dict[str, Any]:
        preset = PROVIDER_PRESETS[provider]
        headers = (
            {"HTTP-Referer": "https://microdrama-promo-engine.local", "X-Title": "Microdrama Promo Engine"}
            if provider == "openrouter"
            else None
        )
        return {
            "provider": provider,
            "model": model or preset["default_model"],
            "strong_model": strong_model or preset["strong_model"],
            "supports_json_mode": preset["supports_json_mode"],
            "client": OpenAI(
                api_key=api_key,
                base_url=preset["base_url"],
                default_headers=headers,
                timeout=timeout_sec,
                max_retries=0,
            ),
        }

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_override: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        use_strong = bool(model_override and model_override == self.strong_model)
        errors: List[str] = []

        for idx, entry in enumerate(self.chain):
            p_name = entry["provider"]
            target_model = (
                model_override
                if (idx == 0 and model_override)
                else (entry["strong_model"] if use_strong else entry["model"])
            )

            for attempt in range(max_retries):
                elapsed = time.time() - self._last_call_ts
                if elapsed < self.min_delay_sec:
                    time.sleep(self.min_delay_sec - elapsed)

                t0 = time.time()
                print(f"    -> Calling {p_name}/{target_model} (try {attempt+1}/{max_retries})...", end="", flush=True)
                try:
                    kwargs: Dict[str, Any] = {
                        "model": target_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": 4096,
                    }
                    if entry["supports_json_mode"]:
                        kwargs["response_format"] = {"type": "json_object"}

                    # Disable silent reasoning chains on NVIDIA Nemotron so batches finish in seconds
                    if p_name == "nvidia" and "nemotron" in target_model:
                        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

                    self._last_call_ts = time.time()
                    resp = entry["client"].chat.completions.create(**kwargs)
                    self.total_calls += 1

                    dt = time.time() - t0
                    in_tok = resp.usage.prompt_tokens if getattr(resp, "usage", None) else 0
                    out_tok = resp.usage.completion_tokens if getattr(resp, "usage", None) else 0
                    self.total_prompt_tokens += in_tok or 0
                    self.total_completion_tokens += out_tok or 0
                    print(f" done in {dt:.1f}s ({in_tok} in / {out_tok} out)", flush=True)

                    content = resp.choices[0].message.content or ""
                    if not content.strip():
                        raise ValueError(f"Empty content returned by {target_model}.")
                    return extract_json(content)

                except Exception as e:
                    dt = time.time() - t0
                    print(f" FAILED in {dt:.1f}s: {str(e)[:140]}", flush=True)
                    errors.append(f"{p_name}/{target_model} (try {attempt+1}): {e}")
                    time.sleep(min(8, 2 ** (attempt + 1)))

            if idx + 1 < len(self.chain):
                next_p = self.chain[idx + 1]["provider"]
                note = f"Failover: {p_name} -> {next_p} after error: {errors[-1][:140]}"
                self.failover_events.append(note)
                print(f"    [!] {note}", flush=True)

        raise RuntimeError("All providers in chain failed:\n" + "\n".join(errors))