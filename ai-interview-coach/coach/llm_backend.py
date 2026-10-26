"""
Pluggable LLM Backend interface and implementations.
Supports Ollama, OpenAI-compatible APIs, and a deterministic Mock backend for testing.
"""

from abc import ABC, abstractmethod
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.error


class BaseLLMBackend(ABC):
    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.extra_kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate completion for prompt.
        Returns: (raw_output_text, metadata_dict)
        """
        pass

    def compute_prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class MockLLMBackend(BaseLLMBackend):
    """
    Deterministic rule-anchored mock LLM backend for testing and offline execution.
    Generates realistic, valid rubric-anchored JSON responses grounded in the candidate's transcript.
    """
    def __init__(self, model_name: str = "mock-rubric-v1", temperature: float = 0.0, **kwargs):
        super().__init__(model_name=model_name, temperature=temperature, **kwargs)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        time.sleep(0.01)  # Simulate minimal latency
        
        # Check if this is an unconstrained B3 prompt ("score out of 10")
        if ("scale from 1 to 10" in prompt.lower() or "out of 10" in prompt.lower()
                or "score between 1 and 10" in prompt.lower()):
            # Estimate answer length / quality heuristic
            score = 7
            if "pessimistic" in prompt.lower() or "rollback" in prompt.lower() or "latency" in prompt.lower():
                score = 8
            elif len(prompt) < 150:
                score = 3
            raw_output = f"Based on the candidate's response, I rate this answer {score} out of 10. The candidate answered the question reasonably well."
            return raw_output, {
                "backend": "mock",
                "model": self.model_name,
                "version": "1.0",
                "latency_ms": 10.0,
            }

        # Otherwise, parse candidate answer from prompt to extract real quotes
        transcript = ""
        if "Candidate Answer:" in prompt:
            transcript = prompt.split("Candidate Answer:")[1].split("Instructions:")[0].strip()
        elif "Answer:" in prompt:
            transcript = prompt.split("Answer:")[1].split("\n\n")[0].strip()

        # Find usable quote excerpts from candidate transcript
        words = transcript.split()
        if len(words) >= 6:
            quote1 = " ".join(words[:min(8, len(words))])
            quote2 = " ".join(words[max(0, len(words)//2 - 3): min(len(words), len(words)//2 + 5)])
            quote3 = " ".join(words[-min(8, len(words)):])
        else:
            quote1 = transcript if transcript else "N/A"
            quote2 = transcript if transcript else "N/A"
            quote3 = transcript if transcript else "N/A"

        # Construct realistic S1 rubric evaluation
        eval_output = [
            {
                "dimension": "relevance_correctness",
                "score": 4 if len(words) > 30 else 3,
                "evidence_quote": quote1,
                "reason": "The response directly addresses the main prompt requirements and uses accurate technical terminology."
            },
            {
                "dimension": "depth_completeness",
                "score": 4 if len(words) > 50 else (3 if len(words) > 20 else 2),
                "evidence_quote": quote2,
                "reason": "Explains underlying mechanics and rationale with concrete trade-offs."
            },
            {
                "dimension": "structure_clarity",
                "score": 4 if any(k in transcript.lower() for k in ["situation", "task", "action", "result", "first", "second"]) else 3,
                "evidence_quote": quote3,
                "reason": "Presents ideas in a coherent logical sequence with clear topical milestones."
            },
            {
                "dimension": "language_use",
                "score": 4 if len(words) > 20 else 3,
                "evidence_quote": quote1,
                "reason": "Uses professional syntax and accurate domain terminology with minimal ambiguity."
            }
        ]

        if "delivery" in prompt.lower():
            eval_output.append({
                "dimension": "delivery",
                "score": 4,
                "evidence_quote": quote2,
                "reason": "Cadence is conversational and paced appropriately."
            })

        raw_output = json.dumps(eval_output, indent=2)
        return raw_output, {
            "backend": "mock",
            "model": self.model_name,
            "version": "1.0",
            "latency_ms": 15.0,
        }


class OllamaLLMBackend(BaseLLMBackend):
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434", temperature: float = 0.0, **kwargs):
        super().__init__(model_name=model_name, temperature=temperature, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._version_cache: Optional[str] = None

    def _resolve_version(self) -> str:
        """Model digest from Ollama /api/tags, so logs identify the exact model weights used."""
        if self._version_cache is None:
            self._version_cache = "unknown"
            try:
                with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10) as r:
                    tags = json.loads(r.read().decode("utf-8"))
                for m in tags.get("models", []):
                    names = {m.get("name"), m.get("model")}
                    if self.model_name in names or f"{self.model_name}:latest" in names:
                        self._version_cache = str(m.get("digest", "unknown"))[:12]
                        break
            except Exception:
                pass
        return self._version_cache

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        start_t = time.time()
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": int(self.extra_kwargs.get("seed", 42)),
            }
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                latency = round((time.time() - start_t) * 1000, 2)
                return res_body.get("response", ""), {
                    "backend": "ollama",
                    "model": self.model_name,
                    "version": self._resolve_version(),
                    "latency_ms": latency,
                }
        except Exception as e:
            raise RuntimeError(f"Ollama backend failure ({self.base_url}): {e}")


class OpenAICompatibleBackend(BaseLLMBackend):
    def __init__(self, model_name: str = "gpt-3.5-turbo", base_url: str = "https://api.openai.com/v1", api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0, **kwargs):
        super().__init__(model_name=model_name, temperature=temperature, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv(api_key_env, "")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        start_t = time.time()
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        req_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=req_data, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                latency = round((time.time() - start_t) * 1000, 2)
                content = res_body["choices"][0]["message"]["content"]
                return content, {
                    "backend": "openai_compatible",
                    "model": self.model_name,
                    "version": res_body.get("model", self.model_name),
                    "latency_ms": latency,
                }
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible backend failure: {e}")


def get_llm_backend(config: Dict[str, Any]) -> BaseLLMBackend:
    llm_cfg = config.get("llm", {})
    backend_type = llm_cfg.get("backend", "ollama").lower()
    model_name = llm_cfg.get("model", "llama3")
    temperature = float(llm_cfg.get("temperature", 0.0))

    if backend_type == "mock":
        return MockLLMBackend(model_name=model_name, temperature=temperature)
    elif backend_type == "ollama":
        base_url = llm_cfg.get("ollama_base_url", "http://localhost:11434")
        return OllamaLLMBackend(model_name=model_name, base_url=base_url, temperature=temperature)
    elif backend_type in ["openai", "openai_compatible", "api"]:
        base_url = llm_cfg.get("api_base_url", "https://api.openai.com/v1")
        api_key_env = llm_cfg.get("api_key_env", "LLM_API_KEY")
        return OpenAICompatibleBackend(model_name=model_name, base_url=base_url, api_key_env=api_key_env, temperature=temperature)
    else:
        # Never fall back silently to the mock: that would produce fake scores.
        raise ValueError(
            f"Unknown llm.backend '{backend_type}'. Use one of: mock, ollama, openai_compatible."
        )
