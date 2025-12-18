"""Unified LLM provider interface for all supported providers.

This module provides a consistent API for interacting with different LLM providers
(DeepSeek, Ollama, GLM) through a single class interface.
"""

import json
import requests
from typing import Dict, Optional


class LLMProvider:
    """Unified interface for LLM providers (DeepSeek, Ollama, GLM)."""

    def __init__(self, config_path: str = "config.json", provider: Optional[str] = None):
        """Initialize LLM provider from config.

        Args:
            config_path: Path to configuration file.
            provider: Optional provider name to override config. If None, uses config value.
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.provider = provider or self.config.get("provider", "deepseek")
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize provider-specific configuration."""
        if self.provider == "deepseek":
            deepseek_config = self.config.get("deepseek", {})
            self.api_key = deepseek_config.get("api_key", "").strip() if deepseek_config.get("api_key") else ""
            self.api_url = deepseek_config.get("api_url", "").strip() if deepseek_config.get("api_url") else ""
            self.editor_model = deepseek_config.get("editor_model", "deepseek-chat")
            self.critic_model = deepseek_config.get("critic_model", self.editor_model)

            if not self.api_key:
                raise ValueError(
                    "DeepSeek API key not found in config. Please set 'deepseek.api_key' in config.json. "
                    "Get your API key at https://platform.deepseek.com"
                )
            if not self.api_url:
                raise ValueError(
                    "DeepSeek API URL not found in config. Please set 'deepseek.api_url' in config.json"
                )

        elif self.provider == "ollama":
            ollama_config = self.config.get("ollama", {})
            self.api_url = ollama_config.get("url", "http://localhost:11434/api/chat")
            self.editor_model = ollama_config.get("editor_model", "mistral-nemo")
            self.critic_model = ollama_config.get("critic_model", "qwen3:8b")
            self.keep_alive = ollama_config.get("keep_alive", "10m")
            self.api_key = None  # Ollama doesn't use API keys

        elif self.provider == "glm":
            glm_config = self.config.get("glm", {})
            self.api_key = glm_config.get("api_key", "").strip() if glm_config.get("api_key") else ""
            self.api_url = glm_config.get("api_url", "").strip() if glm_config.get("api_url") else ""
            self.editor_model = glm_config.get("editor_model", "glm-4.6")
            self.critic_model = glm_config.get("critic_model", self.editor_model)

            if not self.api_key:
                raise ValueError(
                    "GLM API key not found in config. Please set 'glm.api_key' in config.json. "
                    "Get your API key at https://open.bigmodel.cn"
                )
            if not self.api_url:
                raise ValueError(
                    "GLM API URL not found in config. Please set 'glm.api_url' in config.json"
                )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_type: str = "editor",
        require_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Call LLM API with unified interface.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt with the request.
            model_type: Which model to use ("editor" or "critic"). Default: "editor".
            require_json: If True, request JSON format (for critic). If False, plain text.
            temperature: Optional temperature override. Default: provider-specific.
            max_tokens: Optional max_tokens override. Default: provider-specific.

        Returns:
            LLM response text.
        """
        model = self.critic_model if model_type == "critic" else self.editor_model

        if self.provider == "deepseek":
            return self._call_deepseek_api(
                system_prompt, user_prompt, model, require_json, temperature, max_tokens
            )
        elif self.provider == "ollama":
            return self._call_ollama_api(
                system_prompt, user_prompt, model, require_json, temperature, max_tokens
            )
        elif self.provider == "glm":
            return self._call_glm_api(
                system_prompt, user_prompt, model, require_json, temperature, max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_deepseek_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        require_json: bool,
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> str:
        """Call DeepSeek API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature if temperature is not None else 0.3,
        }

        # Only add JSON format requirement for critic evaluation
        if require_json:
            payload["response_format"] = {"type": "json_object"}
        else:
            # For text generation/editing, add max_tokens instead
            payload["max_tokens"] = max_tokens if max_tokens is not None else 200

        # Validate model name for DeepSeek API
        valid_deepseek_models = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner", "deepseek-chat-v3"]
        if model not in valid_deepseek_models:
            print(f"    ⚠ Warning: Model '{model}' may not be valid for DeepSeek API. Valid models: {valid_deepseek_models}")

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                error_detail = e.response.text
                raise RuntimeError(f"DeepSeek API 400 Bad Request: {error_detail}. Check model name '{model}' and request format.")
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DeepSeek API request failed: {e}")

    def _call_ollama_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        require_json: bool,
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> str:
        """Call Ollama API."""
        # Ollama uses /api/chat endpoint
        if "/api/generate" in self.api_url:
            api_url = self.api_url.replace("/api/generate", "/api/chat")
        else:
            api_url = self.api_url

        # For JSON format requests (critic), add format instruction to system prompt
        if require_json:
            enhanced_system_prompt = system_prompt + "\n\nIMPORTANT: You must respond with valid JSON only. No additional text or explanation."
        else:
            enhanced_system_prompt = system_prompt

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,  # Always use non-streaming for consistency
            "keep_alive": self.keep_alive
        }

        # Add options for temperature and max tokens
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if require_json:
            data["format"] = "json"  # Request JSON format for Ollama
        if options:
            data["options"] = options

        try:
            response = requests.post(api_url, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Try to get available models for better error message
                try:
                    models_url = api_url.replace("/api/chat", "/api/tags")
                    models_response = requests.get(models_url, timeout=5)
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        available_models = [m.get("name", "") for m in models_data.get("models", [])]
                        raise RuntimeError(
                            f"Ollama model '{model}' not found. Available models: {', '.join(available_models[:5])}"
                        )
                except:
                    pass
                raise RuntimeError(f"Ollama API 404: Model '{model}' not found or endpoint incorrect. Check model name and Ollama service.")
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")

    def _call_glm_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        require_json: bool,
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> str:
        """Call GLM (Zhipu AI) API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature if temperature is not None else 0.3,
        }

        # Only add JSON format requirement for critic evaluation
        if require_json:
            payload["response_format"] = {"type": "json_object"}
        else:
            # For text generation/editing, add max_tokens instead
            payload["max_tokens"] = max_tokens if max_tokens is not None else 200

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                error_detail = e.response.text
                raise RuntimeError(f"GLM API 400 Bad Request: {error_detail}. Check model name '{model}' and request format.")
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"GLM API request failed: {e}")

