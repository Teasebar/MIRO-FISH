"""
LLM客户端封装
使用Anthropic Claude API，提供OpenAI兼容接口
"""

import json
import re
from typing import Optional, Dict, Any, List
import anthropic

from ..config import Config


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, content: str, finish_reason: str):
        self.choices = [_Choice(content, finish_reason)]


class _Completions:
    def __init__(self, client: anthropic.Anthropic):
        self._client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> _Response:
        system_content = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})

        if response_format and response_format.get("type") == "json_object":
            json_instruction = (
                "You must respond with valid JSON only. "
                "No markdown formatting, no code blocks, no explanation — pure JSON."
            )
            system_content = (
                (system_content + "\n\n" + json_instruction).strip()
                if system_content else json_instruction
            )

        if not user_messages:
            user_messages = [{"role": "user", "content": "Please proceed."}]

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": user_messages,
            "temperature": temperature,
        }
        if system_content:
            create_kwargs["system"] = system_content

        response = self._client.messages.create(**create_kwargs)
        content = response.content[0].text
        finish_reason = "length" if response.stop_reason == "max_tokens" else "stop"

        return _Response(content, finish_reason)


class _Chat:
    def __init__(self, client: anthropic.Anthropic):
        self.completions = _Completions(client)


class OpenAI:
    """
    Anthropic Claude的OpenAI格式兼容包装器。
    允许使用OpenAI SDK风格的代码调用Anthropic API。
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self._anthropic = anthropic.Anthropic(api_key=api_key)
        self.chat = _Chat(self._anthropic)


class LLMClient:
    """LLM客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self._openai_compat = OpenAI(api_key=self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self._openai_compat.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
