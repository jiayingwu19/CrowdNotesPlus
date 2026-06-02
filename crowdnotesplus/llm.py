"""Asynchronous LLM client adapters used by the pipeline."""

from __future__ import annotations

from typing import Optional

from httpx import AsyncClient, Limits
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import NotesConfig


class BaseLLM:

    async def aclose(self):
        pass

    async def chat(self, system: str, user: str, max_tokens: int=8, temperature: float=0.0, top_p: float=1.0) -> str:
        raise NotImplementedError

class OpenAIClient(BaseLLM):

    def __init__(self, api_key: str, model: str, base_url: Optional[str], max_conn: int):
        from openai import AsyncOpenAI
        self.model = model
        self.http = AsyncClient(timeout=30.0, limits=Limits(max_connections=max_conn, max_keepalive_connections=20))
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=self.http)

    async def aclose(self):
        await self.http.aclose()

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def chat(self, system: str, user: str, max_tokens: int=8, temperature: float=0.0, top_p: float=1.0) -> str:
        resp = await self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return (resp.choices[0].message.content or '').strip()

class AnthropicClient(BaseLLM):

    def __init__(self, api_key: str, model: str, base_url: Optional[str], max_conn: int):
        from anthropic import AsyncAnthropic
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self.http = AsyncClient(timeout=30.0, limits=Limits(max_connections=max_conn, max_keepalive_connections=20))

    async def aclose(self):
        await self.http.aclose()

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def chat(self, system: str, user: str, max_tokens: int=8, temperature: float=0.0, top_p: float=1.0) -> str:
        resp = await self.client.messages.create(model=self.model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, system=system if system else None, messages=[{'role': 'user', 'content': user}], thinking={'type': 'disabled'})
        parts = []
        for block in resp.content:
            if getattr(block, 'type', None) == 'text':
                parts.append(block.text)
        return '\n'.join(parts).strip()

class GeminiClient(BaseLLM):

    def __init__(self, api_key: str, model: str, base_url: Optional[str], max_conn: int):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model_name = model
        self.genai = genai
        self.model = genai.GenerativeModel(model)

    async def aclose(self):
        pass

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def chat(self, system: str, user: str, max_tokens: int=8, temperature: float=0.0, top_p: float=1.0) -> str:
        prompt = (system.strip() + '\n\n' + user.strip()).strip()

        def _call_sync():
            resp = self.model.generate_content(prompt, generation_config={'temperature': temperature, 'top_p': top_p, 'max_output_tokens': max_tokens})
            try:
                return (resp.text or '').strip()
            except Exception:
                if hasattr(resp, 'candidates') and resp.candidates:
                    parts = []
                    for cand in resp.candidates:
                        try:
                            parts.append(cand.content.parts[0].text)
                        except Exception:
                            pass
                    return '\n'.join([p for p in parts if p]).strip()
                return ''
        return await asyncio.to_thread(_call_sync)

class VLLMClient(BaseLLM):

    def __init__(self, port: int):
        from openai import AsyncOpenAI
        openai_api_key = 'EMPTY'
        openai_api_base = f'http://localhost:{port}/v1'
        self.client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=36000.0)

    async def chat(self, system: str, user: str, max_tokens: int=8, temperature: float=0.0, top_p: float=1.0) -> str:
        models = await self.client.models.list()
        model = models.data[0].id
        prompt = (system.strip() + '\n\n' + user.strip()).strip()
        completion = await self.client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': prompt}], max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        content = completion.choices[0].message.content
        result = content or ''
        return result

class OpenRouterClient(BaseLLM):

    def __init__(self, api_key: str, model: str, base_url: Optional[str]='https://openrouter.ai/api/v1', max_conn: int=100, http_referer: Optional[str]=None, x_title: Optional[str]=None):
        from openai import AsyncOpenAI
        self.model = f'x-ai/{model}'
        self.http = AsyncClient(timeout=30.0, limits=Limits(max_connections=max_conn, max_keepalive_connections=20))
        base_url = 'https://openrouter.ai/api/v1'
        default_headers = {}
        if http_referer:
            default_headers['HTTP-Referer'] = http_referer
        if x_title:
            default_headers['X-Title'] = x_title
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=self.http, default_headers=default_headers or None)

    async def aclose(self):
        await self.http.aclose()

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def chat(self, system: str, user: str, max_tokens: int=8, temperature: float=0.0, top_p: float=1.0) -> str:
        resp = await self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return (resp.choices[0].message.content or '').strip()


def create_llm_client(cfg: NotesConfig) -> BaseLLM:
    """Create a provider-specific LLM client from a shared configuration object."""
    provider = (cfg.provider or "").lower()
    if provider == "openai":
        if not cfg.api_key:
            raise EnvironmentError("Missing OPENAI_API_KEY")
        return OpenAIClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore[arg-type]
    if provider == "anthropic":
        if not cfg.api_key:
            raise EnvironmentError("Missing ANTHROPIC_API_KEY")
        return AnthropicClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore[arg-type]
    if provider == "gemini":
        if not cfg.api_key:
            raise EnvironmentError("Missing GOOGLE_API_KEY")
        return GeminiClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore[arg-type]
    if provider == "vllm":
        if not cfg.port:
            raise EnvironmentError("Missing VLLM port")
        return VLLMClient(cfg.port)
    if provider == "openrouter":
        if not cfg.api_key:
            raise EnvironmentError("Missing OPENROUTER_API_KEY")
        return OpenRouterClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore[arg-type]
    raise ValueError("provider must be one of: openai | anthropic | gemini | vllm | openrouter")
