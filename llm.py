from dataclasses import dataclass
from openai import OpenAI


@dataclass
class LLMConfig:
    model: str
    provider: str = "openai"


class LLMClient:
    def __init__(self, cfg: LLMConfig, api_key: str | None = None):
        self.cfg = cfg

        if cfg.provider == "qwen":
            # DeepSeek / Alibaba Qwen API compatible OpenAI protocol
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        elif cfg.provider == "openai":
            self.client = OpenAI(api_key=api_key)

        else:
            raise ValueError("Provider not supported")

    # ---------- GENERAL CHAT ----------
    def chat(self, prompt: str):
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content

    # ---------- USED BY FMEA ----------
    def generate_text(self, prompt: str, system: str = "You are a helpful assistant"):
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content