import os
import httpx

from .utils import compose_payload, compose_text_payload


class OpenAIConnector:

    def __init__(self, api_key: str):
        if api_key is None:
            raise ValueError("API_KEY is not set")
        self.api_key = api_key

    async def simple_prompt(self, prompt: str, image_path: str = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        if image_path is None:
            payload = compose_text_payload(prompt=prompt)
        else:
            payload = compose_payload(image_path=image_path, prompt=prompt)

        async with httpx.AsyncClient() as client:
            response = (await client.post("https://api.openai.com/v1/chat/completions",
                                         headers=headers, json=payload, timeout=20)).json()

        return response['choices'][0]['message']['content']
