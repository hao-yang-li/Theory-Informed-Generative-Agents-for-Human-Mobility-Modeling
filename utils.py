"""
TIMA Unified LLM Interface.
Handles OpenAI-compatible API calls, automatic retries,
and normalization of response objects across different providers.
"""

import time
from openai import OpenAI, RateLimitError
from constant import PLATFORM, BASE_URL_MAP

class LLM:
    def __init__(self, model_name: str, platform: PLATFORM = 'openai', api_key: str = None):
        self.model_name = model_name
        self.platform = platform
        self.api_key = api_key

        self.client = self._init_client(platform)

        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.history = []

        self.batch_list = []

    def _init_client(self, platform: PLATFORM):
        if not self.api_key:
            raise ValueError(f"API key for platform {platform} not found.")

        base_url = BASE_URL_MAP.get(platform)
        client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        return client

    def generate(
            self,
            prompt: list[dict] | str,
            model: str | None = None,
            temperature: float = 0,
            max_tokens: int = 8192,
            response_format: dict = None
    ):
        """
        Sends a chat completion request to the specified LLM.

        Args:
            prompt: String or message list to send.
            temperature: Sampling temperature for creativity control.
            max_tokens: Maximum length of the generated response.

        Returns:
            The text content of the model's response.
        """
        if model is None:
            model = self.model_name

        if isinstance(prompt, str):
            messages = [{'role': 'user', 'content': prompt}]
        else:
            messages = prompt

        for m in messages:
            m.update({"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})
            self.history.append(m)

        max_retries = 10

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                if hasattr(completion, 'usage') and hasattr(completion, 'choices'):
                    self.completion_tokens += completion.usage.completion_tokens
                    self.prompt_tokens += completion.usage.prompt_tokens
                    response = completion.choices[0].message.content
                else:
                    response = str(completion)

                self.history.append({'role': 'assistant', 'content': response,
                                     'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})

                return response

            except RateLimitError as e:
                print(f"Rate limit error encountered: {e}. This is attempt {attempt + 1} of {max_retries}.")
                if attempt < max_retries - 1:
                    if attempt < 2:
                        wait_time = 5 * (attempt + 1)  # Wait 5, 10 seconds
                        print(f"Waiting for {wait_time} seconds before retrying.")
                        time.sleep(wait_time)
                    else:
                        print("Waiting for 60 seconds before making a more spaced-out retry.")
                        time.sleep(60)
                else:
                    print("Maximum retries reached. Failing.")
                    raise

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise

    def __repr__(self):
        return f"LLM(model_name={self.model_name}, platform={self.platform})"