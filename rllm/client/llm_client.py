import google.generativeai as genai
from anthropic import Anthropic, NotGiven
from func_timeout import func_set_timeout
from openai import OpenAI
from together import Together
from strenum import StrEnum

class LLMClientType(StrEnum):
    OpenAI = "OpenAI"
    Claude = "Claude"
    Gemini = "Gemini"
    TogetherAI = "TogetherAI"


class LLMClient:
    def __init__(
        self,
        llm_client_type: LLMClientType,
        llm_name: str,
        llm_server_url: str | None = None,
        llm_server_api_key: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        if llm_client_type == LLMClientType.OpenAI:
            self.client = OpenAI(
                base_url=llm_server_url,
                api_key=llm_server_api_key,
                timeout=timeout,
            )
        elif llm_client_type == LLMClientType.TogetherAI:
            self.client = Together(
                api_key=llm_server_api_key,
                timeout=timeout,
            )
        elif llm_client_type == LLMClientType.Gemini:
            # NOTE: GenerativeModel does not support timeout
            genai.configure(api_key=llm_server_api_key)
            self.client = genai.GenerativeModel(
                model_name=llm_name,
                system_instruction=system_prompt,
            )
        elif llm_client_type == LLMClientType.Claude:
            self.client = Anthropic(api_key=llm_server_api_key, timeout=timeout)

        self.llm_name = llm_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries


    async def generate(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        if isinstance(self.client, genai.GenerativeModel):
            response = self.client.start_chat(history=[]).send_message(
                content=messages[-1]["content"],
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                ),
            )
            model_output = response.text
            finish_reason = response._result.candidates[0].finish_reason.name.lower()
        elif isinstance(self.client, Anthropic):
            system = (
                messages[0]["content"] if messages[0]["role"] == "system" else NotGiven
            )
            response = self.client.messages.create(
                system=system,
                model=self.llm_name,
                max_tokens=self.max_tokens,
                messages=[messages[-1]],
                temperature=self.temperature,
            )
            model_output = response.content
            finish_reason = (
                "stop" if response.stop_reason == "stop_sequence" else "unexpected"
            )
        elif isinstance(self.client, OpenAI) or isinstance(self.client, Together):
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                n=1,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            model_output = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

        return model_output, finish_reason

    @func_set_timeout(20)
    def test_connection(self):
        import asyncio
        asyncio.run(self.generate([{"role": "user", "content": "say hi"}]))
