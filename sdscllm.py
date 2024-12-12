from typing import Any, Dict, Iterator, List, Optional
import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


class CustomSDSCLLM(LLM):
    """Custom LLM to interface with the SDSC API."""

    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    base_url: str = "https://sdsc-llm-openwebui.nrp-nautilus.io/api/chat/completions"
    """Base URL for the SDSC LLM API."""

    def __init__(self, api_key_path: str = "api_key.txt", **kwargs: Any):
        """Initialize the CustomSDSCLLM with an API key from a file."""
        super().__init__(**kwargs)
        try:
            with open(api_key_path, "r") as file:
                self.api_key = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at: {api_key_path}")
        except Exception as e:
            raise Exception(f"Error reading API key file: {e}")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the SDSC LLM API with a single prompt."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,  # Non-streaming for _call
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            output = response_data["choices"][0]["message"]["content"].strip()

            # Optionally send callback signal for final token
            if run_manager:
                run_manager.on_llm_end({"output": output})
            return output
        else:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream output from the SDSC LLM API if supported."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": True,  # Enable streaming
        }

        response = requests.post(self.base_url, headers=headers, json=payload, stream=True)

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = line.decode("utf-8")
                        chunk = GenerationChunk(text=chunk_data)
                        if run_manager:
                            run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                        yield chunk
                    except Exception as e:
                        raise Exception(f"Error processing stream response: {e}")
        else:
            raise Exception(f"Streaming request failed with status {response.status_code}: {response.text}")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model,
            "base_url": self.base_url,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of the language model."""
        return "custom_sdscllm"
