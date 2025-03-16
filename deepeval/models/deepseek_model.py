import json
import logging
from typing import Union, Optional, Dict
import re
from openai import OpenAI

from pydantic import BaseModel, ValidationError

from deepeval.models.base_model import DeepEvalBaseLLM, DeepEvalBaseMLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

valid_deepseek_models = [
    "deepseek-chat",
    "deepseek-reasoner"
]

default_deepseek_model = "deepseek-chat"
default_system_message = "You are a helpful AI assistant. Always generate your response as a valid json. No explanation or extra information is needed just the json."

class DeepSeekModel(DeepEvalBaseLLM):
    """A class that integrates with DeepSeek DeepSeek for model inference and text generation.

    This class communicates with the DeepSeek DeepSeek service to invoke models for generating text and extracting
    JSON responses from the model outputs.

    Attributes:
        model_id (str): The ID of the DeepSeek model to use for inference.
        system_prompt (str): A predefined system prompt for DeepSeek models that directs their behavior.
        access_key_id (str, optional): DeepSeek access key ID for authentication. Can be provided or fetched from the key handler.
        secret_access_key (str, optional): DeepSeek secret access key for authentication. Can be provided or fetched from the key handler.
        session_token (str, optional): DeepSeek session token for temporary authentication. Can be provided or fetched from the key handler.
        region (str, optional): DeepSeek region where the DeepSeek client will be created. If not provided, defaults to fetched value.

    Example:
        ```python
        from deepeval.models import DeepSeekModel

        # Initialize the model with your own model ID and system prompt
        model = DeepSeekModel(
            model_id="your-deepseek-model-id",
            system_prompt="You are a helpful AI assistant. Always generate your response as a valid json. No explanation is needed just the json."
        )
        
        # Generate text with a prompt
        response = model.generate("What is the capital of France?", schema)
        ```
    """
    def __init__(
        self,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        *args,
        **kwargs
    ):
        """Initializes the DeepSeekModel with model_id, system_prompt, and optional DeepSeek credentials."""
        self.model_id = model_id or default_deepseek_model
        
        if self.model_id not in valid_deepseek_models:
            raise ValueError(
                f"Invalid model: {self.model_id}. Available DeepSeek models: {', '.join(model for model in valid_deepseek_models)}"
            )

        self.system_prompt = system_prompt or default_system_message
        self.api_key = api_key or KEY_FILE_HANDLER.fetch_data(KeyValues.DEEPSEEK_API_KEY)

        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

        super().__init__(model_id, *args, **kwargs)
        self.model = self.load_model(*args, **kwargs)

        print("DEBUG: deepseek called")

    def load_model(self):
        """Loads the DeepSeek client."""
        
        return self.client

    def extract_json(self, text: str) -> dict:
        """Attempts to parse the given text into a valid JSON dictionary."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error("Error decoding JSON")
            return {}
        
    def trim_and_load_json(
        self,
        input_string: str,
    ) -> Dict:
        start = input_string.find("{")
        end = input_string.rfind("}") + 1
        if end == 0 and start != -1:
            input_string = input_string + "}"
            end = len(input_string)
        jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
        jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)
        try:
            return json.loads(jsonStr)
        except json.JSONDecodeError:
            error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
            raise ValueError(error_str)
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[BaseModel, dict]:
        """Generates text using the DeepSeek model and returns the response as a Pydantic model."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": prompt})
        print("DEBUG: generate",messages) 
        print("DEBUG: schema",schema)

        try:
            client = self.load_model()
            if schema:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    response_format={
                        'type': 'json_object'
                    }
                    # max_tokens=1024,
                    # temperature=0.7,
                    # stream=False
                )
                content = self.trim_and_load_json(response.choices[0].message.content)
                content = schema.model_validate(content)
                print("DEBUG: content", content["statements"])
                for key, value in content.items():
                        print(f"DEBUG: key={key}, value={value}")
                # if isinstance(content, dict):
                    
                print("DEBUG: content type", type(content))
                return content
            else:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7,
                    stream=False
                )
                return response.choices[0].message.content
            #TODO add reasoning resp
            #reasoning_content = response.choices[0].message.reasoning_content
        
        except Exception as e:
            logger.error(f"An error occurred while generating the result: {e}")
            return {} if schema is None else None
    
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[BaseModel, dict]:
        return self.generate(prompt, schema)

    def get_model_name(self):
        """Returns the model ID being used."""
        return self.model_id
