from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.openai import OpenAIProvider
#1. Initialize model via names. 


# Original version. 
#"""
ollama_model = OpenAIModel(
    model_name='llama3.2', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)
#"""
"""
ollama_model = OpenAIModel(
 #   model_name='mistral:latest', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)
"""
