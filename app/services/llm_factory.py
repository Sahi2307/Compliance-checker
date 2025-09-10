from typing import Any, Dict, List, Type
import google.generativeai as genai
from pydantic import BaseModel
import json

from config.settings import get_settings


class LLMFactory:
	def __init__(self, provider: str):
		self.provider = provider
		self.settings = getattr(get_settings(), provider)
		self.client = self._initialize_client()

	def _initialize_client(self) -> Any:
		if self.provider == "google_gemini":
			genai.configure(api_key=self.settings.api_key)
			return genai.GenerativeModel(self.settings.default_model)
		raise ValueError(f"Unsupported LLM provider: {self.provider}")

	def create_completion(
		self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
	) -> Any:
		if self.provider == "google_gemini":
			# Convert messages to Gemini format
			prompt = self._convert_messages_to_prompt(messages)
			
			# Generate content with Gemini
			response = self.client.generate_content(
				prompt,
				generation_config=genai.types.GenerationConfig(
					temperature=kwargs.get("temperature", self.settings.temperature),
					max_output_tokens=kwargs.get("max_tokens", self.settings.max_tokens),
				)
			)
			
			# Parse the response and create the response model instance
			try:
				response_text = response.text
				# Try to parse as JSON first
				if response_text.strip().startswith('{'):
					response_data = json.loads(response_text)
				else:
					# If not JSON, create a simple response structure
					response_data = {
						"thought_process": ["Generated response using Google Gemini"],
						"answer": response_text,
						"enough_context": True
					}
				return response_model(**response_data)
			except Exception as e:
				# Fallback: create response with the raw text
				return response_model(
					thought_process=["Generated response using Google Gemini"],
					answer=response.text if hasattr(response, 'text') else str(response),
					enough_context=True
				)
		raise ValueError(f"Unsupported LLM provider: {self.provider}")
	
	def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
		"""Convert OpenAI-style messages to a single prompt for Gemini."""
		prompt_parts = []
		for message in messages:
			role = message.get("role", "")
			content = message.get("content", "")
			
			if role == "system":
				prompt_parts.append(f"System: {content}")
			elif role == "user":
				prompt_parts.append(f"User: {content}")
			elif role == "assistant":
				prompt_parts.append(f"Assistant: {content}")
		
		return "\n\n".join(prompt_parts)
