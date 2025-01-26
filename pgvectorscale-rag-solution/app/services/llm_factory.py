from typing import List, Dict, Type, Any
from pydantic import BaseModel
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class LLMFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize Hugging Face model"""
        try:
            model_name = "tiiuae/falcon-7b-instruct"  # Or another suitable model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create a text generation pipeline
            return pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer,
                max_length=512
            )
        except Exception as e:
            logging.error(f"Model initialization error: {e}")
            raise

    def create_completion(self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs) -> Any:
        """Generate response from Hugging Face model"""
        # Combine messages into a single prompt
        prompt = self._format_messages(messages)
        
        try:
            # Generate response using the pipeline
            responses = self.model(prompt, max_new_tokens=200)
            response_text = responses[0]['generated_text']

            # Parse the response into the expected format
            return self._parse_response(response_text, response_model)
        except Exception as e:
            logging.error(f"Response generation error: {e}")
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format the messages into a prompt for the model"""
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    def _parse_response(self, response_text: str, response_model: Type[BaseModel]) -> BaseModel:
        """Parse the response from the model into the expected response model"""
        try:
            # Create an instance of the response model
            return response_model(
                thought_process=["Analyzed document"],
                answer=response_text,
                enough_context=True
            )
        except Exception as e:
            logging.error(f"Response parsing error: {e}")
            raise