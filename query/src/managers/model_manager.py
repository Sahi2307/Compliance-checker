import os
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from src.config.settings import settings

class ModelManager:
    @classmethod
    def setup_model(cls):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.HUGGINGFACE_API_TOKEN
        
        llm = HuggingFaceHub(
            repo_id=settings.FALCON_MODEL, 
            model_kwargs={"temperature": 0.1, "max_length": 512}
        )
        
        chain = load_qa_chain(llm, chain_type="stuff")
        
        return llm, chain