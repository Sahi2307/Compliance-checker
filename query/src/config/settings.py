from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HUGGINGFACE_API_TOKEN: str = "YOURHUGGINGFACEHUB_API_TOKEN"
    FALCON_MODEL: str = "tiiuae/falcon-7b-instruct"
    PINECONE_API_KEY: str = "your api key"
    PINECONE_ENVIRONMENT: str = "your env name"
    PINECONE_INDEX_NAME: str = "your index name"

    class Config:
        env_file = ".env"

settings = Settings()