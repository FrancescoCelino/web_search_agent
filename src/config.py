import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

    DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "./content/docs")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///Chinook.db")
    # vector store
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_collection")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Search settings
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "2"))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "2"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is required")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        return True