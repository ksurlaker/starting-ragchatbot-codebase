import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    MAX_TOOL_ROUNDS: int = 2     # Maximum sequential tool execution rounds per query

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    def __post_init__(self):
        """Validate configuration on initialization"""
        if self.MAX_RESULTS <= 0:
            raise ValueError(f"MAX_RESULTS must be positive, got {self.MAX_RESULTS}")
        if self.CHUNK_SIZE <= 0:
            raise ValueError(f"CHUNK_SIZE must be positive, got {self.CHUNK_SIZE}")
        if self.MAX_HISTORY < 0:
            raise ValueError(f"MAX_HISTORY cannot be negative, got {self.MAX_HISTORY}")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError(f"CHUNK_OVERLAP cannot be negative, got {self.CHUNK_OVERLAP}")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({self.CHUNK_SIZE})")
        if self.MAX_TOOL_ROUNDS <= 0:
            raise ValueError(f"MAX_TOOL_ROUNDS must be positive, got {self.MAX_TOOL_ROUNDS}")
        if self.MAX_TOOL_ROUNDS > 5:
            raise ValueError(f"MAX_TOOL_ROUNDS exceeds maximum of 5, got {self.MAX_TOOL_ROUNDS}")

config = Config()


