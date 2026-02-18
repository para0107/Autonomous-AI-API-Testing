"""
Global configuration settings for the API Testing Agent
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAINING_DATA_DIR = DATA_DIR / "training"
VECTOR_STORE_DIR = DATA_DIR / "vectors"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"

# Create directories if they don't exist
for dir_path in [DATA_DIR, TRAINING_DATA_DIR, VECTOR_STORE_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# System settings
class Settings:
    # Application settings
    APP_NAME = "API Testing Agent"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Processing settings
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

    # Language support
    SUPPORTED_LANGUAGES = ["csharp", "python", "java", "cpp"]

    # Test generation settings
    MAX_TESTS_PER_ENDPOINT = int(os.getenv("MAX_TESTS_PER_ENDPOINT", "50"))
    INCLUDE_EDGE_CASES = True
    INCLUDE_NEGATIVE_TESTS = True
    INCLUDE_SECURITY_TESTS = True

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = BASE_DIR / "logs" / "agent.log"

    # API Testing
    DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }


# Paths configuration
class PathConfig:
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    TRAINING_DATA_DIR = TRAINING_DATA_DIR
    VECTOR_STORE_DIR = VECTOR_STORE_DIR
    MODELS_DIR = MODELS_DIR
    REPORTS_DIR = REPORTS_DIR


settings = Settings()
paths = PathConfig()