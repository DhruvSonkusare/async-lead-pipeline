"""
app/config.py
=============
Configuration management for the Async Lead Processing Pipeline.

Handles:
  - Environment variables
  - Database settings
  - API configuration
  - ML model paths
  - Application settings
"""

import os
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv

# Loads .env into the process environment once, at import time - values
# already set in the real environment (e.g. by docker-compose, or a shell
# `$env:VAR=...`) take priority and are never overridden by this.
load_dotenv()


class Settings:
    """Application settings and configuration"""
    
    def __init__(self):
        """Initialize settings from environment variables and defaults"""
        
        # ====================================================================
        # APPLICATION
        # ====================================================================
        
        self.APP_NAME = "Async Lead Processing Pipeline"
        self.APP_VERSION = "1.0.0"
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        
        # ====================================================================
        # API CONFIGURATION
        # ====================================================================
        
        self.API_HOST = os.getenv("API_HOST", "127.0.0.1")
        self.API_PORT = int(os.getenv("API_PORT", "8001"))
        self.API_WORKERS = int(os.getenv("API_WORKERS", "4"))
        
        # ====================================================================
        # DATABASE (DYNAMODB)
        # ====================================================================
        
        self.USE_LOCAL_DYNAMODB = os.getenv("USE_LOCAL_DYNAMODB", "True").lower() == "true"
        self.DYNAMODB_ENDPOINT = os.getenv("DYNAMODB_ENDPOINT", "http://localhost:8000")
        self.DYNAMODB_REGION = os.getenv("DYNAMODB_REGION", "us-east-1")
        
        # AWS Credentials (for real DynamoDB)
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        # ====================================================================
        # MACHINE LEARNING
        # ====================================================================
        
        self.ML_MODEL_PATH = os.getenv(
            "ML_MODEL_PATH",
            "models/intent_classifier.pkl"
        )
        self.ML_MODEL_NAME = "intent_classifier"
        
        # ====================================================================
        # PROCESSING
        # ====================================================================
        
        # Spec caps this at 50,000 per upload - kept as the default; override
        # via env var for local testing only, not for submission.
        self.MAX_LEADS_PER_REQUEST = int(os.getenv("MAX_LEADS_PER_REQUEST", "50000"))
        self.BATCH_SIZE = 100  # how many leads between job-counter flushes
        self.CONCURRENT_WORKERS = 10  # bounded concurrency for ingestion

        # Retry configuration
        self.MAX_RETRIES = 3
        self.RETRY_BACKOFF_BASE = 1  # seconds
        self.RETRY_BACKOFF_MAX = 30  # seconds

        # Fault injection (for demonstrating retry/backoff/DLQ behavior).
        # 0.0 = disabled. Set to e.g. 0.10 to simulate a 10% random write failure rate.
        self.SIMULATED_FAILURE_RATE = float(os.getenv("SIMULATED_FAILURE_RATE", "0.0"))

        # ====================================================================
        # CLASSIFICATION WORKER (Part 5 - scheduled, Lambda-style handler)
        # ====================================================================

        self.WORKER_POLL_INTERVAL_SECONDS = int(os.getenv("WORKER_POLL_INTERVAL_SECONDS", "10"))
        self.WORKER_BATCH_SIZE = int(os.getenv("WORKER_BATCH_SIZE", "100"))
        self.WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "10"))
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # ====================================================================
        # PATHS
        # ====================================================================
        
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "data")
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, "models")
        
        self.TRAINING_DATA_PATH = os.path.join(self.DATA_DIR, "leads_labelled.csv")
        self.UNLABELED_DATA_PATH = os.path.join(self.DATA_DIR, "leads_50k.csv")
        
        # Initialize
        self._validate_paths()
        self._setup_directories()
    
    def _validate_paths(self):
        """Validate that required paths exist"""
        if not os.path.exists(self.DATA_DIR):
            # Create data directory if it doesn't exist
            os.makedirs(self.DATA_DIR, exist_ok=True)
        
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR, exist_ok=True)
    
    def _setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
    
    def get_model_path(self) -> str:
        """Get the full path to the ML model"""
        if os.path.isabs(self.ML_MODEL_PATH):
            return self.ML_MODEL_PATH
        return os.path.join(self.PROJECT_ROOT, self.ML_MODEL_PATH)
    
    def get_training_data_path(self) -> str:
        """Get the full path to training data"""
        if os.path.isabs(self.TRAINING_DATA_PATH):
            return self.TRAINING_DATA_PATH
        return os.path.join(self.PROJECT_ROOT, self.TRAINING_DATA_PATH)
    
    def get_unlabeled_data_path(self) -> str:
        """Get the full path to unlabeled data"""
        if os.path.isabs(self.UNLABELED_DATA_PATH):
            return self.UNLABELED_DATA_PATH
        return os.path.join(self.PROJECT_ROOT, self.UNLABELED_DATA_PATH)


# Create singleton instance
_settings = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Alternative: use lru_cache decorator
@lru_cache(maxsize=1)
def get_settings_cached() -> Settings:
    """Get application settings (cached version)"""
    return Settings()


if __name__ == "__main__":
    # Test configuration
    settings = get_settings()
    print(f"App: {settings.APP_NAME}")
    print(f"Version: {settings.APP_VERSION}")
    print(f"API: {settings.API_HOST}:{settings.API_PORT}")
    print(f"Model Path: {settings.get_model_path()}")
    print(f"Data Dir: {settings.DATA_DIR}")
    print(f"Project Root: {settings.PROJECT_ROOT}")