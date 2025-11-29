from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Server
    PORT: int = 8080  # Cloud Run default
    ENVIRONMENT: str = "production"

    # Google Gemini
    GEMINI_API_KEY: str

    # Firebase
    FIREBASE_PROJECT_ID: str
    FIREBASE_PRIVATE_KEY: Optional[str] = None
    FIREBASE_CLIENT_EMAIL: Optional[str] = None

    # Cloudinary
    CLOUDINARY_URL: str

    # Model Configuration
    DEFAULT_MODEL: str = "gemini-1.5-flash"
    MAX_CONCURRENT_AGENTS: int = 3  # Reduced for Cloud Run memory limits
    AGENT_TIMEOUT_SECONDS: int = 60

    # Forensics
    ELA_QUALITY: int = 95
    FORENSIC_THRESHOLD: float = 0.7
    
    # Cloud Run Optimizations
    WORKER_TIMEOUT: int = 300  # 5 minutes
    KEEP_ALIVE: int = 120
    MAX_REQUESTS: int = 1000
    MAX_REQUESTS_JITTER: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
