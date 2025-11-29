#!/usr/bin/env python3
"""
Convenience script to run the ConfirmIT AI Service
Usage: python run.py
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("🚀 Starting ConfirmIT AI Service...")
    print(f"📍 Environment: {settings.ENVIRONMENT}")
    print(f"🌐 Running on: http://0.0.0.0:{settings.PORT}")
    print(f"📖 API Docs: http://0.0.0.0:{settings.PORT}/docs")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level="info",
    )
