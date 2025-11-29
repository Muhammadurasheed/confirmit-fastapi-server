from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn
import os

from app.config import settings
from app.core import firebase  

# Import routers
from app.routers import receipts, accounts

# ============================================================
# 🔧 Structured Logging Setup
# ============================================================
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

# ============================================================
# 🚀 FastAPI Initialization
# ============================================================
app = FastAPI(
    title="ConfirmIT AI Service",
    description="Multi-Agent AI system for receipt verification and fraud detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============================================================
# 🌍 CORS Setup
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "*",  # Allow all origins for testing; tighten in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🧩 Routers
# ============================================================
app.include_router(receipts.router, prefix="/api", tags=["receipts"])
app.include_router(accounts.router, prefix="/api", tags=["accounts"])

# ============================================================
# 🩺 Health & Root Routes
# ============================================================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ConfirmIT AI Service",
        "status": "operational",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "agents": {
            "vision": "ready",
            "forensic": "ready",
            "metadata": "ready",
            "reputation": "ready",
            "reasoning": "ready",
        },
    }

# ============================================================
# ⚠️ Global Exception Handler
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc),
        },
    )

# ============================================================
# ▶️ Entry Point (Local or Render)
# ============================================================
if __name__ == "__main__":
    # On Render, PORT is provided by environment (default 10000)
    port = int(os.environ.get("PORT", settings.PORT or 10000))
    env = settings.ENVIRONMENT or "development"

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=env == "development",
        log_level="info",
        workers=1,  # ⚙️ Keep memory low for Render Free Plan
    )
