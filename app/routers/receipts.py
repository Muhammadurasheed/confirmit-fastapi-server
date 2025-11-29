from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import os
import httpx
from app.agents.orchestrator import ReceiptAnalysisOrchestrator
from app.agents.vision_agent import VisionAgent
from app.agents.forensic_agent import EnhancedForensicAgent
from app.agents.metadata_agent import MetadataAgent
from app.agents.reputation_agent import ReputationAgent
from app.agents.reasoning_agent import ReasoningAgent

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize agents
from app.config import settings
gemini_api_key = settings.GEMINI_API_KEY
vision_agent = VisionAgent(gemini_api_key) if gemini_api_key else None
forensic_agent = EnhancedForensicAgent()  # Now using enhanced forensics
metadata_agent = MetadataAgent()
reputation_agent = ReputationAgent()
reasoning_agent = ReasoningAgent()

# Initialize orchestrator
if vision_agent:
    orchestrator = ReceiptAnalysisOrchestrator(
        vision_agent=vision_agent,
        forensic_agent=forensic_agent,
        metadata_agent=metadata_agent,
        reputation_agent=reputation_agent,
        reasoning_agent=reasoning_agent,
    )
else:
    logger.warning("Gemini API key not configured. Vision agent disabled.")
    orchestrator = None


class AnalyzeReceiptRequest(BaseModel):
    image_url: str
    receipt_id: str


@router.post("/analyze-receipt")
async def analyze_receipt(request: AnalyzeReceiptRequest) -> Dict[str, Any]:
    """
    Analyze receipt image for authenticity using multi-agent system

    This endpoint coordinates multiple AI agents:
    - Vision Agent (Gemini): OCR and visual analysis
    - Forensic Agent: Image manipulation detection
    - Metadata Agent: EXIF data analysis
    - Reputation Agent: Merchant/account verification
    - Reasoning Agent: Synthesis and verdict
    """
    try:
        logger.info(f"🎯 Received analysis request for receipt: {request.receipt_id}")
        logger.info(f"📸 Image URL: {request.image_url[:100]}...")

        if not orchestrator:
            logger.error("❌ Orchestrator not initialized - missing GEMINI_API_KEY")
            raise HTTPException(
                status_code=503,
                detail="AI service not properly configured. Check GEMINI_API_KEY.",
            )

        # Download image from Cloudinary
        logger.info(f"⬇️ Downloading image from Cloudinary...")
        image_path = await download_image(request.image_url, request.receipt_id)
        logger.info(f"✅ Image downloaded: {image_path}")

        # Run multi-agent analysis
        logger.info(f"🤖 Starting multi-agent analysis...")
        result = await orchestrator.analyze_receipt(image_path, request.receipt_id)

        logger.info(f"✅ Analysis completed for receipt: {request.receipt_id}")
        logger.info(f"📊 Trust Score: {result.get('trust_score')}, Verdict: {result.get('verdict')}")
        
        # CRITICAL DEBUGGING: Log the exact structure being returned
        # Note: Heavy objects are now pre-serialized as JSON strings
        forensic_details = result.get('forensic_details', {})
        logger.info(f"🔍 AI SERVICE RETURNING:")
        logger.info(f"  - ocr_text length: {len(result.get('ocr_text', ''))}")
        
        # Handle pre-serialized forensic_findings (now a JSON string)
        findings = forensic_details.get('forensic_findings', '[]')
        if isinstance(findings, str):
            logger.info(f"  - forensic_findings: PRE-SERIALIZED STRING ({len(findings)} chars)")
        else:
            logger.info(f"  - forensic_findings: {len(findings)} items")
        
        # Handle pre-serialized technical_details (now a JSON string)
        tech_details = forensic_details.get('technical_details', '{}')
        if isinstance(tech_details, str):
            logger.info(f"  - technical_details: PRE-SERIALIZED STRING ({len(tech_details)} chars)")
        else:
            logger.info(f"  - technical_details keys: {list(tech_details.keys())}")
        
        # Handle pre-serialized heatmap (now a JSON string)
        heatmap = forensic_details.get('heatmap', '[]')
        if isinstance(heatmap, str):
            logger.info(f"  - heatmap: PRE-SERIALIZED STRING ({len(heatmap)} chars)")
        else:
            logger.info(f"  - heatmap shape: {len(heatmap)} rows")
        
        logger.info(f"  - Full forensic_details keys: {list(forensic_details.keys())}")
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed for receipt {request.receipt_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def download_image(image_url: str, receipt_id: str) -> str:
    """Download image from Cloudinary to local temp storage"""
    try:
        import tempfile

        logger.info(f"⬇️ Downloading image from: {image_url[:80]}...")
        
        async with httpx.AsyncClient() as client:
            # Cloudinary public images don't need authentication
            # Just make sure we're using the secure_url from the upload
            response = await client.get(image_url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()

            # Save to temp file
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, f"{receipt_id}.jpg")

            with open(image_path, "wb") as f:
                f.write(response.content)

            logger.info(f"✅ Image downloaded successfully: {image_path} ({len(response.content)} bytes)")
            return image_path

    except httpx.HTTPStatusError as e:
        logger.error(f"❌ HTTP error downloading image: {e.response.status_code} - {e.response.text}")
        raise Exception(f"Failed to download image from Cloudinary: HTTP {e.response.status_code}")
    except httpx.TimeoutException:
        logger.error(f"❌ Timeout downloading image after 30s")
        raise Exception("Image download timed out - check Cloudinary URL")
    except Exception as e:
        logger.error(f"❌ Failed to download image: {str(e)}")
        raise Exception(f"Image download failed: {str(e)}")
