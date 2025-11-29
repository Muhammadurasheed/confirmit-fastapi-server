"""
Vision Agent - Gemini Vision Primary (2025 Architecture)
"""
import logging
import json
from typing import Dict, Any
import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)


class VisionAgent:
    """Gemini Vision API wrapper for receipt OCR and visual analysis"""

    def __init__(self, api_key: str):
        if not api_key:
            logger.error("Missing GEMINI_API_KEY")
            raise ValueError("GEMINI_API_KEY is required")
        
        genai.configure(api_key=api_key)
        self.model_name = 'models/gemini-2.5-flash'  # Fast, accurate, available
        self.model = genai.GenerativeModel(self.model_name)

    async def analyze(self, image_path: str, progress=None) -> Dict[str, Any]:
        """
        Extract structured data from receipt using Gemini Vision.
        
        Returns:
            - merchant_name: Business/person name
            - total_amount: Final amount (float)
            - currency: Currency symbol
            - ocr_text: Full text extraction
            - confidence: Extraction confidence (0-100)
        """
        try:
            if progress:
                await progress.emit(
                    agent="vision",
                    stage="ocr_started",
                    message="🤖 Gemini Vision analyzing receipt structure...",
                    progress=10
                )

            # Load image using PIL
            from PIL import Image
            logger.info(f"Loading image with PIL: {image_path}")
            img = Image.open(image_path)

            # Structured Prompt for JSON extraction
            prompt = """
            Analyze this image. It is likely a payment receipt. 
            Extract the following information in strict JSON format:
            1. merchant_name (string): The name of the business/person.
            2. total_amount (number): The final total amount paid (remove currency symbols, handle 'k' or 'm' notation if present).
            3. currency (string): The currency symbol (e.g., ₦, $, £).
            4. date (string): The transaction date (ISO format YYYY-MM-DD if possible).
            5. transaction_id (string): Any reference number, session ID, or transaction hash.
            6. ocr_text (string): A raw transcription of all visible text.
            7. confidence (number): Your confidence score (0-100) in this extraction.
            8. visual_anomalies (array): List of any suspicious patterns (font differences, color inconsistencies, overlays).

            Return ONLY the JSON object. Do not include Markdown formatting like ```json.
            """

            # Call Gemini with PIL Image directly (with robust fallback chain)
            try:
                response = await self.model.generate_content_async([prompt, img])
            except Exception as model_error:
                if "404" in str(model_error) or "not found" in str(model_error).lower():
                    logger.warning(f"Model {self.model_name} failed, trying models/gemini-2.5-pro...")
                    try:
                        self.model = genai.GenerativeModel('models/gemini-2.5-pro')
                        response = await self.model.generate_content_async([prompt, img])
                    except Exception as second_error:
                        logger.error(f"All Gemini models failed. Last error: {str(second_error)}")
                        raise ValueError(f"Gemini API unavailable. Please verify your GEMINI_API_KEY has model access.")
                else:
                    raise model_error
            
            # Clean response (strip markdown if Gemini adds it)
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            
            data = json.loads(raw_text.strip())
            
            # Sanitize Amount (Handle strings like "1,500,000" or "1.5m")
            amount = data.get('total_amount', 0)
            if isinstance(amount, str):
                # Remove commas and currency symbols
                clean_amount = ''.join(c for c in amount if c.isdigit() or c == '.')
                data['total_amount'] = float(clean_amount) if clean_amount else 0.0
            
            # Ensure confidence is set
            if 'confidence' not in data:
                data['confidence'] = 85  # Default high confidence for Gemini

            if progress:
                await progress.emit(
                    agent="vision",
                    stage="ocr_complete",
                    message=f"Extracted: {data.get('merchant_name', 'Unknown')} - {data.get('currency', '₦')}{data.get('total_amount', 0)}",
                    progress=30,
                    details={
                        'merchant': data.get('merchant_name'),
                        'amount': data.get('total_amount'),
                        'confidence': data.get('confidence')
                    }
                )

            logger.info(f"✅ Gemini extraction complete - {data.get('merchant_name')}: {data.get('total_amount')}")
            return data

        except Exception as e:
            logger.error(f"Gemini Vision failed: {str(e)}")
            # CRITICAL FALLBACK: If Gemini completely fails, return mock data so pipeline doesn't break
            # This ensures forensic analysis can still run
            return {
                "merchant_name": "Unknown Merchant",
                "total_amount": 0.0,
                "currency": "₦",
                "date": "",
                "transaction_id": "",
                "ocr_text": "OCR extraction unavailable - Gemini API error",
                "confidence": 0,
                "visual_anomalies": [],
                "error": "Gemini API unavailable - check GEMINI_API_KEY and model access"
            }

