"""
Real-time Progress Emitter for AI Agents
Emits detailed progress updates to Firebase for backend consumption
"""
import logging
import json
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
from app.core.firebase import db

logger = logging.getLogger(__name__)


class NumpyJsonEncoder(json.JSONEncoder):
    """
    A specific JSON encoder that handles NumPy types cleanly.
    This prevents the fallback to str() which creates invalid JSON for frontend.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj) if not (np.isnan(obj) or np.isinf(obj)) else 0.0
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ProgressEmitter:
    """Emits detailed agent progress to Firebase Firestore"""
    
    def __init__(self, receipt_id: str):
        self.receipt_id = receipt_id
        self.progress_ref = db.collection('receipts').document(receipt_id)
        
    async def emit(
        self,
        agent: str,
        stage: str,
        message: str,
        progress: int,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Emit progress update to Firebase
        
        Args:
            agent: Name of agent (vision, forensic, metadata, reputation, reasoning)
            stage: Current stage (ocr_started, forensics_running, etc.)
            message: User-friendly message describing what's happening
            progress: Progress percentage (0-100)
            details: Optional dict with specific details (e.g., extracted data)
        """
        try:
            # Base update data
            update_data = {
                'progress_agent': str(agent),
                'progress_stage': str(stage),
                'progress_message': str(message),
                'progress_percentage': int(progress),
                'progress_timestamp': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat(),
            }
            
            # Flatten details into JSON strings using the NumpyEncoder
            # This ensures the frontend receives valid JSON strings, not Python string representations
            if details:
                # CRITICAL FIX: Ensure details is actually a dict before iterating
                if isinstance(details, dict):
                    for key, value in details.items():
                        # AGGRESSIVE FILTERING: Skip nulls, Nones, empty dicts, empty lists
                        if value is None or value == {} or value == []:
                            continue
                            
                        if isinstance(value, (str, int, float, bool)):
                            update_data[f'progress_detail_{key}'] = value
                        else:
                            # CRITICAL FIX: Use NumpyJsonEncoder to guarantee valid JSON
                            try:
                                update_data[f'progress_detail_{key}'] = json.dumps(
                                    value, 
                                    cls=NumpyJsonEncoder
                                )
                            except Exception as e:
                                logger.error(f"Failed to serialize {key}: {e}")
                                update_data[f'progress_detail_{key}'] = str(value)  # Last resort
                else:
                    # If details is just a string (e.g., an error message), store it safely
                    update_data['progress_detail_info'] = str(details)
            
            self.progress_ref.update(update_data)
            
            logger.info(f"📡 [{self.receipt_id}] {agent}: {message} ({progress}%)")
            
        except Exception as e:
            logger.error(f"❌ Failed to emit progress for {self.receipt_id}: {str(e)}", exc_info=True)
            # Don't fail the analysis if progress emission fails - just log it
