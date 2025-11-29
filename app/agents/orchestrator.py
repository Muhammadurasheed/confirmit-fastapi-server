"""
Multi-Agent Orchestrator for Receipt Analysis
Coordinates vision, forensic, metadata, reputation, and reasoning agents
"""
import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ReceiptAnalysisOrchestrator:
    """Orchestrates multiple AI agents for comprehensive receipt analysis"""

    def __init__(
        self,
        vision_agent,
        forensic_agent,
        metadata_agent,
        reputation_agent,
        reasoning_agent,
    ):
        self.vision_agent = vision_agent
        self.forensic_agent = forensic_agent
        self.metadata_agent = metadata_agent
        self.reputation_agent = reputation_agent
        self.reasoning_agent = reasoning_agent
        self.forensic_progress_log = []  # Track forensic agent progress
        
    def _forensic_progress_callback(self, progress_data):
        """Callback to capture forensic agent progress"""
        self.forensic_progress_log.append(progress_data)
        logger.info(f"[Forensic Progress] {progress_data['stage']}: {progress_data['message']}")

    async def analyze_receipt(
        self, image_path: str, receipt_id: str
    ) -> Dict[str, Any]:
        """
        Run all agents in parallel and synthesize results

        Returns comprehensive analysis including trust score, verdict, issues, etc.
        """
        from app.core.progress_emitter import ProgressEmitter
        
        start_time = datetime.now()
        agent_results = {}
        agent_logs = []
        self.forensic_progress_log = []  # Reset progress log
        
        # Initialize progress emitter
        progress = ProgressEmitter(receipt_id)

        try:
            # Run agents in parallel with timeouts
            logger.info(f"Starting multi-agent analysis for receipt {receipt_id}")
            
            await progress.emit(
                agent="orchestrator",
                stage="analysis_started",
                message="Initializing AI agents for receipt analysis",
                progress=5
            )

            # Execute vision and metadata first (parallel)
            await progress.emit(
                agent="orchestrator",
                stage="agents_running",
                message="Running vision and metadata analysis",
                progress=20
            )
            
            vision_result, metadata_result = await asyncio.gather(
                self._run_vision_agent(image_path, receipt_id, progress),
                self._run_metadata_agent(image_path, receipt_id, progress),
                return_exceptions=True,
            )
            
            # Check vision result
            if not isinstance(vision_result, Exception):
                agent_results["vision"] = vision_result
                agent_logs.append({
                    "agent": "vision",
                    "status": "success",
                    "confidence": vision_result.get("confidence", 0),
                })
                # CRITICAL DEBUGGING: Log what vision agent returned
                logger.info(f"✅ Vision Agent Result:")
                logger.info(f"  - ocr_text length: {len(vision_result.get('ocr_text', ''))}")
                logger.info(f"  - ocr_text preview: {vision_result.get('ocr_text', '')[:100]}...")
                logger.info(f"  - merchant_name: {vision_result.get('merchant_name')}")
                logger.info(f"  - total_amount: {vision_result.get('total_amount')}")
                logger.info(f"  - confidence: {vision_result.get('confidence')}")
            else:
                logger.error(f"❌ Vision Agent Failed: {str(vision_result)}")
            
            # Check metadata result
            if not isinstance(metadata_result, Exception):
                agent_results["metadata"] = metadata_result
                agent_logs.append({
                    "agent": "metadata",
                    "status": "success",
                    "flags": len(metadata_result.get("flags", [])),
                })
                logger.info(f"✅ Metadata Agent Result: {len(metadata_result.get('flags', []))} flags")
            else:
                logger.error(f"❌ Metadata Agent Failed: {str(metadata_result)}")
            
            # Now run forensic with vision context for better analysis
            await progress.emit(
                agent="orchestrator",
                stage="forensic_analysis",
                message="Running forensic ELA analysis with vision context",
                progress=40
            )
            
            forensic_result = await self._run_forensic_agent(
                image_path, receipt_id, progress, vision_data=vision_result if not isinstance(vision_result, Exception) else None
            )
            
            if not isinstance(forensic_result, Exception):
                agent_results["forensic"] = forensic_result
                agent_logs.append({
                    "agent": "forensic",
                    "status": "success",
                    "manipulation_score": forensic_result.get("manipulation_score", 0),
                })
                # CRITICAL DEBUGGING: Log forensic data structure
                logger.info(f"✅ Forensic Agent Result:")
                logger.info(f"  - manipulation_score: {forensic_result.get('manipulation_score')}")
                logger.info(f"  - verdict: {forensic_result.get('verdict')}")
                logger.info(f"  - forensic_findings: {len(forensic_result.get('forensic_findings', []))} items")
                logger.info(f"  - technical_details keys: {list(forensic_result.get('technical_details', {}).keys())}")
                
                # Check if ELA heatmap exists
                ela_data = forensic_result.get('technical_details', {}).get('ela_analysis', {})
                logger.info(f"  - ELA heatmap: {len(ela_data.get('heatmap', []))} rows")
                logger.info(f"  - ELA suspicious_regions: {len(ela_data.get('suspicious_regions', []))}")
            else:
                logger.error(f"❌ Forensic Agent Failed: {str(forensic_result)}")

            # Run reputation agent if we have extracted text
            if "vision" in agent_results:
                # FIX: Handle None values safely before string formatting
                # Python's None displays as "None" in UI if not handled
                merchant_name = agent_results['vision'].get('merchant_name')
                merchant_name = merchant_name if merchant_name else "Unknown Merchant"
                
                total_amount = agent_results['vision'].get('total_amount')
                total_amount = total_amount if total_amount else "0.00"
                
                await progress.emit(
                    agent="orchestrator",
                    stage="reputation_check",
                    message=f"Verifying {merchant_name} (₦{total_amount})",
                    progress=70,
                    details={
                        'merchant': merchant_name,
                        'amount': total_amount
                    }
                )
                
                ocr_text = agent_results["vision"].get("ocr_text", "")
                reputation_result = await self._run_reputation_agent(
                    ocr_text, receipt_id, progress
                )
                if not isinstance(reputation_result, Exception):
                    agent_results["reputation"] = reputation_result
                    agent_logs.append(
                        {
                            "agent": "reputation",
                            "status": "success",
                            "accounts_checked": len(
                                reputation_result.get("accounts_analyzed", [])
                            ),
                        }
                    )

            # Run reasoning agent to synthesize all results
            final_analysis = await self._run_reasoning_agent(agent_results, receipt_id, progress)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # --- PREPARE FINAL RESPONSE (FLATTENING STRATEGY) ---
            # We serialize heavy objects HERE in Python to ensure they survive the trip to Firestore
            
            import json
            from app.core.progress_emitter import NumpyJsonEncoder
            
            forensic_data = agent_results.get("forensic", {})
            technical_details = forensic_data.get("technical_details", {})
            ela_analysis = technical_details.get("ela_analysis", {})
            
            # CRITICAL FIX: Force default values for image_dimensions to prevent undefined downstream
            # If image_dimensions is missing, provide a safe default
            safe_image_dimensions = ela_analysis.get("image_dimensions")
            if not safe_image_dimensions:
                safe_image_dimensions = {"width": 0, "height": 0}
            
            # Safe statistics with defaults
            safe_statistics = ela_analysis.get("statistics")
            if not safe_statistics:
                safe_statistics = {}
            
            # Force serialization of deep structures
            # This solves "INVALID_ARGUMENT: Property analysis contains an invalid nested entity"
            serialized_technical = json.dumps(technical_details, cls=NumpyJsonEncoder)
            serialized_heatmap = json.dumps(ela_analysis.get("heatmap", []), cls=NumpyJsonEncoder)
            serialized_pixel_diff = json.dumps(ela_analysis.get("pixel_diff", {}), cls=NumpyJsonEncoder)
            serialized_forensic_findings = json.dumps(forensic_data.get("forensic_findings", []), cls=NumpyJsonEncoder)
            serialized_agent_logs = json.dumps(agent_logs, cls=NumpyJsonEncoder)
            
            vision_data = agent_results.get("vision", {})
            
            # Compile final response with ALL data pre-serialized for Firestore safety
            final_response = {
                "receipt_id": receipt_id,
                "trust_score": final_analysis.get("trust_score", 50),
                "verdict": final_analysis.get("verdict", "unclear"),
                "issues": final_analysis.get("issues", []),
                "recommendation": final_analysis.get("recommendation", ""),
                "ocr_text": vision_data.get("ocr_text", "") or "No text extracted",
                
                # Construct a SAFE forensic_details object
                "forensic_details": {
                    "ocr_confidence": vision_data.get("confidence", 0),
                    "manipulation_score": forensic_data.get("manipulation_score", 0),
                    "metadata_flags": agent_results.get("metadata", {}).get("flags", []),
                    "forensic_progress": self.forensic_progress_log,
                    "forensic_verdict": forensic_data.get("verdict", "unclear"),
                    "forensic_summary": forensic_data.get("summary", ""),
                    
                    # Store heavy/nested data as JSON Strings (already serialized)
                    "forensic_findings": serialized_forensic_findings,
                    "technical_details": serialized_technical,
                    "heatmap": serialized_heatmap,
                    "pixel_diff": serialized_pixel_diff,
                    
                    # Flattened top-level stats for easy access
                    "techniques_detected": forensic_data.get("techniques_detected", []),
                    "authenticity_indicators": forensic_data.get("authenticity_indicators", []),
                    "suspicious_regions_count": len(ela_analysis.get("suspicious_regions", [])),
                    "manipulation_detected": ela_analysis.get("manipulation_detected", False),
                    
            # CONDITIONAL ASSIGNMENT - Only add if data exists
            # NEVER send None/empty dicts to avoid Firestore undefined errors
            **({"image_dimensions": safe_image_dimensions} if safe_image_dimensions else {}),
            **({"statistics": safe_statistics} if safe_statistics else {}),
                },
                "merchant": agent_results.get("reputation", {}).get("merchant"),
                "agent_logs": serialized_agent_logs,  # Pre-serialized
                "processing_time_seconds": processing_time,
            }
            
            # No need for numpy conversion - we've pre-serialized all heavy objects above
            
            # CRITICAL DEBUGGING: Log final response structure before returning
            logger.info(f"📊 ORCHESTRATOR FINAL RESPONSE:")
            logger.info(f"  - receipt_id: {final_response['receipt_id']}")
            logger.info(f"  - ocr_text length: {len(final_response['ocr_text'])}")
            logger.info(f"  - trust_score: {final_response['trust_score']}")
            logger.info(f"  - verdict: {final_response['verdict']}")
            logger.info(f"  - forensic_findings: SERIALIZED (string)")
            logger.info(f"  - technical_details: SERIALIZED (string)")
            logger.info(f"  - heatmap: SERIALIZED (string)")
            logger.info(f"  - agent_logs: SERIALIZED (string)")
            
            # Store complete results to Firebase
            await progress.emit(
                agent="orchestrator",
                stage="storing_results",
                message="Saving analysis results",
                progress=95
            )
            
            return final_response

        except Exception as e:
            logger.error(f"Orchestrator error for receipt {receipt_id}: {str(e)}")
            return {
                "receipt_id": receipt_id,
                "trust_score": 0,
                "verdict": "unclear",
                "issues": [
                    {
                        "type": "analysis_error",
                        "severity": "high",
                        "description": f"Analysis failed: {str(e)}",
                    }
                ],
                "recommendation": "Unable to verify receipt. Please try again.",
                "forensic_details": {
                    "ocr_confidence": 0,
                    "manipulation_score": 0,
                    "metadata_flags": [],
                },
                "merchant": None,
                "agent_logs": agent_logs,
            }

    async def _run_vision_agent(self, image_path: str, receipt_id: str, progress) -> Dict:
        """Run Gemini Vision agent for OCR and visual analysis"""
        try:
            logger.info(f"Running vision agent for {receipt_id}")
            result = await self.vision_agent.analyze(image_path, progress)
            logger.info(f"Vision agent completed for {receipt_id}")
            return result
        except Exception as e:
            logger.error(f"Vision agent failed for {receipt_id}: {str(e)}")
            raise

    async def _run_forensic_agent(self, image_path: str, receipt_id: str, progress, vision_data: Dict = None) -> Dict:
        """Run enhanced forensic analysis with progress tracking and vision context"""
        try:
            logger.info(f"Running enhanced forensic agent for {receipt_id}")
            # Create new forensic agent instance with progress callback
            from app.agents.forensic_agent import EnhancedForensicAgent
            
            async def forensic_progress_wrapper(data):
                """Wrapper to emit forensic progress through main progress emitter"""
                await progress.emit(
                    agent="forensic",
                    stage=data.get('stage', 'analyzing'),
                    message=data.get('message', 'Running forensic checks'),
                    progress=data.get('progress', 40),
                    details=data.get('details', {})
                )
            
            forensic = EnhancedForensicAgent(progress_callback=forensic_progress_wrapper)
            
            # Pass vision data for context-aware analysis
            receipt_context = {
                'merchant_name': vision_data.get('merchant_name') if vision_data else None,
                'total_amount': vision_data.get('total_amount') if vision_data else None,
            }
            
            result = await forensic.analyze(image_path, receipt_data=receipt_context)
            logger.info(f"Forensic agent completed for {receipt_id}")
            return result
        except Exception as e:
            logger.error(f"Forensic agent failed for {receipt_id}: {str(e)}")
            raise

    async def _run_metadata_agent(self, image_path: str, receipt_id: str, progress) -> Dict:
        """Run metadata extraction agent"""
        try:
            logger.info(f"Running metadata agent for {receipt_id}")
            result = await self.metadata_agent.analyze(image_path, progress)
            logger.info(f"Metadata agent completed for {receipt_id}")
            return result
        except Exception as e:
            logger.error(f"Metadata agent failed for {receipt_id}: {str(e)}")
            raise

    async def _run_reputation_agent(self, ocr_text: str, receipt_id: str, progress) -> Dict:
        """Run reputation checking agent"""
        try:
            logger.info(f"Running reputation agent for {receipt_id}")
            result = await self.reputation_agent.analyze(ocr_text, progress)
            logger.info(f"Reputation agent completed for {receipt_id}")
            return result
        except Exception as e:
            logger.error(f"Reputation agent failed for {receipt_id}: {str(e)}")
            raise

    async def _run_reasoning_agent(
        self, agent_results: Dict, receipt_id: str, progress
    ) -> Dict:
        """Run reasoning agent to synthesize all results"""
        try:
            logger.info(f"Running reasoning agent for {receipt_id}")
            result = await self.reasoning_agent.synthesize(agent_results, progress)
            logger.info(f"Reasoning agent completed for {receipt_id}")
            return result
        except Exception as e:
            logger.error(f"Reasoning agent failed for {receipt_id}: {str(e)}")
            # Return default safe result
            return {
                "trust_score": 50,
                "verdict": "unclear",
                "issues": [],
                "recommendation": "Unable to complete analysis",
            }
