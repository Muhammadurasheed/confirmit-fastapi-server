"""
Reasoning Agent - Synthesizes all agent outputs into final verdict
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """Synthesize multi-agent results into coherent verdict and recommendation"""

    def __init__(self):
        pass

    async def synthesize(self, agent_results: Dict[str, Any], progress=None) -> Dict[str, Any]:
        """
        Combine results from all agents into final analysis

        Returns:
            - trust_score: 0-100
            - verdict: authentic|suspicious|fraudulent|unclear
            - issues: List of detected issues
            - recommendation: Action recommendation for user
        """
        try:
            logger.info("Reasoning agent synthesizing results")
            
            if progress:
                await progress.emit(
                    agent="reasoning",
                    stage="synthesis_started",
                    message="Synthesizing all agent findings into final verdict",
                    progress=85
                )

            vision_data = agent_results.get("vision", {})
            forensic_data = agent_results.get("forensic", {})
            metadata_data = agent_results.get("metadata", {})
            reputation_data = agent_results.get("reputation", {})

            # Calculate trust score (0-100)
            trust_score = await self._calculate_trust_score(
                vision_data, forensic_data, metadata_data, reputation_data
            )

            # Determine verdict
            verdict = self._determine_verdict(
                trust_score, forensic_data, reputation_data
            )

            # Compile issues
            issues = self._compile_issues(
                vision_data, forensic_data, metadata_data, reputation_data
            )

            # Generate recommendation
            recommendation = self._generate_recommendation(verdict, trust_score, issues)

            result = {
                "trust_score": trust_score,
                "verdict": verdict,
                "issues": issues,
                "recommendation": recommendation,
            }

            logger.info(
                f"Reasoning agent completed. Verdict: {verdict}, Score: {trust_score}"
            )
            
            if progress:
                verdict_messages = {
                    'authentic': 'Receipt appears authentic',
                    'suspicious': 'Receipt has suspicious elements',
                    'fraudulent': 'Receipt appears fraudulent',
                    'unclear': 'Unable to determine authenticity'
                }
                await progress.emit(
                    agent="reasoning",
                    stage="synthesis_complete",
                    message=f"{verdict_messages.get(verdict, 'Analysis complete')} - Trust Score: {trust_score}%",
                    progress=95,
                    details={
                        'trust_score': trust_score,
                        'verdict': verdict,
                        'issues_count': len(issues)
                    }
                )
            
            return result

        except Exception as e:
            logger.error(f"Reasoning agent error: {str(e)}")
            return {
                "trust_score": 50,
                "verdict": "unclear",
                "issues": [
                    {
                        "type": "analysis_error",
                        "severity": "medium",
                        "description": "Unable to complete full analysis",
                    }
                ],
                "recommendation": "Manual verification recommended",
            }

    async def _calculate_trust_score(
        self,
        vision: Dict,
        forensic: Dict,
        metadata: Dict,
        reputation: Dict,
    ) -> int:
        """Calculate weighted trust score from all agents - STRICT SCORING"""
        
        # Start with neutral base score
        score = 50

        # CRITICAL: Vision confidence (weight: 40%) - this is most important
        ocr_confidence = vision.get("confidence", 0) if vision else 0
        ocr_text_length = len(vision.get("ocr_text", "")) if vision else 0
        
        # Harsh penalties for poor vision analysis
        if ocr_confidence < 30:
            score -= 30  # Severe penalty for failed OCR
        elif ocr_confidence < 50:
            score -= 15  # Moderate penalty for weak OCR
        else:
            score += (ocr_confidence - 50) * 0.6  # Reward good OCR
        
        # Penalty if almost no text extracted
        if ocr_text_length < 20:
            score -= 25  # Cannot trust a receipt with no extracted text
        
        # Visual anomalies (weight: 30%) - CRITICAL fraud indicators
        visual_anomalies = vision.get("visual_anomalies", []) if vision else []
        anomaly_count = len(visual_anomalies)
        if anomaly_count > 0:
            # Each anomaly is a major red flag
            score -= anomaly_count * 8
            logger.warning(f"⚠️ Visual anomalies detected: {anomaly_count} issues")

        # Forensic analysis (weight: 20%)
        manipulation_score = forensic.get("manipulation_score", 0) if forensic else 0
        if manipulation_score > 0:
            score -= manipulation_score * 0.4

        # Metadata (weight: 5%)
        metadata_flags = len(metadata.get("flags", [])) if metadata else 0
        score -= metadata_flags * 5

        # Reputation (weight: 5%)
        fraud_reports = reputation.get("total_fraud_reports", 0) if reputation else 0
        if fraud_reports > 0:
            score -= fraud_reports * 15  # Major penalty for fraud history

        # Bonus for verified merchant (only if other signals are positive)
        merchant = reputation.get("merchant") if reputation else None
        if merchant and isinstance(merchant, dict) and merchant.get("verified") and score > 40:
            score += 10

        # Clamp to 0-100
        final_score = max(0, min(100, int(score)))
        
        logger.info(f"📊 Trust Score Breakdown: Base=50, OCR={ocr_confidence}%, Anomalies={anomaly_count}, Final={final_score}")
        
        return final_score

    def _determine_verdict(
        self, trust_score: int, forensic: Dict, reputation: Dict
    ) -> str:
        """Determine final verdict based on score and critical findings - STRICT THRESHOLDS"""
        
        # Critical red flags - instant fraud verdict
        fraud_reports = reputation.get("total_fraud_reports", 0) if reputation else 0
        manipulation_score = forensic.get("manipulation_score", 0) if forensic else 0

        if fraud_reports >= 2 or manipulation_score >= 70:
            logger.warning(f"🚨 FRAUD VERDICT: fraud_reports={fraud_reports}, manipulation={manipulation_score}%")
            return "fraudulent"
        
        # Strict thresholds
        if trust_score >= 80:
            return "authentic"
        elif trust_score >= 60:
            return "suspicious"
        elif trust_score >= 35:
            return "unclear"
        else:
            logger.warning(f"🚨 FRAUDULENT: trust_score={trust_score} below threshold")
            return "fraudulent"

    def _compile_issues(
        self,
        vision: Dict,
        forensic: Dict,
        metadata: Dict,
        reputation: Dict,
    ) -> List[Dict]:
        """Compile all detected issues"""
        issues = []

        # Vision issues - treat visual anomalies as HIGH severity
        visual_anomalies = vision.get("visual_anomalies", [])
        for anomaly in visual_anomalies:
            # Visual anomalies detected by AI are RED FLAGS
            issues.append({
                "type": "visual_manipulation",
                "severity": "high",
                "description": f"🚨 {anomaly}",
            })

        # Flag poor OCR extraction
        ocr_confidence = vision.get("confidence", 0)
        if ocr_confidence < 50:
            issues.append({
                "type": "ocr_failure",
                "severity": "high",
                "description": f"Failed to extract text reliably (confidence: {ocr_confidence}%) - possible tampering or poor image quality",
            })
        
        # Flag if almost no text extracted
        ocr_text_length = len(vision.get("ocr_text", ""))
        if ocr_text_length < 20:
            issues.append({
                "type": "insufficient_data",
                "severity": "high",
                "description": f"Very little text extracted ({ocr_text_length} characters) - receipt may be fake or image corrupted",
            })

        # Forensic issues
        techniques = forensic.get("techniques_detected", [])
        for technique in techniques:
            issues.append({
                "type": "forensic_finding",
                "severity": "high",
                "description": f"Detected: {technique}",
            })

        # Metadata issues
        metadata_flags = metadata.get("flags", [])
        for flag in metadata_flags:
            issues.append({
                "type": "metadata_issue",
                "severity": "medium",
                "description": flag,
            })

        # Reputation issues
        fraud_reports = reputation.get("total_fraud_reports", 0)
        if fraud_reports > 0:
            issues.append({
                "type": "fraud_history",
                "severity": "high",
                "description": f"Account has {fraud_reports} verified fraud report(s)",
            })

        return issues

    def _generate_recommendation(
        self, verdict: str, trust_score: int, issues: List[Dict]
    ) -> str:
        """Generate actionable recommendation - CLEAR AND DIRECT"""
        
        high_severity_issues = [i for i in issues if i.get("severity") == "high"]
        
        if verdict == "fraudulent":
            issue_summary = f" {len(high_severity_issues)} critical issues detected." if high_severity_issues else ""
            return f"🚨 FRAUDULENT RECEIPT DETECTED - DO NOT TRUST.{issue_summary} Report this immediately and verify through official channels."
        
        elif verdict == "suspicious":
            issue_list = ", ".join([i["description"][:50] for i in high_severity_issues[:2]])
            return f"⚠️ HIGHLY SUSPICIOUS - {issue_list}. Verify independently before accepting this receipt."
        
        elif verdict == "unclear":
            if len(issues) >= 3:
                return f"⚠️ CANNOT VERIFY - {len(issues)} issues detected including {len(high_severity_issues)} critical problems. Do not rely on this receipt."
            else:
                return f"ℹ️ INSUFFICIENT DATA - Unable to fully verify authenticity ({len(issues)} issues). Request clearer documentation."
        
        else:  # authentic
            if trust_score >= 85:
                return "✅ AUTHENTIC - This receipt appears completely legitimate. All verification checks passed."
            else:
                return f"✅ LIKELY AUTHENTIC - Receipt appears genuine (score: {trust_score}/100). Minor concerns noted but overall trustworthy."
