"""
Reputation Agent - Checks account numbers and merchant reputation
"""
import logging
import re
from typing import Dict, Any, List
import firebase_admin
from firebase_admin import firestore

logger = logging.getLogger(__name__)


class ReputationAgent:
    """Check merchant and account reputation against fraud database"""

    def __init__(self):
        self.db = firestore.client()

    async def analyze(self, ocr_text: str, progress=None) -> Dict[str, Any]:
        """
        Extract account numbers and check reputation

        Returns:
            - accounts_analyzed: List of account numbers found
            - fraud_reports: Number of fraud reports for accounts
            - merchant: Verified merchant info if found
            - trust_level: Overall trust level
        """
        try:
            logger.info("Reputation agent analyzing extracted text")
            
            if progress:
                await progress.emit(
                    agent="reputation",
                    stage="reputation_check_started",
                    message="Checking account numbers and merchant reputation",
                    progress=72
                )

            # Extract account numbers (Nigerian format: 10 digits)
            account_numbers = self._extract_account_numbers(ocr_text)

            # Extract phone numbers
            phone_numbers = self._extract_phone_numbers(ocr_text)

            # Check each account against fraud database
            accounts_analyzed = []
            total_fraud_reports = 0

            for account_number in account_numbers:
                account_data = await self._check_account_reputation(account_number)
                accounts_analyzed.append(account_data)
                total_fraud_reports += account_data.get("fraud_reports", 0)

            # Check for verified merchant
            merchant = await self._check_merchant_verification(ocr_text)

            # Calculate trust level
            trust_level = self._calculate_trust_level(
                total_fraud_reports, merchant, accounts_analyzed
            )

            result = {
                "accounts_analyzed": accounts_analyzed,
                "phone_numbers": phone_numbers,
                "total_fraud_reports": total_fraud_reports,
                "merchant": merchant,
                "trust_level": trust_level,
            }

            logger.info(
                f"Reputation agent completed. Accounts checked: {len(accounts_analyzed)}"
            )
            
            if progress:
                merchant_status = f"Verified: {merchant['business_name']}" if merchant else f"{len(accounts_analyzed)} accounts checked"
                fraud_info = f"{total_fraud_reports} fraud reports" if total_fraud_reports > 0 else "No fraud history"
                await progress.emit(
                    agent="reputation",
                    stage="reputation_complete",
                    message=f"{merchant_status} - {fraud_info}",
                    progress=80,
                    details={
                        'accounts_checked': len(accounts_analyzed),
                        'fraud_reports': total_fraud_reports,
                        'merchant_verified': merchant is not None,
                        'trust_level': trust_level
                    }
                )
            
            return result

        except Exception as e:
            logger.error(f"Reputation agent error: {str(e)}")
            return {
                "accounts_analyzed": [],
                "phone_numbers": [],
                "total_fraud_reports": 0,
                "merchant": None,
                "trust_level": "unknown",
            }

    def _extract_account_numbers(self, text: str) -> List[str]:
        """Extract Nigerian account numbers (10 digits)"""
        pattern = r'\b\d{10}\b'
        matches = re.findall(pattern, text)
        return list(set(matches))  # Remove duplicates

    def _extract_phone_numbers(self, text: str) -> List[str]:
        """Extract Nigerian phone numbers"""
        pattern = r'\b(?:\+234|0)[789]\d{9}\b'
        matches = re.findall(pattern, text)
        return list(set(matches))

    async def _check_account_reputation(self, account_number: str) -> Dict:
        """Check account against fraud reports database"""
        try:
            # Hash account number for privacy
            import hashlib
            account_hash = hashlib.sha256(account_number.encode()).hexdigest()

            # Query Firestore for fraud reports using filter keyword argument
            from google.cloud.firestore_v1 import FieldFilter
            fraud_reports = (
                self.db.collection("fraud_reports")
                .where(filter=FieldFilter("account_hash", "==", account_hash))
                .where(filter=FieldFilter("status", "==", "verified"))
                .get()
            )

            fraud_count = len(fraud_reports)

            return {
                "account_number": account_number[:3] + "****" + account_number[-2:],  # Masked
                "fraud_reports": fraud_count,
                "risk_level": "high" if fraud_count >= 3 else "medium" if fraud_count >= 1 else "low",
            }

        except Exception as e:
            logger.error(f"Error checking account reputation: {str(e)}")
            return {
                "account_number": account_number[:3] + "****" + account_number[-2:],
                "fraud_reports": 0,
                "risk_level": "unknown",
            }

    async def _check_merchant_verification(self, text: str) -> Dict | None:
        """Check if merchant name in receipt is verified business"""
        try:
            # Extract potential business names (simplified)
            # In production, use NER (Named Entity Recognition)
            words = text.split()
            potential_names = [
                " ".join(words[i : i + 3]) for i in range(len(words) - 2)
            ]

            # Query verified businesses using filter keyword argument
            for name in potential_names:
                from google.cloud.firestore_v1 import FieldFilter
                businesses = (
                    self.db.collection("businesses")
                    .where(filter=FieldFilter("name", "==", name))
                    .where(filter=FieldFilter("verification.verified", "==", True))
                    .limit(1)
                    .get()
                )

                if len(businesses) > 0:
                    business_data = businesses[0].to_dict()
                    return {
                        "name": business_data.get("name"),
                        "verified": True,
                        "trust_score": business_data.get("trust_score", 75),
                        "business_id": businesses[0].id,
                    }

            return None

        except Exception as e:
            logger.error(f"Error checking merchant verification: {str(e)}")
            return None

    def _calculate_trust_level(
        self, fraud_reports: int, merchant: Dict | None, accounts: List[Dict]
    ) -> str:
        """Calculate overall trust level"""
        if fraud_reports >= 3:
            return "very_low"
        elif fraud_reports >= 1:
            return "low"
        elif merchant and merchant.get("verified"):
            return "very_high"
        elif len(accounts) == 0:
            return "medium"
        else:
            return "high"
