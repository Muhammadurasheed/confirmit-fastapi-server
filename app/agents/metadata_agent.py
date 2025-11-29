"""
Metadata Agent - Extracts and analyzes EXIF and file metadata
"""
import logging
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataAgent:
    """Extract and analyze image metadata for authenticity"""

    def __init__(self):
        self.editing_software = [
            "adobe",
            "photoshop",
            "gimp",
            "paint",
            "canva",
            "pixlr",
            "snapseed",
        ]

    async def analyze(self, image_path: str, progress=None) -> Dict[str, Any]:
        """
        Extract and analyze metadata

        Returns:
            - exif_data: Extracted EXIF data
            - flags: List of suspicious metadata findings
            - software_detected: Editing software detected
            - datetime_consistency: Whether dates are consistent
        """
        try:
            logger.info(f"Metadata agent analyzing: {image_path}")
            
            if progress:
                await progress.emit(
                    agent="metadata",
                    stage="metadata_extraction",
                    message="Analyzing image metadata and EXIF data",
                    progress=45
                )

            # Load image
            img = Image.open(image_path)

            # Extract EXIF data
            exif_data = self._extract_exif(img)

            # Analyze metadata
            flags = []
            software_detected = None

            # Check for editing software
            software = exif_data.get("Software", "").lower()
            for editor in self.editing_software:
                if editor in software:
                    software_detected = editor
                    flags.append(
                        f"Edited with {editor} - may indicate manipulation"
                    )
                    break

            # Check if EXIF data was stripped (suspicious)
            if not exif_data or len(exif_data) < 3:
                flags.append("EXIF data missing or stripped - suspicious")

            # Check datetime consistency
            datetime_consistent = self._check_datetime_consistency(exif_data)
            if not datetime_consistent:
                flags.append("Inconsistent datetime metadata")

            # Check for GPS data (receipts shouldn't have GPS usually)
            if "GPSInfo" in exif_data:
                flags.append("GPS data present (unusual for receipts)")

            result = {
                "exif_data": exif_data,
                "flags": flags,
                "software_detected": software_detected,
                "datetime_consistent": datetime_consistent,
                "risk_level": "high" if len(flags) >= 3 else "medium" if len(flags) >= 1 else "low",
            }

            logger.info(
                f"Metadata agent completed. Flags: {len(flags)}"
            )
            
            if progress:
                message = f"Found {len(flags)} metadata flags" if flags else "Clean metadata - no editing detected"
                await progress.emit(
                    agent="metadata",
                    stage="metadata_complete",
                    message=message,
                    progress=50,
                    details={
                        'flags_count': len(flags),
                        'editing_software': software_detected
                    }
                )
            
            return result

        except Exception as e:
            logger.error(f"Metadata agent error: {str(e)}")
            # Return safe default
            return {
                "exif_data": {},
                "flags": ["Unable to extract metadata"],
                "software_detected": None,
                "datetime_consistent": True,
                "risk_level": "low",
            }

    def _extract_exif(self, img: Image.Image) -> Dict[str, Any]:
        """Extract EXIF data from image"""
        try:
            exif_data = {}
            exif = img._getexif()

            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)

            return exif_data
        except:
            return {}

    def _check_datetime_consistency(self, exif_data: Dict) -> bool:
        """Check if datetime fields are consistent"""
        try:
            datetime_original = exif_data.get("DateTimeOriginal")
            datetime_digitized = exif_data.get("DateTimeDigitized")

            if datetime_original and datetime_digitized:
                # Parse dates
                fmt = "%Y:%m:%d %H:%M:%S"
                dt1 = datetime.strptime(datetime_original, fmt)
                dt2 = datetime.strptime(datetime_digitized, fmt)

                # Check if within 1 minute
                diff = abs((dt1 - dt2).total_seconds())
                return diff < 60

            return True  # If dates missing, assume consistent
        except:
            return True
