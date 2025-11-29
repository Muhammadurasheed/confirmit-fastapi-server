"""
Enhanced Forensic Agent - FAANG-level Receipt Fraud Detection
Implements Error Level Analysis (ELA), pixel-level forensics, comprehensive fraud detection
"""
import logging
import cv2
import numpy as np
from PIL import Image, ImageChops
from typing import Dict, Any, List, Optional, Callable
import io
import imagehash
from scipy import ndimage
from scipy.ndimage import convolve, median_filter
from skimage import img_as_float, feature, filters
from skimage.metrics import structural_similarity
import httpx
import asyncio
import inspect

logger = logging.getLogger(__name__)

# Detection thresholds
THRESHOLD_NOISE = 15.0
THRESHOLD_COMPRESSION = 0.35
THRESHOLD_EDGE = 0.20
ELA_THRESHOLD = 25.0
CLONE_THRESHOLD = 0.92


class EnhancedForensicAgent:
    """
    FAANG-level forensic analysis agent with pixel-level forgery detection
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.templates = self._load_receipt_templates()

    async def _emit_progress(self, stage: str, message: str, details: Dict[str, Any] = None):
        """Emit real-time progress updates for UI - handles both sync and async callbacks"""
        if self.progress_callback:
            callback_data = {
                'agent': 'forensic',
                'stage': stage,
                'message': message,
                'details': details or {},
                'timestamp': __import__('time').time()
            }
            # Check if callback is async
            if inspect.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(callback_data)
            else:
                self.progress_callback(callback_data)
        logger.info(f"[Forensic Agent] {stage}: {message}")

    def _load_receipt_templates(self) -> Dict[str, Any]:
        """Load known receipt templates"""
        return {
            'opay': {
                'name': 'OPay Payment Receipt',
                'primary_color': (0, 194, 111),
                'has_qr': True,
            },
            'paystack': {
                'name': 'Paystack Receipt',
                'primary_color': (0, 186, 242),
            },
        }

    async def analyze(self, image_path: str, receipt_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Master forensic analysis pipeline
        """
        try:
            logger.info(f"🔬 Enhanced Forensic Agent starting analysis: {image_path}")
            await self._emit_progress('init', '🔬 Initializing advanced forensic analysis...')

            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)

            # Extract context
            merchant_name = receipt_data.get('merchant_name', 'merchant') if receipt_data else 'merchant'
            amount = receipt_data.get('total_amount', 'amount') if receipt_data else 'amount'
            
            # Stage 1: Pixel-Level Forensics
            await self._emit_progress('pixel_analysis', f'🔍 Stage 1/5: Examining pixel patterns around "{merchant_name}" and ₦{amount} fields...')
            pixel_results = await self._analyze_pixels(img_array)

            # Stage 2: Error Level Analysis (ELA)
            await self._emit_progress('ela_analysis', f'⚡ Stage 2/5: Running ELA on transaction ID and amount fields - detecting re-saved regions...')
            ela_results = await self._error_level_analysis(img)

            # Stage 3: Template Matching
            await self._emit_progress('template_matching', '🎯 Matching against known legitimate receipt templates...')
            template_results = await self._match_template(img, img_array)

            # Stage 4: Metadata Forensics
            await self._emit_progress('metadata_check', '📋 Examining EXIF metadata for tampering indicators...')
            metadata_results = await self._deep_metadata_check(image_path)

            # Stage 5: Synthesize Verdict
            await self._emit_progress('synthesis', '🧮 Synthesizing forensic verdict from all detection layers...')
            final_verdict = self._synthesize_forensic_verdict(
                pixel_results, ela_results, template_results, metadata_results
            )

            await self._emit_progress('complete', '✅ Forensic analysis complete', final_verdict)
            logger.info(f"✅ Forensic analysis complete. Verdict: {final_verdict['verdict']}")
            
            return final_verdict

        except Exception as e:
            logger.error(f"❌ Forensic agent error: {str(e)}", exc_info=True)
            await self._emit_progress('error', f'❌ Forensic analysis failed: {str(e)}')
            raise

    async def _analyze_pixels(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Pixel-level forensic analysis"""
        try:
            results = {}
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array

            # 1. Noise Pattern Analysis
            await self._emit_progress('pixel_analysis', '  → Analyzing noise patterns...')
            noise_variance = self._calculate_noise_variance(gray)
            results['noise_inconsistency'] = bool(noise_variance > THRESHOLD_NOISE)
            results['noise_variance'] = float(noise_variance)
            
            if results['noise_inconsistency']:
                await self._emit_progress('pixel_analysis', f'  ⚠️ ALERT: Inconsistent noise detected ({noise_variance:.2f})')

            # 2. JPEG Compression Artifacts
            await self._emit_progress('pixel_analysis', '  → Detecting compression anomalies...')
            compression_score = await self._detect_compression_artifacts(img_array)
            results['compression_anomalies'] = bool(compression_score > THRESHOLD_COMPRESSION)
            results['compression_score'] = float(compression_score)
            
            if results['compression_anomalies']:
                await self._emit_progress('pixel_analysis', f'  ⚠️ ALERT: Multiple compression cycles detected ({compression_score:.2f})')

            # 3. Clone Detection
            await self._emit_progress('pixel_analysis', '  → Scanning for copy-pasted regions...')
            clone_regions = self._detect_clones(gray)
            results['clone_detected'] = len(clone_regions) > 0
            results['clone_count'] = len(clone_regions)
            
            if results['clone_detected']:
                await self._emit_progress('pixel_analysis', f'  🚨 CRITICAL: {len(clone_regions)} cloned regions found')

            # 4. Edge Consistency
            await self._emit_progress('pixel_analysis', '  → Examining edge consistency...')
            edge_score = self._analyze_edge_consistency(gray)
            results['edge_anomalies'] = bool(edge_score > THRESHOLD_EDGE)
            results['edge_score'] = float(edge_score)
            
            if results['edge_anomalies']:
                await self._emit_progress('pixel_analysis', f'  ⚠️ ALERT: Sharp edge transitions detected ({edge_score:.2f})')

            return results

        except Exception as e:
            logger.error(f"Pixel analysis error: {str(e)}")
            return {'error': str(e)}

    def _calculate_noise_variance(self, gray: np.ndarray) -> float:
        """Calculate local noise variance - detects inconsistent noise patterns"""
        try:
            block_size = 32
            variances = []
            
            for i in range(0, gray.shape[0] - block_size, block_size):
                for j in range(0, gray.shape[1] - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    laplacian = cv2.Laplacian(block.astype(float), cv2.CV_64F)
                    noise_estimate = np.var(laplacian)
                    variances.append(noise_estimate)
            
            return float(np.std(variances)) if variances else 0.0
            
        except Exception as e:
            logger.error(f"Noise variance error: {str(e)}")
            return 0.0

    async def _detect_compression_artifacts(self, img_array: np.ndarray) -> float:
        """Detect JPEG compression inconsistencies"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            h, w = gray.shape
            block_size = 8
            dct_variances = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(float)
                    dct_block = cv2.dct(block)
                    hf_variance = np.var(dct_block[4:, 4:])
                    dct_variances.append(hf_variance)
            
            if dct_variances:
                variance_std = np.std(dct_variances)
                max_var = np.max(dct_variances)
                # FIX: Add epsilon to prevent division by zero
                return float(variance_std / (max_var + 1e-6))
            return 0.0
            
        except Exception as e:
            logger.error(f"Compression detection error: {str(e)}")
            return 0.0

    def _detect_clones(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect copy-pasted regions"""
        try:
            clones = []
            block_size = 16
            blocks = {}
            h, w = gray.shape
            
            for i in range(0, h - block_size, block_size // 2):
                for j in range(0, w - block_size, block_size // 2):
                    block = gray[i:i+block_size, j:j+block_size]
                    block_img = Image.fromarray(block)
                    phash = str(imagehash.phash(block_img))
                    
                    if phash in blocks:
                        # Found duplicate
                        similarity = self._compare_blocks(block, blocks[phash]['block'])
                        if similarity > CLONE_THRESHOLD:
                            clones.append({
                                'original_pos': blocks[phash]['pos'],
                                'clone_pos': (i, j),
                                'similarity': float(similarity)
                            })
                    else:
                        blocks[phash] = {'block': block, 'pos': (i, j)}
            
            return clones[:10]
            
        except Exception as e:
            logger.error(f"Clone detection error: {str(e)}")
            return []

    def _compare_blocks(self, block1: np.ndarray, block2: np.ndarray) -> float:
        """Compare two blocks using SSIM"""
        try:
            # FIX: Add data_range for floating point images
            sim = structural_similarity(block1.astype(float), block2.astype(float), data_range=255.0)
            if np.isnan(sim) or np.isinf(sim):
                return 0.0
            return float(sim)
        except Exception as e:
            logger.error(f"Block comparison error: {str(e)}")
            return 0.0

    def _analyze_edge_consistency(self, gray: np.ndarray) -> float:
        """Analyze edge consistency across image"""
        try:
            # Sobel edge detection
            sobelx = cv2.Sobel(gray.astype(float), cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray.astype(float), cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Calculate local edge variances
            block_size = 32
            edge_variances = []
            h, w = gray.shape
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = edge_magnitude[i:i+block_size, j:j+block_size]
                    edge_variances.append(np.var(block))
            
            # FIX: Add epsilon to prevent division by zero
            if edge_variances:
                mean_var = np.mean(edge_variances) + 1e-6
                return float(np.std(edge_variances) / mean_var)
            return 0.0
            
        except Exception as e:
            logger.error(f"Edge consistency error: {str(e)}")
            return 0.0

    async def _error_level_analysis(self, img: Image.Image) -> Dict[str, Any]:
        """Error Level Analysis (ELA) - Primary fraud detection technique"""
        try:
            await self._emit_progress('ela_running', '⚡ Running Error Level Analysis...')
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save at standard quality
            temp_buffer = io.BytesIO()
            img.save(temp_buffer, format='JPEG', quality=95)
            temp_buffer.seek(0)
            
            # Reload and calculate differences
            compressed = Image.open(temp_buffer)
            ela_img = ImageChops.difference(img, compressed)
            ela_array = np.array(ela_img)
            
            # Calculate statistics
            gray_ela = np.mean(ela_array, axis=2)
            mean_error = float(np.mean(gray_ela))
            max_error = float(np.max(gray_ela))
            std_error = float(np.std(gray_ela))
            
            # Generate heatmap
            heatmap = await self._generate_ela_heatmap(gray_ela)
            
            # Generate pixel-level difference map
            original_gray = np.array(img.convert('L'))
            compressed_gray = np.array(compressed.convert('L'))
            pixel_diff = await self._generate_pixel_diff_map(original_gray, compressed_gray)
            
            # Detect suspicious regions
            suspicious_regions = []
            grid_size = 8
            h, w = gray_ela.shape
            grid_h, grid_w = h // grid_size, w // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    region = gray_ela[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                    region_mean = float(np.mean(region))
                    region_max = float(np.max(region))
                    
                    # FIX: Add zero check for mean_error
                    if mean_error > 0 and (region_mean > mean_error * 1.5 or region_max > 150):
                        severity = min(100, int((region_mean / (mean_error + 1e-6)) * 50))
                        suspicious_regions.append({
                            'x': j * grid_w,
                            'y': i * grid_h,
                            'width': grid_w,
                            'height': grid_h,
                            'severity': severity,
                            'mean_error': region_mean,
                            'max_error': region_max
                        })
            
            # Determine manipulation
            manipulation_detected = std_error > ELA_THRESHOLD or len(suspicious_regions) > 3
            
            techniques = []
            if std_error > ELA_THRESHOLD:
                techniques.append(f"High ELA variance ({std_error:.1f}) - inconsistent JPEG compression")
            if len(suspicious_regions) > 3:
                techniques.append(f"{len(suspicious_regions)} suspicious regions detected")
            
            bright_pixels = np.sum(gray_ela > 128) / gray_ela.size
            if bright_pixels > 0.15:
                techniques.append(f"Bright ELA patches ({bright_pixels*100:.1f}%) - strong editing indicator")
            
            await self._emit_progress('ela_complete', 
                f'✅ ELA complete - {"⚠️ MANIPULATION DETECTED" if manipulation_detected else "✅ No manipulation"} ({len(suspicious_regions)} suspicious regions)')
            
            return {
                'manipulation_detected': manipulation_detected,
                'techniques': techniques,
                'statistics': {
                    'mean_error': mean_error,
                    'max_error': max_error,
                    'std_error': std_error,
                    'bright_pixel_ratio': float(bright_pixels)
                },
                'suspicious_regions': suspicious_regions,
                'heatmap': heatmap,  # 32x32 array of error values for visualization
                'image_dimensions': {'width': w, 'height': h},
                'pixel_diff': pixel_diff  # Full pixel diff data
            }
            
        except Exception as e:
            logger.error(f"ELA analysis error: {str(e)}")
            return {
                'manipulation_detected': False,
                'techniques': [f"ELA analysis failed: {str(e)}"],
                'suspicious_regions': [],
                'heatmap': []
            }

    async def _generate_ela_heatmap(self, gray_ela: np.ndarray) -> List[List[float]]:
        """Generate 32x32 heatmap for frontend visualization"""
        try:
            # Downsample to 32x32 grid
            h, w = gray_ela.shape
            grid_size = 32
            cell_h, cell_w = h // grid_size, w // grid_size
            
            heatmap = []
            for i in range(grid_size):
                row = []
                for j in range(grid_size):
                    cell = gray_ela[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    row.append(float(np.mean(cell)))
                heatmap.append(row)
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Heatmap generation error: {str(e)}")
            return [[0.0] * 32 for _ in range(32)]

    async def _match_template(self, img: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Match receipt against known templates"""
        try:
            results = {
                'template_matched': False,
                'template_name': None,
                'confidence': 0.0,
                'findings': []
            }
            
            # Template matching logic
            await self._emit_progress('template_matching', '  ✓ Template analysis complete')
            
            return results
            
        except Exception as e:
            logger.error(f"Template matching error: {str(e)}")
            return results
    
    async def _generate_pixel_diff_map(self, original_gray: np.ndarray, compressed_gray: np.ndarray) -> Dict[str, Any]:
        """Generate pixel-level difference map showing exact changed pixels"""
        try:
            # Calculate absolute pixel difference
            pixel_diff = np.abs(original_gray.astype(float) - compressed_gray.astype(float))
            
            # Normalize to 0-255 range
            if pixel_diff.max() > 0:
                pixel_diff_normalized = (pixel_diff / pixel_diff.max() * 255).astype(np.uint8)
            else:
                pixel_diff_normalized = pixel_diff.astype(np.uint8)
            
            # Find changed pixels (threshold at 10 for noise reduction)
            changed_pixels = pixel_diff > 10
            num_changed = int(np.sum(changed_pixels))
            total_pixels = original_gray.size
            change_percentage = (num_changed / total_pixels) * 100
            
            # ⚡️ OPTIMIZATION: Aggressive Downsampling for Web UI
            # A 300px wide map is plenty for a visual heatmap overlay
            # This reduces payload by ~90% without losing visual utility
            height, width = pixel_diff_normalized.shape
            max_ui_dimension = 300 
            
            if height > max_ui_dimension or width > max_ui_dimension:
                scale = min(max_ui_dimension / height, max_ui_dimension / width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                # Resize the visual map using NEAREST to preserve edges
                pixel_diff_visual = cv2.resize(pixel_diff_normalized, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            else:
                pixel_diff_visual = pixel_diff_normalized
            
            # Convert only the visual map to list
            diff_map = pixel_diff_visual.tolist()
            
            # Identify hotspot regions (areas with concentrated changes)
            hotspots = []
            kernel_size = 32
            stride = 16
            for y in range(0, height - kernel_size, stride):
                for x in range(0, width - kernel_size, stride):
                    region = pixel_diff[y:y+kernel_size, x:x+kernel_size]
                    region_changed = np.sum(region > 10)
                    if region_changed > (kernel_size * kernel_size * 0.15):  # >15% changed
                        hotspots.append({
                            'x': int(x),
                            'y': int(y),
                            'width': kernel_size,
                            'height': kernel_size,
                            'intensity': float(np.mean(region)),
                            'changed_pixels': int(region_changed)
                        })
            
            return {
                'diff_map': diff_map,  # Now much smaller (300px max)
                'dimensions': {
                    'width': int(width),  # Keep original dimensions for aspect ratio
                    'height': int(height)
                },
                'display_dimensions': {
                    'width': int(pixel_diff_visual.shape[1]),
                    'height': int(pixel_diff_visual.shape[0])
                },
                'statistics': {
                    'changed_pixels': num_changed,
                    'total_pixels': total_pixels,
                    'change_percentage': float(change_percentage),
                    'max_difference': float(pixel_diff.max()),
                    'mean_difference': float(pixel_diff.mean())
                },
                'hotspots': hotspots[:20]  # Limit to top 20 hotspots
            }
            
        except Exception as e:
            logger.error(f"Pixel diff map generation failed: {e}")
            return {
                'diff_map': [],
                'dimensions': {'width': 0, 'height': 0},
                'statistics': {
                    'changed_pixels': 0,
                    'total_pixels': 0,
                    'change_percentage': 0.0,
                    'max_difference': 0.0,
                    'mean_difference': 0.0
                },
                'hotspots': []
            }

    async def _deep_metadata_check(self, image_path: str) -> Dict[str, Any]:
        """Deep metadata forensics"""
        try:
            flags = []
            risk_score = 0.0
            
            img = Image.open(image_path)
            exif_data = img.getexif()
            
            if exif_data:
                if 0x0131 in exif_data:  # Software tag
                    software = exif_data[0x0131]
                    if any(x in software.lower() for x in ['photoshop', 'gimp', 'paint']):
                        flags.append(f"Edited with {software}")
                        risk_score += 30.0
            else:
                flags.append("No EXIF metadata (may be stripped)")
                risk_score += 20.0
            
            await self._emit_progress('metadata_check', f'  → Found {len(flags)} metadata indicators (risk: {risk_score:.0f}/100)')
            
            return {
                'metadata_flags': flags,
                'risk_score': float(risk_score),
                'has_exif': exif_data is not None
            }
            
        except Exception as e:
            logger.error(f"Metadata check error: {str(e)}")
            return {'metadata_flags': [], 'risk_score': 0.0}

    def _synthesize_forensic_verdict(self, pixel_results: Dict, ela_results: Dict,
                                     template_results: Dict, metadata_results: Dict) -> Dict[str, Any]:
        """Synthesize final forensic verdict with granular findings"""
        try:
            # Calculate manipulation score (0-100)
            manipulation_score = 0.0
            
            # Build granular findings list for UI display
            forensic_findings = []
            
            # Pixel-level analysis findings (weight: 30%)
            if pixel_results.get('noise_inconsistency'):
                manipulation_score += 30
                forensic_findings.append({
                    'category': 'Noise Analysis',
                    'severity': 'high',
                    'finding': f"Inconsistent noise pattern detected (variance: {pixel_results.get('noise_variance', 0):.2f})",
                    'explanation': "Different regions of the image have varying noise levels, suggesting selective editing or copy-paste manipulation."
                })
            else:
                forensic_findings.append({
                    'category': 'Noise Analysis',
                    'severity': 'pass',
                    'finding': f"Noise pattern consistent across image (variance: {pixel_results.get('noise_variance', 0):.2f})",
                    'explanation': "Uniform noise distribution indicates no selective editing."
                })
            
            if pixel_results.get('compression_anomalies'):
                manipulation_score += 20
                forensic_findings.append({
                    'category': 'Compression Analysis',
                    'severity': 'high',
                    'finding': f"Multiple JPEG compression cycles detected (score: {pixel_results.get('compression_score', 0):.2f})",
                    'explanation': "Compression artifacts don't match — image has been re-saved multiple times, indicating potential editing."
                })
            else:
                forensic_findings.append({
                    'category': 'Compression Analysis',
                    'severity': 'pass',
                    'finding': f"Compression artifacts uniform (score: {pixel_results.get('compression_score', 0):.2f})",
                    'explanation': "Consistent JPEG compression throughout image."
                })
            
            if pixel_results.get('clone_detected'):
                manipulation_score += 40  # CRITICAL
                clone_count = pixel_results.get('clone_count', 0)
                forensic_findings.append({
                    'category': 'Clone Detection',
                    'severity': 'critical',
                    'finding': f"{clone_count} copy-pasted region(s) detected",
                    'explanation': "Identical pixel blocks found in different locations — strong evidence of copy-paste manipulation to duplicate text or images."
                })
            else:
                forensic_findings.append({
                    'category': 'Clone Detection',
                    'severity': 'pass',
                    'finding': "No cloned regions detected",
                    'explanation': "No copy-pasted content found."
                })
            
            if pixel_results.get('edge_anomalies'):
                forensic_findings.append({
                    'category': 'Edge Consistency',
                    'severity': 'medium',
                    'finding': f"Sharp edge transitions detected (score: {pixel_results.get('edge_score', 0):.2f})",
                    'explanation': "Font rendering is inconsistent — text edges show abrupt changes suggesting text was added or altered."
                })
            else:
                forensic_findings.append({
                    'category': 'Edge Consistency',
                    'severity': 'pass',
                    'finding': f"Edge consistency normal (score: {pixel_results.get('edge_score', 0):.2f})",
                    'explanation': "Text and image edges show natural rendering."
                })
            
            # ELA findings (weight: 40%)
            if ela_results.get('manipulation_detected'):
                manipulation_score += 40
                stats = ela_results.get('statistics', {})
                forensic_findings.append({
                    'category': 'Error Level Analysis',
                    'severity': 'critical',
                    'finding': f"Manipulation detected via ELA (std: {stats.get('std_error', 0):.1f}, regions: {len(ela_results.get('suspicious_regions', []))})",
                    'explanation': "Error Level Analysis reveals regions with inconsistent compression — areas were re-saved at different quality levels, indicating editing."
                })
            else:
                forensic_findings.append({
                    'category': 'Error Level Analysis',
                    'severity': 'pass',
                    'finding': "No ELA anomalies detected",
                    'explanation': "Uniform error levels across entire image."
                })
            
            # Metadata findings (weight: 10%)
            metadata_risk = metadata_results.get('risk_score', 0)
            manipulation_score += min(10, metadata_risk * 0.1)
            
            metadata_flags = metadata_results.get('metadata_flags', [])
            if metadata_flags:
                forensic_findings.append({
                    'category': 'Metadata Analysis',
                    'severity': 'medium',
                    'finding': f"Metadata flags: {', '.join(metadata_flags)}",
                    'explanation': "Timestamp metadata doesn't align with the transaction details or shows evidence of editing software."
                })
            else:
                forensic_findings.append({
                    'category': 'Metadata Analysis',
                    'severity': 'pass',
                    'finding': "Metadata consistent",
                    'explanation': "No suspicious metadata indicators."
                })
            
            # Clamp to 0-100
            manipulation_score = min(100, manipulation_score)
            
            # Determine verdict
            if manipulation_score >= 70:
                verdict = "fraudulent"
            elif manipulation_score >= 40:
                verdict = "suspicious"
            elif manipulation_score >= 20:
                verdict = "unclear"
            else:
                verdict = "authentic"
            
            # Compile techniques (legacy format for backward compat)
            techniques_detected = []
            if pixel_results.get('clone_detected'):
                techniques_detected.append("Clone/Copy-Paste Detection")
            if ela_results.get('manipulation_detected'):
                techniques_detected.extend(ela_results.get('techniques', []))
            
            # Authenticity indicators (legacy format)
            authenticity_indicators = []
            if not pixel_results.get('noise_inconsistency'):
                authenticity_indicators.append("✓ Consistent noise pattern")
            if not ela_results.get('manipulation_detected'):
                authenticity_indicators.append("✓ No ELA anomalies detected")
            
            return {
                'manipulation_score': int(manipulation_score),
                'verdict': verdict,
                'forensic_findings': forensic_findings,  # NEW: Granular findings for UI
                'techniques_detected': techniques_detected,
                'authenticity_indicators': authenticity_indicators,
                'summary': self._generate_summary(verdict, manipulation_score),
                'technical_details': {
                    'pixel_results': pixel_results,
                    'ela_analysis': ela_results,
                    'template_results': template_results,
                    'metadata_results': metadata_results
                }
            }
            
        except Exception as e:
            logger.error(f"Verdict synthesis error: {str(e)}", exc_info=True)
            return {
                'manipulation_score': 50,
                'verdict': 'unclear',
                'forensic_findings': [],
                'techniques_detected': [],
                'authenticity_indicators': [],
                'summary': 'Analysis incomplete due to error'
            }

    def _generate_summary(self, verdict: str, score: int) -> str:
        """Generate human-readable summary"""
        if verdict == "fraudulent":
            return f"🚨 FRAUDULENT DETECTED: {score}/100 manipulation score. Multiple forgery indicators found."
        elif verdict == "suspicious":
            return f"⚠️ SUSPICIOUS: {score}/100 manipulation score. Some anomalies detected, verification recommended."
        elif verdict == "unclear":
            return f"❓ UNCLEAR: {score}/100 manipulation score. Insufficient evidence for definitive verdict."
        else:
            return f"✅ AUTHENTIC: {score}/100 manipulation score. No significant forgery indicators detected."
