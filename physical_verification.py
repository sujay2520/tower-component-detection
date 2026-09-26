"""
physical_verification.py — 2-Step Tower Structure Physical Verification
=========================================================================
Uses computer vision (OpenCV + NumPy) to analyze the PHYSICAL structure of
detected towers inside YOLO bounding boxes:

Step 1 (Porosity / Sky-Gap Analysis):
    Supporting (lattice) towers have open truss frameworks — sky/background
    is visible THROUGH the tower structure. Monopole towers are solid tubes
    with no internal gaps.

    We measure the "void ratio" = fraction of sky-colored pixels inside the
    convex hull of the detected tower region.

Step 2 (Horizontal Scanline Strut Frequency):
    A horizontal scanline across a lattice tower crosses multiple metal-sky
    edges (leg → sky → brace → sky → leg). A monopole tower has exactly
    2 outer edges (background → pole wall → pole wall → background).

These two physical tests act as a SECOND VERIFICATION step after YOLO detection,
boosting confidence when they agree and acting as a tiebreaker when YOLO is
ambiguous (55-65% confidence).
"""

import numpy as np
import cv2


class PhysicalVerifier:
    """
    Analyzes the physical structure of a tower crop to determine if it's
    a lattice (supporting) or solid (monopole) tower based on porosity
    and internal edge patterns.
    """

    # Thresholds tuned for telecommunication tower classification
    POROSITY_LATTICE_THRESHOLD = 0.18   # > 18% void → likely lattice
    POROSITY_MONOPOLE_THRESHOLD = 0.08  # < 8% void → likely monopole
    SCANLINE_LATTICE_MIN_CROSSINGS = 6  # ≥ 6 distinct edge transitions → lattice cross-bracing
    SCANLINE_MONOPOLE_MAX_CROSSINGS = 4 # ≤ 4 transitions → solid tube (2 outer edges + noise)

    def __init__(self):
        pass

    @staticmethod
    def _extract_tower_crop(image, box, margin_ratio=0.05):
        """
        Extract the tower region from the image using the YOLO bounding box.
        Focuses on the middle 60% of the box height (excluding antenna top
        and base/ground which add noise).
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        box_h = y2 - y1
        box_w = x2 - x1

        if box_h < 20 or box_w < 10:
            return None

        # Focus on the middle 60% vertically (skip antennas at top, base at bottom)
        top_skip = int(box_h * 0.20)
        bottom_skip = int(box_h * 0.20)
        crop_y1 = y1 + top_skip
        crop_y2 = y2 - bottom_skip

        if crop_y2 - crop_y1 < 15:
            crop_y1 = y1
            crop_y2 = y2

        crop = image[crop_y1:crop_y2, x1:x2]
        return crop

    @staticmethod
    def _compute_porosity(crop):
        """
        Compute the porosity (void/sky-gap ratio) of the tower crop.

        Method:
        1. Convert to HSV color space
        2. Detect sky/background pixels using:
           - High brightness (V > 150) AND low saturation (S < 80) for clear sky
           - Blue-ish hue (H 90-130) with moderate brightness for blue sky
           - Very high brightness (V > 200) for overcast white sky
        3. Apply morphological operations to clean up noise
        4. Compute the convex hull of the foreground (tower structure)
        5. Void ratio = sky pixels inside hull / total hull area

        Returns:
            porosity (float): 0.0 to 1.0, fraction of sky visible through tower
            sky_mask (numpy array): binary mask of detected sky pixels
            debug_info (dict): intermediate values for debugging
        """
        if crop is None or crop.size == 0:
            return 0.0, None, {"error": "empty crop"}

        h, w = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Sky detection using multiple criteria (union of conditions)
        # Condition 1: Bright + low saturation (white/gray sky, overcast)
        bright_low_sat = (hsv[:, :, 2] > 160) & (hsv[:, :, 1] < 70)

        # Condition 2: Blue sky (hue 90-130 in OpenCV HSV where H is 0-180)
        blue_sky = (
            (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 135) &
            (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 100)
        )

        # Condition 3: Very bright overcast/white (any hue)
        very_bright = hsv[:, :, 2] > 210

        # Combine sky masks
        sky_mask = (bright_low_sat | blue_sky | very_bright).astype(np.uint8) * 255

        # Morphological cleanup: remove tiny noise, fill small holes
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel_small)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel_med)

        # Detect foreground (tower metal) as inverse of sky
        foreground = 255 - sky_mask

        # Find contours of the foreground to build convex hull
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # All sky? Unlikely to be a tower
            total_pixels = h * w
            sky_pixels = np.count_nonzero(sky_mask)
            porosity = sky_pixels / total_pixels if total_pixels > 0 else 0
            return porosity, sky_mask, {"method": "full_frame", "sky_pct": porosity}

        # Merge all contour points and compute convex hull
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)

        # Sky pixels INSIDE the hull = gaps in the tower structure
        sky_inside_hull = cv2.bitwise_and(sky_mask, hull_mask)
        hull_area = np.count_nonzero(hull_mask)
        sky_inside_area = np.count_nonzero(sky_inside_hull)

        porosity = sky_inside_area / hull_area if hull_area > 0 else 0.0

        debug_info = {
            "hull_area": int(hull_area),
            "sky_inside_hull": int(sky_inside_area),
            "total_sky": int(np.count_nonzero(sky_mask)),
            "porosity": round(porosity, 4),
            "crop_size": f"{w}x{h}"
        }

        return porosity, sky_mask, debug_info

    @staticmethod
    def _scanline_analysis(crop, num_scanlines=7):
        """
        Perform horizontal scanline analysis to count metal-sky edge transitions.

        Lattice towers: Multiple boundary crossings per scanline (leg→sky→brace→sky→leg)
        Monopole towers: Typically 2 boundary crossings (background→pole→background)

        We take `num_scanlines` horizontal slices at evenly spaced vertical
        positions through the crop and count DISTINCT transition groups
        (foreground↔background boundaries), not individual gradient pixels.

        Returns:
            avg_crossings (float): average edge boundary crossings per scanline
            crossing_details (list): per-scanline crossing counts
        """
        if crop is None or crop.size == 0:
            return 0, []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if w < 15 or h < 15:
            return 0, []

        crossing_counts = []

        for i in range(num_scanlines):
            # Evenly spaced scanline positions (avoid exact edges)
            y_pos = int(h * (i + 1) / (num_scanlines + 1))
            y_pos = max(1, min(y_pos, h - 2))

            scanline = gray[y_pos, :].astype(np.float32)

            # Smooth to remove pixel noise
            if len(scanline) > 7:
                kernel = np.ones(5) / 5.0
                scanline = np.convolve(scanline, kernel, mode='same')

            # Binarize: foreground (dark/metal) vs background (bright/sky)
            # Use Otsu-like adaptive threshold on the scanline
            scan_mean = np.mean(scanline)
            binary = (scanline > scan_mean).astype(np.int32)

            # Count transitions (0→1 or 1→0) = distinct edge boundaries
            transitions = np.count_nonzero(np.diff(binary))

            crossing_counts.append(transitions)

        avg_crossings = float(np.mean(crossing_counts)) if crossing_counts else 0
        return avg_crossings, crossing_counts

    def verify(self, image, box, yolo_class=None, yolo_confidence=0.0):
        """
        Perform 2-step physical verification on a detected tower region.

        Args:
            image: Full image (numpy array, BGR)
            box: YOLO bounding box [x1, y1, x2, y2]
            yolo_class: YOLO's predicted class ('supporting_tower' or 'monopole_tower')
            yolo_confidence: Raw YOLO detection confidence (0.0-1.0)

        Returns:
            dict with:
                - physical_class: class determined by physical analysis
                - porosity: void ratio (0.0 to 1.0)
                - avg_crossings: average horizontal edge crossings
                - agreement: True if physical analysis agrees with YOLO
                - final_class: definitive class after dual verification
                - final_confidence: boosted or adjusted confidence
                - verification_details: human-readable explanation
        """
        crop = self._extract_tower_crop(image, box)
        if crop is None:
            return {
                "physical_class": yolo_class,
                "porosity": -1,
                "avg_crossings": -1,
                "agreement": True,
                "final_class": yolo_class,
                "final_confidence": yolo_confidence,
                "verification_details": "Crop too small for physical verification; using YOLO result."
            }

        # Step 1: Porosity / Sky-Gap Analysis
        porosity, sky_mask, porosity_debug = self._compute_porosity(crop)

        # Step 2: Horizontal Scanline Strut Frequency
        avg_crossings, crossing_details = self._scanline_analysis(crop)

        # Physical classification decision
        porosity_vote = None
        scanline_vote = None
        reasons = []

        # Porosity vote
        if porosity > self.POROSITY_LATTICE_THRESHOLD:
            porosity_vote = "supporting_tower"
            reasons.append(f"Sky-gap porosity {porosity*100:.1f}% > {self.POROSITY_LATTICE_THRESHOLD*100}% → open lattice framework detected")
        elif porosity < self.POROSITY_MONOPOLE_THRESHOLD:
            porosity_vote = "monopole_tower"
            reasons.append(f"Sky-gap porosity {porosity*100:.1f}% < {self.POROSITY_MONOPOLE_THRESHOLD*100}% → solid tubular structure detected")
        else:
            reasons.append(f"Sky-gap porosity {porosity*100:.1f}% is in ambiguous zone ({self.POROSITY_MONOPOLE_THRESHOLD*100}-{self.POROSITY_LATTICE_THRESHOLD*100}%)")

        # Scanline vote
        if avg_crossings >= self.SCANLINE_LATTICE_MIN_CROSSINGS:
            scanline_vote = "supporting_tower"
            reasons.append(f"Avg {avg_crossings:.1f} edge crossings/scanline ≥ {self.SCANLINE_LATTICE_MIN_CROSSINGS} → cross-bracing struts detected")
        elif avg_crossings <= self.SCANLINE_MONOPOLE_MAX_CROSSINGS:
            scanline_vote = "monopole_tower"
            reasons.append(f"Avg {avg_crossings:.1f} edge crossings/scanline ≤ {self.SCANLINE_MONOPOLE_MAX_CROSSINGS} → smooth solid column")
        else:
            reasons.append(f"Avg {avg_crossings:.1f} crossings/scanline — inconclusive edge pattern")

        # Determine physical class by combining votes
        votes = [v for v in [porosity_vote, scanline_vote] if v is not None]
        if len(votes) == 2 and votes[0] == votes[1]:
            physical_class = votes[0]
            physical_strength = "strong"
        elif len(votes) == 1:
            physical_class = votes[0]
            physical_strength = "moderate"
        elif porosity_vote is not None:
            # Porosity is more reliable than scanline
            physical_class = porosity_vote
            physical_strength = "moderate"
        else:
            # No clear physical signal — defer to YOLO
            physical_class = yolo_class
            physical_strength = "weak"

        # Dual Verification: Compare YOLO vs Physical
        agreement = (physical_class == yolo_class) or yolo_class is None

        # Confidence Fusion
        if agreement and physical_strength == "strong":
            # Both YOLO and physical analysis agree strongly
            # Boost confidence significantly (cap at 0.99)
            boost = 0.12 if yolo_confidence < 0.85 else 0.06
            final_confidence = min(yolo_confidence + boost, 0.99)
            final_class = yolo_class or physical_class
            reasons.append(f"✅ DUAL VERIFICATION PASSED — YOLO + Physical analysis both confirm {final_class.replace('_', ' ').title()}")
        elif agreement and physical_strength == "moderate":
            # Partial agreement
            boost = 0.06
            final_confidence = min(yolo_confidence + boost, 0.97)
            final_class = yolo_class or physical_class
            reasons.append(f"✅ Verification agrees (moderate) — consistent with {final_class.replace('_', ' ').title()}")
        elif not agreement and physical_strength == "strong" and yolo_confidence < 0.65:
            # YOLO is uncertain AND physical strongly disagrees → override YOLO
            final_class = physical_class
            final_confidence = max(yolo_confidence, 0.75)
            reasons.append(f"⚠️ YOLO uncertain ({yolo_confidence*100:.0f}%), physical verification OVERRIDES → {final_class.replace('_', ' ').title()}")
        elif not agreement and physical_strength == "strong" and yolo_confidence >= 0.65:
            # YOLO is fairly confident but physical disagrees → flag conflict, keep YOLO but don't boost
            final_class = yolo_class
            final_confidence = yolo_confidence  # no boost
            reasons.append(f"⚠️ Physical verification DISAGREES (physical says {physical_class}), but YOLO confidence is high ({yolo_confidence*100:.0f}%) — keeping YOLO prediction")
        else:
            # Weak physical signal — just use YOLO as-is
            final_class = yolo_class or physical_class
            final_confidence = yolo_confidence
            reasons.append(f"Physical signal too weak to modify YOLO prediction")

        return {
            "physical_class": physical_class,
            "physical_strength": physical_strength,
            "porosity": round(porosity, 4),
            "porosity_debug": porosity_debug,
            "avg_crossings": round(avg_crossings, 2),
            "crossing_details": crossing_details,
            "agreement": agreement,
            "final_class": final_class,
            "final_confidence": round(final_confidence, 4),
            "verification_details": " | ".join(reasons)
        }

    def verify_full_image(self, image, detections):
        """
        Run physical verification on all detections in an image.

        Args:
            image: Full image (numpy array, BGR)
            detections: List of dicts with 'class', 'confidence', 'box' keys

        Returns:
            dict with:
                - verified_class: final classification after dual verification
                - verified_confidence: final confidence
                - verification_summary: human-readable summary
                - per_detection: list of per-box verification results
        """
        if not detections:
            return {
                "verified_class": None,
                "verified_confidence": 0.0,
                "verification_summary": "No detections to verify",
                "per_detection": []
            }

        per_detection = []
        class_scores = {}

        for det in detections:
            result = self.verify(
                image,
                det["box"],
                yolo_class=det["class"],
                yolo_confidence=det["confidence"]
            )
            per_detection.append(result)

            fc = result["final_class"]
            if fc not in class_scores:
                class_scores[fc] = []
            class_scores[fc].append(result["final_confidence"])

        # Pick the class with highest average verified confidence
        best_class = max(class_scores, key=lambda c: max(class_scores[c]))
        best_conf = max(class_scores[best_class])

        # Count agreements
        agreements = sum(1 for r in per_detection if r["agreement"])
        total = len(per_detection)

        summary_parts = [
            f"Verified {total} detection(s): {agreements}/{total} YOLO-Physical agreements",
            f"Final: {best_class.replace('_', ' ').title()} @ {best_conf*100:.1f}%"
        ]

        # Add porosity summary
        porosities = [r["porosity"] for r in per_detection if r["porosity"] >= 0]
        if porosities:
            avg_por = np.mean(porosities)
            summary_parts.append(f"Avg porosity: {avg_por*100:.1f}%")

        # Add crossings summary
        crossings = [r["avg_crossings"] for r in per_detection if r["avg_crossings"] >= 0]
        if crossings:
            avg_cross = np.mean(crossings)
            summary_parts.append(f"Avg scanline crossings: {avg_cross:.1f}")

        return {
            "verified_class": best_class,
            "verified_confidence": round(best_conf, 4),
            "verification_summary": " | ".join(summary_parts),
            "per_detection": per_detection
        }


# Quick self-test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python physical_verification.py <image_path> [x1 y1 x2 y2]")
        print("  If no box given, uses full image as bounding box.")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Cannot read: {sys.argv[1]}")
        sys.exit(1)

    h, w = img.shape[:2]
    if len(sys.argv) >= 6:
        box = [float(sys.argv[i]) for i in range(2, 6)]
    else:
        box = [0, 0, w, h]

    verifier = PhysicalVerifier()
    result = verifier.verify(img, box, yolo_class="supporting_tower", yolo_confidence=0.60)

    print("\n" + "=" * 60)
    print("  PHYSICAL VERIFICATION RESULT")
    print("=" * 60)
    print(f"  Physical Class    : {result['physical_class']}")
    print(f"  Physical Strength : {result['physical_strength']}")
    print(f"  Porosity (void %) : {result['porosity']*100:.1f}%")
    print(f"  Avg Crossings     : {result['avg_crossings']:.1f}")
    print(f"  YOLO Agreement    : {'✅ YES' if result['agreement'] else '⚠️ NO'}")
    print(f"  Final Class       : {result['final_class']}")
    print(f"  Final Confidence  : {result['final_confidence']*100:.1f}%")
    print(f"  Details           : {result['verification_details']}")
    print("=" * 60)
