"""
structural_analyzer.py — Python-based Tower Detection using Computer Vision
===========================================================================
Complementary to YOLO: uses pure OpenCV techniques to detect and classify
tower structures WITHOUT a neural network.

Methods used:
  1. Canny Edge Detection — find structural edges
  2. Hough Line Transform — detect dominant vertical lines (tower = tall vertical)
  3. Contour Analysis — find the main structural shape
  4. Texture Variance — lattice towers have high texture (many edges), monopoles are smooth
  5. Vertical Line Density — lattice has many criss-crossing lines, monopole has few straight ones
  6. Color Analysis — sky ratio, metal/concrete detection

Returns: tower type classification + confidence + analysis details
"""

import cv2
import numpy as np
from collections import Counter


class StructuralAnalyzer:
    """Detect and classify tower structures using pure OpenCV."""

    def __init__(self):
        # Thresholds tuned for telecom tower images
        self.TEXTURE_LATTICE_THRESHOLD = 35.0   # Laplacian std > this = complex structure
        self.VERTICAL_LINE_MIN = 5              # Min vertical lines to confirm tower presence
        self.LATTICE_LINE_RATIO = 0.4           # If >40% of lines are non-vertical = lattice (diagonals)
        self.SKY_RATIO_LATTICE = 0.15           # >15% sky within tower region = lattice (see-through)

    def analyze(self, image):
        """
        Analyze an image and classify the tower type.

        Returns dict with:
          - tower_detected: bool
          - tower_type: 'supporting_tower' | 'monopole_tower' | 'unknown'
          - confidence: float 0-1
          - method: str description
          - details: dict with all analysis metrics
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize for consistent analysis (max 800px)
        scale = min(800 / max(h, w), 1.0)
        if scale < 1.0:
            gray_r = cv2.resize(gray, None, fx=scale, fy=scale)
            img_r = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            gray_r = gray
            img_r = image

        rh, rw = gray_r.shape[:2]

        # --- 1. Edge Detection ---
        edges = cv2.Canny(gray_r, 50, 150, apertureSize=3)
        edge_density = float(np.count_nonzero(edges)) / (rh * rw)

        # --- 2. Hough Line Transform ---
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=rh * 0.1, maxLineGap=20
        )

        vertical_lines = 0
        horizontal_lines = 0
        diagonal_lines = 0
        total_lines = 0
        vertical_line_x_positions = []

        if lines is not None:
            total_lines = len(lines)
            for line in lines:
                l = line[0] if len(line.shape) > 1 and line.shape[0] == 1 else line
                if len(l) == 4:
                    x1, y1, x2, y2 = l
                else:
                    continue
                if abs(x2 - x1) < 1:
                    angle = 90.0
                else:
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

                if 75 <= angle <= 105:
                    vertical_lines += 1
                    vertical_line_x_positions.append((x1 + x2) / 2)
                elif angle <= 15 or angle >= 165:
                    horizontal_lines += 1
                else:
                    diagonal_lines += 1

        # --- 3. Texture Analysis (Laplacian variance) ---
        # Focus on center region where tower likely is
        cx_start, cx_end = rw // 4, 3 * rw // 4
        cy_start, cy_end = rh // 6, 5 * rh // 6
        center_crop = gray_r[cy_start:cy_end, cx_start:cx_end]

        laplacian = cv2.Laplacian(center_crop, cv2.CV_64F)
        texture_std = float(np.std(laplacian))
        texture_var = float(np.var(laplacian))

        # --- 4. Contour Analysis ---
        _, thresh = cv2.threshold(gray_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the tallest contour (likely the tower)
        tallest_contour = None
        max_height = 0
        tower_aspect_ratio = 0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch > max_height and ch > rh * 0.2:  # At least 20% of image height
                max_height = ch
                tallest_contour = cnt
                tower_aspect_ratio = ch / max(cw, 1)

        # --- 5. Sky/Background Analysis ---
        # Convert to HSV and detect sky pixels (blue-ish, high value)
        hsv = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)
        # Sky: hue 90-130, sat 30-255, val 120-255
        sky_mask = cv2.inRange(hsv, (90, 30, 120), (130, 255, 255))
        # Also white sky (overcast): low sat, high val
        white_sky = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        sky_mask = cv2.bitwise_or(sky_mask, white_sky)

        # Sky ratio in center region (where tower is)
        center_sky = sky_mask[cy_start:cy_end, cx_start:cx_end]
        sky_ratio = float(np.count_nonzero(center_sky)) / max(center_sky.size, 1)

        # --- 6. Vertical Line Clustering ---
        # Lattice towers have many vertical lines spread across width
        # Monopoles have few vertical lines clustered in center
        vertical_spread = 0
        if len(vertical_line_x_positions) >= 2:
            vx = np.array(vertical_line_x_positions)
            vertical_spread = float(np.std(vx)) / rw  # Normalized spread

        # --- CLASSIFICATION LOGIC ---
        scores = {"supporting_tower": 0.0, "monopole_tower": 0.0}

        # Texture: lattice = complex (high variance), monopole = smooth (low variance)
        if texture_std > self.TEXTURE_LATTICE_THRESHOLD:
            scores["supporting_tower"] += 0.25
        else:
            scores["monopole_tower"] += 0.25

        # Diagonal lines: lattice HAS to have diagonal cross-bracing
        diag_ratio = 0.0
        if total_lines > 0:
            diag_ratio = diagonal_lines / total_lines
            if diag_ratio > self.LATTICE_LINE_RATIO and diagonal_lines >= 8:
                scores["supporting_tower"] += 0.35
            elif diag_ratio < 0.05:
                # No diagonal cross-bracing at all -> strong monopole indicator
                scores["monopole_tower"] += 0.35
            else:
                scores["monopole_tower"] += 0.15

        # Vertical line spread: lattice = spread out, monopole = clustered
        if vertical_spread > 0.15:
            scores["supporting_tower"] += 0.15
        elif vertical_spread < 0.08 and vertical_lines >= 2:
            scores["monopole_tower"] += 0.15

        # Sky ratio in tower region: lattice = see-through (high sky), monopole = solid (low sky)
        if sky_ratio > self.SKY_RATIO_LATTICE:
            scores["supporting_tower"] += 0.20
        else:
            scores["monopole_tower"] += 0.15

        # Aspect ratio: monopole = very tall and narrow, lattice = shorter and wider base
        if tower_aspect_ratio > 5.0:
            scores["monopole_tower"] += 0.15
        elif 1.5 < tower_aspect_ratio <= 5.0:
            scores["supporting_tower"] += 0.10

        # Edge density: lattice = dense edges, monopole = sparse
        if edge_density > 0.15:
            scores["supporting_tower"] += 0.10
        elif edge_density < 0.08:
            scores["monopole_tower"] += 0.10

        # CRITICAL STRUCTURAL RULE: A supporting / lattice tower MUST have diagonal cross-braces
        # If there are no diagonal lines (or < 5%), it cannot be a lattice truss framework.
        if diagonal_lines < 5 or diag_ratio < 0.05:
            scores["supporting_tower"] = min(scores["supporting_tower"], 0.15)
            scores["monopole_tower"] += 0.30

        # Determine winner
        tower_detected = total_lines >= 3 or (tallest_contour is not None and max_height > rh * 0.3)

        if scores["supporting_tower"] > scores["monopole_tower"]:
            tower_type = "supporting_tower"
            confidence = min(0.95, scores["supporting_tower"] / max(scores["supporting_tower"] + scores["monopole_tower"], 0.01))
        elif scores["monopole_tower"] > scores["supporting_tower"]:
            tower_type = "monopole_tower"
            confidence = min(0.95, scores["monopole_tower"] / max(scores["supporting_tower"] + scores["monopole_tower"], 0.01))
        else:
            tower_type = "unknown"
            confidence = 0.3

        return {
            "tower_detected": tower_detected,
            "tower_type": tower_type,
            "confidence": round(confidence, 4),
            "method": "Structural Analysis (Edge + Hough + Texture + Sky)",
            "details": {
                "edge_density": round(edge_density, 4),
                "total_lines": total_lines,
                "vertical_lines": vertical_lines,
                "horizontal_lines": horizontal_lines,
                "diagonal_lines": diagonal_lines,
                "diagonal_ratio": round(diagonal_lines / max(total_lines, 1), 3),
                "texture_std": round(texture_std, 2),
                "sky_ratio": round(sky_ratio, 4),
                "tower_aspect_ratio": round(tower_aspect_ratio, 2),
                "vertical_spread": round(vertical_spread, 4),
                "scores": {k: round(v, 3) for k, v in scores.items()},
            }
        }
