"""quality_filter.py — Image quality assessment (blur, exposure, resolution)."""
import cv2
import numpy as np


def check_blur(image, threshold=80.0):
    """Laplacian variance — lower = more blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, round(variance, 2)


def check_exposure(image, dark_thresh=40, bright_thresh=210):
    """Mean brightness check."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < dark_thresh:
        return "underexposed", round(mean_brightness, 2)
    elif mean_brightness > bright_thresh:
        return "overexposed", round(mean_brightness, 2)
    return "ok", round(mean_brightness, 2)


def check_resolution(image, min_dim=100):
    """Reject very small images."""
    h, w = image.shape[:2]
    if min(h, w) < min_dim:
        return False, f"{w}x{h}"
    return True, f"{w}x{h}"


def assess_image_quality(image):
    """
    Returns (is_acceptable: bool, reason: str, details: dict)
    Call this before running detection.
    """
    details = {}
    
    res_ok, res_info = check_resolution(image)
    details["resolution"] = res_info
    if not res_ok:
        return False, "too_small", details
    
    is_blurry, blur_score = check_blur(image)
    details["blur_score"] = blur_score
    if is_blurry:
        return False, f"blurred (laplacian variance={blur_score}, threshold=80)", details
    
    exposure_status, brightness = check_exposure(image)
    details["brightness"] = brightness
    details["exposure"] = exposure_status
    if exposure_status != "ok":
        return False, f"{exposure_status} (mean brightness={brightness})", details
    
    return True, "acceptable", details
