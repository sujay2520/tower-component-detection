"""quality_filter.py — Robust image quality assessment (blur, exposure, resolution)."""
import cv2
import numpy as np

def check_blur(image, threshold=60.0):
    """Laplacian variance — lower = more blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, round(variance, 2)


def check_exposure(image, dark_thresh=25, bright_thresh=240):
    """Mean brightness check."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < dark_thresh:
        return "underexposed", round(mean_brightness, 2)
    elif mean_brightness > bright_thresh:
        return "overexposed", round(mean_brightness, 2)
    return "ok", round(mean_brightness, 2)


def check_resolution(image, min_dim=80):
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
        return False, "Image resolution too low (minimum 80px required)", details
    
    is_blurry, blur_score = check_blur(image)
    details["blur_score"] = blur_score
    if is_blurry:
        return False, f"Image is blurry (Laplacian variance {blur_score} < 60 threshold)", details
    
    exposure_status, brightness = check_exposure(image)
    details["brightness"] = brightness
    details["exposure"] = exposure_status
    if exposure_status != "ok":
        return False, f"Image is {exposure_status} (mean brightness {brightness})", details
    
    return True, "acceptable", details
