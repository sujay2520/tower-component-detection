"""quality_filter.py — Robust image quality assessment (blur, exposure, resolution).

Automatically rejects:
  - Blurry images (Laplacian variance < threshold)
  - Underexposed images (too dark)
  - Overexposed images (too bright / washed out)
  - Too-small images (< 80px in any dimension)

Used in backend/app.py before running YOLO detection.
"""
import cv2
import numpy as np


def check_blur(image, threshold=50.0):
    """
    Laplacian variance — lower = more blurry.
    Threshold 50: rejects significantly blurred images while allowing
    slightly soft outdoor photos (common with telecom tower photography).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, round(variance, 2)


def check_exposure(image, dark_thresh=30, bright_thresh=235):
    """
    Multi-method exposure check:
    1. Mean brightness — catches obvious under/overexposure
    2. Histogram clipping — catches washed-out images where >40% of pixels
       are at extreme ends (0-10 or 245-255)
    3. Standard deviation — catches flat/low-contrast images
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    total_pixels = gray.size

    # Check 1: Mean brightness
    if mean_brightness < dark_thresh:
        return "underexposed", round(mean_brightness, 2), {
            "mean": round(mean_brightness, 2),
            "std": round(std_brightness, 2),
            "detail": f"Image too dark (mean brightness {mean_brightness:.0f} < {dark_thresh})"
        }

    if mean_brightness > bright_thresh:
        return "overexposed", round(mean_brightness, 2), {
            "mean": round(mean_brightness, 2),
            "std": round(std_brightness, 2),
            "detail": f"Image too bright (mean brightness {mean_brightness:.0f} > {bright_thresh})"
        }

    # Check 2: Histogram clipping — detect washed out images
    dark_pixels = float(np.sum(gray < 10)) / total_pixels
    bright_pixels = float(np.sum(gray > 245)) / total_pixels

    if dark_pixels > 0.40:
        return "underexposed", round(mean_brightness, 2), {
            "mean": round(mean_brightness, 2),
            "dark_clip": round(dark_pixels * 100, 1),
            "detail": f"Image heavily underexposed ({dark_pixels*100:.0f}% pixels near black)"
        }

    if bright_pixels > 0.40:
        return "overexposed", round(mean_brightness, 2), {
            "mean": round(mean_brightness, 2),
            "bright_clip": round(bright_pixels * 100, 1),
            "detail": f"Image heavily overexposed ({bright_pixels*100:.0f}% pixels near white)"
        }

    # Check 3: Very low contrast (flat image)
    if std_brightness < 15:
        return "low_contrast", round(mean_brightness, 2), {
            "mean": round(mean_brightness, 2),
            "std": round(std_brightness, 2),
            "detail": f"Image has very low contrast (std={std_brightness:.0f})"
        }

    return "ok", round(mean_brightness, 2), {
        "mean": round(mean_brightness, 2),
        "std": round(std_brightness, 2)
    }


def check_resolution(image, min_dim=80):
    """Reject very small images."""
    h, w = image.shape[:2]
    if min(h, w) < min_dim:
        return False, f"{w}x{h}"
    return True, f"{w}x{h}"


def assess_image_quality(image):
    """
    Returns (is_acceptable: bool, reason: str, details: dict)
    Call this before running detection. Automatically removes:
    - Blurry images
    - Overexposed images
    - Underexposed images
    - Low-contrast images
    - Too-small images
    """
    details = {}

    # Resolution check
    res_ok, res_info = check_resolution(image)
    details["resolution"] = res_info
    if not res_ok:
        return False, "Image resolution too low (minimum 80px required)", details

    # Blur check
    is_blurry, blur_score = check_blur(image)
    details["blur_score"] = blur_score
    details["blur_status"] = "blurry" if is_blurry else "sharp"
    if is_blurry:
        return False, f"Image rejected: Too blurry (sharpness score {blur_score:.0f}, need >=50)", details

    # Exposure check (underexposed / overexposed / low contrast)
    exposure_status, brightness, exposure_info = check_exposure(image)
    details["brightness"] = brightness
    details["exposure"] = exposure_status
    details.update(exposure_info)

    if exposure_status == "underexposed":
        return False, f"Image rejected: Underexposed — {exposure_info['detail']}", details
    elif exposure_status == "overexposed":
        return False, f"Image rejected: Overexposed — {exposure_info['detail']}", details
    elif exposure_status == "low_contrast":
        return False, f"Image rejected: Low contrast — {exposure_info['detail']}", details

    return True, "acceptable", details


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            ok, reason, details = assess_image_quality(img)
            status = "[PASSED]" if ok else "[REJECTED]"
            print(f"{status}: {reason}")
            for k, v in details.items():
                print(f"  {k}: {v}")
        else:
            print("Could not read image")
    else:
        print("Usage: python quality_filter.py <image_path>")
