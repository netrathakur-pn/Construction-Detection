import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def detect_changes(before_path, after_path, result_path):

    # Load and grayscale
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    gray1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Blur to remove noise
    gray1 = cv2.GaussianBlur(gray1, (9,9), 0)
    gray2 = cv2.GaussianBlur(gray2, (9,9), 0)

    # Difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morph cleaning
    kernel = np.ones((7,7), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = after.copy()
    mask_bw = np.zeros(after.shape[:2], dtype=np.uint8)

    count = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 350:
            x, y, w, h = cv2.boundingRect(c)
            patch_before = gray1[y:y+h, x:x+w]
            patch_after = gray2[y:y+h, x:x+w]

            # Safety check
            if patch_before.shape == patch_after.shape:
                sim = ssim(patch_before, patch_after)

                # Only accept real changes
                if sim < 0.65:
                    count += 1
                    # Draw RED overlay
                    cv2.drawContours(overlay, [c], -1, (0,0,255), -1)
                    # Draw WHITE on black mask
                    cv2.drawContours(mask_bw, [c], -1, 255, -1)

    # Save overlay
    result_overlay = cv2.addWeighted(overlay, 0.6, after, 0.4, 0)
    cv2.imwrite(result_path, result_overlay)

    # Save BW mask (same name + _mask)
    mask_path = result_path.replace(".jpg", "_mask.png")
    cv2.imwrite(mask_path, mask_bw)

    # Metrics
    change_ratio = (np.sum(clean > 0) / clean.size) * 100
    difference_percent = np.sum(diff > 25) * 100 / diff.size
    confidence_score = (difference_percent / change_ratio) if change_ratio > 0 else 0

    metrics = {
        "detected_changes": count,
        "change_ratio_percent": round(change_ratio, 2),
        "difference_percent": round(difference_percent, 2),
        "confidence_score": round(confidence_score, 2),
        "mask_path": mask_path
    }

    return result_overlay, metrics
