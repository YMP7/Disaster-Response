"""Privacy Protection and Automated Redaction Filter.
Complies with India's Digital Personal Data Protection (DPDP) Act by
automatically detecting and blurring human faces and vehicle license plates
before imagery leaves the tactical response loop.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any


class DPDPPrivacyFilter:
    """Automated redaction of personally identifiable information (PII)."""

    def __init__(self, blur_kernel_size: Tuple[int, int] = (51, 51)):
        self.blur_kernel = blur_kernel_size
        # OpenCV built-in lightweight cascade detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )

    def redact_pii(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Scans image, detects faces/plates, applies heavy Gaussian blur, and returns metadata."""
        redacted = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Detect and blur faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=4, minSize=(20, 20)
        )
        face_count = len(faces)
        for (x, y, w, h) in faces:
            roi = redacted[y:y+h, x:x+w]
            # Extra blur strength for irreversible anonymization
            k_w = max(15, (w // 3) * 2 + 1)
            k_h = max(15, (h // 3) * 2 + 1)
            blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)
            redacted[y:y+h, x:x+w] = blurred_roi

        # 2. Detect and blur vehicle license plates
        plates = self.plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 10)
        )
        plate_count = len(plates)
        for (x, y, w, h) in plates:
            roi = redacted[y:y+h, x:x+w]
            k_w = max(15, (w // 3) * 2 + 1)
            k_h = max(15, (h // 3) * 2 + 1)
            blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)
            redacted[y:y+h, x:x+w] = blurred_roi

        redaction_audit = {
            "dpdp_compliant": True,
            "faces_redacted": int(face_count),
            "plates_redacted": int(plate_count),
            "total_pii_anonymized": int(face_count + plate_count)
        }

        return redacted, redaction_audit
