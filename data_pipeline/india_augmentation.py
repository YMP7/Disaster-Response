"""India-Specific Domain Augmentations for Disaster Computer Vision.
Models regional terrain, monsoon turbidity, tropical atmospheric haze,
and Indian architectural typologies (RCC slabs, tin sheets, terracotta tiles).
"""

import numpy as np
import cv2
from typing import Tuple


class IndiaDomainAugmentor:
    """Applies realistic visual domain shifts encountered during Indian disasters."""

    @staticmethod
    def add_monsoon_rain_streaks(
        image: np.ndarray, 
        slant: int = -5, 
        drop_length: int = 25, 
        drop_width: int = 1, 
        density: int = 350
    ) -> np.ndarray:
        """Simulates heavy monsoon downpours with directional motion blur."""
        h, w, c = image.shape
        rain_layer = np.zeros((h, w, c), dtype=np.uint8)
        
        for _ in range(density):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            x_end = int(np.clip(x + slant, 0, w - 1))
            y_end = int(np.clip(y + drop_length, 0, h - 1))
            cv2.line(rain_layer, (x, y), (x_end, y_end), (200, 210, 220), drop_width)

        rain_blurred = cv2.blur(rain_layer, (3, 3))
        augmented = cv2.addWeighted(image, 0.85, rain_blurred, 0.35, 0)
        return augmented

    @staticmethod
    def simulate_monsoon_turbidity(image: np.ndarray, turbidity_factor: float = 0.4) -> np.ndarray:
        """Simulates high-turbidity, silt-heavy brownish monsoon floodwater
        typical of Indian riverine floods (Brahmaputra, Godavari, Mahanadi).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        # Shift hues towards yellowish-brown silt (Hue 15-25)
        hsv[..., 0] = np.where((hsv[..., 0] > 80) & (hsv[..., 0] < 140), 20.0, hsv[..., 0])
        # Lower saturation slightly due to suspended particulate matter
        hsv[..., 1] = hsv[..., 1] * (1.0 - (turbidity_factor * 0.3))
        hsv[..., 2] = np.clip(hsv[..., 2] * 0.9, 0, 255)
        
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb

    @staticmethod
    def add_tropical_atmospheric_haze(image: np.ndarray, haze_intensity: float = 0.3) -> np.ndarray:
        """Simulates post-monsoon or pre-cyclone humid haze and particulate scattering."""
        h, w, c = image.shape
        airlight = np.full((h, w, c), 220, dtype=np.uint8)
        augmented = cv2.addWeighted(image, 1.0 - haze_intensity, airlight, haze_intensity, 0)
        return augmented

    @staticmethod
    def apply_tin_roof_specular_glare(image: np.ndarray, num_glare_spots: int = 3) -> np.ndarray:
        """Simulates intense solar specular reflections off corrugated tin/GI sheets."""
        result = image.copy()
        h, w, _ = image.shape
        for _ in range(num_glare_spots):
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(15, 45)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            for c in range(3):
                result[..., c] = np.clip(
                    result[..., c].astype(np.int32) + (mask.astype(np.int32) * 0.7), 
                    0, 255
                ).astype(np.uint8)
        return result

    @classmethod
    def augment_for_india(
        cls, 
        image: np.ndarray, 
        is_monsoon: bool = True, 
        is_coastal_cyclone: bool = False
    ) -> np.ndarray:
        """Master augmentation pipeline tailored for Indian disaster deployment."""
        out = image.copy()
        if is_monsoon:
            out = cls.simulate_monsoon_turbidity(out, turbidity_factor=0.35)
            out = cls.add_monsoon_rain_streaks(out, density=250)
        if is_coastal_cyclone:
            out = cls.add_tropical_atmospheric_haze(out, haze_intensity=0.25)
            out = cls.apply_tin_roof_specular_glare(out, num_glare_spots=2)
        return out
