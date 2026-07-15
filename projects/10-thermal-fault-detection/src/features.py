"""OpenCV-based handcrafted features describing the brightest region in a thermal image.

This is the classical computer-vision path: threshold the image, find the brightest
contour, and describe its shape. It's the kind of feature set an inspector's rule-of-thumb
("flag anything with a bright, tight, off-center spot") would look like as code.
"""
import cv2
import numpy as np
import pandas as pd

VMIN, VMAX = 15.0, 70.0
THRESHOLD_PERCENTILE = 90

FEATURE_NAMES = [
    "max_intensity",
    "mean_intensity",
    "hotspot_area",
    "hotspot_eccentricity",
    "hotspot_dist_from_center",
    "n_hotspots",
]


def _to_uint8(image, vmin=VMIN, vmax=VMAX):
    """Scale a temperature-like float image to a fixed 0-255 range.

    Uses a fixed vmin/vmax (not per-image min-max) so intensity is comparable across
    images, the way a calibrated thermal camera's output would be.
    """
    clipped = np.clip(image, vmin, vmax)
    scaled = (clipped - vmin) / (vmax - vmin) * 255.0
    return scaled.astype(np.uint8)


def extract_hotspot_features(image, threshold_percentile=THRESHOLD_PERCENTILE):
    """Extract handcrafted features from the brightest region of a single thermal image."""
    img8 = _to_uint8(image)
    size = img8.shape[0]
    center = np.array([size / 2, size / 2])

    thresh_val = float(np.percentile(img8, threshold_percentile))
    _, binary = cv2.threshold(img8, thresh_val, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "max_intensity": float(img8.max()),
            "mean_intensity": float(img8.mean()),
            "hotspot_area": 0.0,
            "hotspot_eccentricity": 0.0,
            "hotspot_dist_from_center": float(np.linalg.norm(center)),
            "n_hotspots": 0,
        }

    areas = [cv2.contourArea(c) for c in contours]
    largest = contours[int(np.argmax(areas))]
    area = float(cv2.contourArea(largest))

    m = cv2.moments(largest)
    if m["m00"] != 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    else:
        cx, cy = center

    eccentricity = 0.0
    if len(largest) >= 5:
        (_, _), (major, minor), _ = cv2.fitEllipse(largest)
        major, minor = max(major, minor), min(major, minor)
        if major > 0:
            eccentricity = float(np.sqrt(1 - (minor / major) ** 2))

    dist_from_center = float(np.linalg.norm(np.array([cx, cy]) - center))

    return {
        "max_intensity": float(img8.max()),
        "mean_intensity": float(img8.mean()),
        "hotspot_area": area,
        "hotspot_eccentricity": eccentricity,
        "hotspot_dist_from_center": dist_from_center,
        "n_hotspots": len(contours),
    }


def extract_features_batch(images, component=None):
    """Extract handcrafted features for a batch of images into a DataFrame.

    If `component` is given (the component type an inspector would already know when
    taking the photo), it's one-hot encoded and appended as extra columns.
    """
    rows = [extract_hotspot_features(img) for img in images]
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    if component is not None:
        comp_dummies = pd.get_dummies(pd.Series(component, name="component"), prefix="component")
        df = pd.concat([df, comp_dummies], axis=1)
    return df
