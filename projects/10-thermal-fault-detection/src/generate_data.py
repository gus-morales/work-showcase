"""Synthetic thermal-camera inspection images for the equipment-fault-detection problem.

Same fictional mining company and haul-truck fleet as project 06, viewed through a
different data source: a handheld thermal camera used during routine walkarounds,
instead of the continuous sensor telemetry project 06 uses.

Each image is a 64x64 array of temperature-like values for one component. Healthy
images carry that component's normal heat signature plus sensor noise; fault images
carry the same signature plus an extra localized hot spot at a graded severity. A
minority of healthy images also carry a benign warm patch (e.g. sun glare, an exhaust
vent) that is not a fault, so a naive "flag the brightest pixel" rule doesn't trivially
solve the problem.
"""
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IMG_SIZE = 64
N_IMAGES = 2000
FAULT_RATE = 0.12
SEED = 10

COMPONENT_TYPES = ["wheel_motor", "engine_bay", "electrical_cabinet"]
SEVERITY_LEVELS = ["mild", "moderate", "severe"]


def _gaussian_bump(size, cx, cy, amplitude, sigma):
    y, x = np.mgrid[0:size, 0:size]
    return amplitude * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)))


def _baseline_field(component_type, size, rng):
    """The characteristic, healthy heat signature for a component type."""
    field = np.full((size, size), 22.0)
    if component_type == "wheel_motor":
        field += _gaussian_bump(size, size * 0.5, size * 0.5, amplitude=18, sigma=size * 0.18)
    elif component_type == "engine_bay":
        field += _gaussian_bump(size, size * 0.35, size * 0.55, amplitude=22, sigma=size * 0.25)
    elif component_type == "electrical_cabinet":
        for _ in range(3):
            cx, cy = rng.uniform(size * 0.2, size * 0.8, size=2)
            field += _gaussian_bump(size, cx, cy, amplitude=rng.uniform(6, 10), sigma=size * 0.06)
    else:
        raise ValueError(f"unknown component_type: {component_type}")
    return field


def _benign_warm_patch(size, rng):
    """A normal warm spot that is not a fault but can fool a naive intensity threshold."""
    cx, cy = rng.uniform(size * 0.1, size * 0.9, size=2)
    return _gaussian_bump(size, cx, cy, amplitude=rng.uniform(8, 14), sigma=size * 0.08)


def _fault_hotspot(size, rng):
    severity = rng.choice(SEVERITY_LEVELS, p=[0.4, 0.35, 0.25])
    amplitude_ranges = {"mild": (10, 16), "moderate": (16, 26), "severe": (26, 40)}
    amplitude = rng.uniform(*amplitude_ranges[severity])
    cx, cy = rng.uniform(size * 0.15, size * 0.85, size=2)
    sigma = rng.uniform(size * 0.04, size * 0.07)
    return _gaussian_bump(size, cx, cy, amplitude, sigma), str(severity), float(cx), float(cy)


def generate_thermal_domain(n_images=N_IMAGES, fault_rate=FAULT_RATE, seed=SEED):
    rng = np.random.default_rng(seed)
    images = np.zeros((n_images, IMG_SIZE, IMG_SIZE), dtype="float32")
    labels = np.zeros(n_images, dtype="int64")
    component = np.empty(n_images, dtype=object)
    severity = np.empty(n_images, dtype=object)
    fault_x = np.full(n_images, np.nan, dtype="float32")
    fault_y = np.full(n_images, np.nan, dtype="float32")

    for i in range(n_images):
        comp = str(rng.choice(COMPONENT_TYPES))
        field = _baseline_field(comp, IMG_SIZE, rng)

        is_fault = rng.random() < fault_rate
        sev = "none"
        if is_fault:
            bump, sev, cx, cy = _fault_hotspot(IMG_SIZE, rng)
            field = field + bump
            fault_x[i], fault_y[i] = cx, cy
        elif rng.random() < 0.15:
            field = field + _benign_warm_patch(IMG_SIZE, rng)

        field = field + rng.normal(0, 1.2, size=field.shape) + rng.normal(0, 1.5)

        images[i] = field
        labels[i] = int(is_fault)
        component[i] = comp
        severity[i] = sev

    return images, labels, component, severity, fault_x, fault_y


def main():
    DATA_DIR.mkdir(exist_ok=True)
    images, labels, component, severity, fault_x, fault_y = generate_thermal_domain()
    out_path = DATA_DIR / "thermal_images.npz"
    np.savez_compressed(
        out_path,
        images=images,
        labels=labels,
        component=component,
        severity=severity,
        fault_x=fault_x,
        fault_y=fault_y,
    )
    print(f"Wrote {len(labels)} thermal images -> {out_path}")
    print(f"Fault rate: {labels.mean():.1%}")
    for comp in COMPONENT_TYPES:
        mask = component == comp
        print(f"  {comp}: {mask.sum()} images, {labels[mask].mean():.1%} fault rate")


if __name__ == "__main__":
    main()
