import numpy as np

from generate_data import COMPONENT_TYPES, IMG_SIZE, generate_thermal_domain


def test_shapes_and_types():
    images, labels, component, severity, fault_x, fault_y = generate_thermal_domain(
        n_images=200, seed=1
    )
    assert images.shape == (200, IMG_SIZE, IMG_SIZE)
    assert labels.shape == (200,)
    assert set(np.unique(labels)) <= {0, 1}
    assert set(component) <= set(COMPONENT_TYPES)


def test_fault_rate_matches_target():
    images, labels, *_ = generate_thermal_domain(n_images=3000, fault_rate=0.12, seed=2)
    assert abs(labels.mean() - 0.12) < 0.03


def test_fault_location_recorded_only_for_faults():
    images, labels, component, severity, fault_x, fault_y = generate_thermal_domain(
        n_images=500, seed=3
    )
    has_location = ~np.isnan(fault_x)
    assert np.array_equal(has_location, labels.astype(bool))


def test_fault_images_contain_a_real_hotspot_at_recorded_location():
    """The recorded fault (x, y) should genuinely be near one of the brightest points
    in that image, not an arbitrary coordinate the generator forgot to use."""
    images, labels, component, severity, fault_x, fault_y = generate_thermal_domain(
        n_images=500, seed=4
    )
    fault_idx = np.where(labels == 1)[0]
    assert len(fault_idx) > 0
    for i in fault_idx[:30]:
        x, y = int(round(float(fault_x[i]))), int(round(float(fault_y[i])))
        x, y = np.clip(x, 0, IMG_SIZE - 1), np.clip(y, 0, IMG_SIZE - 1)
        y0, y1 = max(0, y - 2), min(IMG_SIZE, y + 3)
        x0, x1 = max(0, x - 2), min(IMG_SIZE, x + 3)
        local_patch = images[i, y0:y1, x0:x1]
        assert local_patch.max() >= np.percentile(images[i], 90)


def test_severity_is_none_only_for_healthy():
    images, labels, component, severity, fault_x, fault_y = generate_thermal_domain(
        n_images=300, seed=5
    )
    assert np.all(severity[labels == 0] == "none")
    assert np.all(severity[labels == 1] != "none")


def test_deterministic_with_fixed_seed():
    a = generate_thermal_domain(n_images=100, seed=7)
    b = generate_thermal_domain(n_images=100, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
