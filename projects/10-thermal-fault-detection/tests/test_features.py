import numpy as np

from features import FEATURE_NAMES, extract_features_batch, extract_hotspot_features


def test_extract_hotspot_features_returns_all_keys():
    img = np.full((64, 64), 22.0)
    feats = extract_hotspot_features(img)
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_brighter_hotspot_increases_max_intensity():
    base = np.full((64, 64), 22.0)
    hot = base.copy()
    hot[30:34, 30:34] = 60.0
    f_base = extract_hotspot_features(base)
    f_hot = extract_hotspot_features(hot)
    assert f_hot["max_intensity"] > f_base["max_intensity"]


def test_tight_hotspot_has_lower_area_than_diffuse_one():
    size = 64
    y, x = np.mgrid[0:size, 0:size]
    center = size / 2
    tight = 22.0 + 30 * np.exp(-(((x - center) ** 2 + (y - center) ** 2) / (2 * 2**2)))
    diffuse = 22.0 + 30 * np.exp(-(((x - center) ** 2 + (y - center) ** 2) / (2 * 15**2)))
    f_tight = extract_hotspot_features(tight)
    f_diffuse = extract_hotspot_features(diffuse)
    assert f_tight["hotspot_area"] < f_diffuse["hotspot_area"]


def test_batch_extraction_shape_and_component_dummies():
    images = np.random.default_rng(0).normal(22, 2, size=(10, 64, 64)).astype("float32")
    component = np.array(["wheel_motor"] * 5 + ["engine_bay"] * 5)
    df = extract_features_batch(images, component)
    assert len(df) == 10
    assert "component_wheel_motor" in df.columns
    assert "component_engine_bay" in df.columns
    assert not df.isna().any().any()


def test_batch_extraction_without_component():
    images = np.random.default_rng(1).normal(22, 2, size=(5, 64, 64)).astype("float32")
    df = extract_features_batch(images)
    assert len(df) == 5
    assert list(df.columns) == FEATURE_NAMES
