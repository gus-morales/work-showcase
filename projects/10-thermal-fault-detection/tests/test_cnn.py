import numpy as np
import tensorflow as tf

from train_cnn import VMAX, VMIN, build_model, normalize


def test_normalize_clips_and_scales():
    img = np.array([[VMIN - 10, VMIN, (VMIN + VMAX) / 2, VMAX, VMAX + 10]], dtype="float32")
    out = normalize(img)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[0, 1], 0.0)
    assert np.isclose(out[0, -1], 1.0)
    assert np.isclose(out[0, -2], 1.0)


def test_build_model_output_shape_and_range():
    model = build_model(input_shape=(64, 64, 1))
    x = np.random.default_rng(0).random((4, 64, 64, 1)).astype("float32")
    out = model(x, training=False).numpy()
    assert out.shape == (4, 1)
    assert np.all((out >= 0) & (out <= 1))


def test_model_has_named_last_conv_layer_for_gradcam():
    model = build_model(input_shape=(64, 64, 1))
    # interpret.py looks this layer up by name; if it's renamed, Grad-CAM breaks silently
    layer = model.get_layer("conv3")
    assert layer.output.shape[-1] == 32


def test_model_weights_deterministic_with_fixed_seed():
    tf.keras.utils.set_random_seed(10)
    m1 = build_model()
    w1 = m1.get_layer("conv1").get_weights()[0]

    tf.keras.utils.set_random_seed(10)
    m2 = build_model()
    w2 = m2.get_layer("conv1").get_weights()[0]

    np.testing.assert_allclose(w1, w2)
