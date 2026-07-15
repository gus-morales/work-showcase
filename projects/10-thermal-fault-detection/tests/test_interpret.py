import numpy as np
import tensorflow as tf

from interpret import grad_cam, heatmap_peak_location
from train_cnn import build_model


def test_grad_cam_output_shape_and_range():
    tf.keras.utils.set_random_seed(0)
    model = build_model(input_shape=(64, 64, 1))
    img = np.random.default_rng(0).random((64, 64, 1)).astype("float32")
    cam, pred = grad_cam(model, img)
    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-5
    assert 0.0 <= pred <= 1.0


def test_heatmap_peak_location_finds_max():
    cam = np.zeros((64, 64))
    cam[10, 50] = 1.0  # array index is [row, col] = [y, x]
    x, y = heatmap_peak_location(cam)
    assert x == 50.0
    assert y == 10.0


def test_heatmap_peak_location_ties_pick_first_occurrence():
    cam = np.zeros((5, 5))
    cam[2, 2] = 1.0
    x, y = heatmap_peak_location(cam)
    assert (x, y) == (2.0, 2.0)
