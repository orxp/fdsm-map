import numpy as np

from fdsm import transform, transform_batch


def synthetic_signal():
    x = np.linspace(0.0, 10.0, 1001)
    y = np.sin(2 * np.pi * 0.7 * x) + 0.25 * np.sin(2 * np.pi * 2.1 * x + 0.4)
    return x, y


def test_transform_shape_coordinates_and_metadata():
    x, y = synthetic_signal()
    result = transform(y, x=x, order_step=0.125)
    assert result.map.shape == (17, 901)
    assert result.x.shape == (901,)
    assert result.orders.shape == (17,)
    assert result.map.dtype == np.float32
    assert result.metadata["crop_samples_per_edge"] == 50
    assert result.metadata["nyquist_policy"] == "retain"
    assert np.all(np.isfinite(result.map))


def test_rows_are_standardized_by_default():
    x, y = synthetic_signal()
    result = transform(y, x=x)
    np.testing.assert_allclose(result.map.mean(axis=1), 0.0, atol=2e-6)
    np.testing.assert_allclose(result.map.std(axis=1), 1.0, atol=2e-6)


def test_descending_axis_is_reversed_with_signal():
    x, y = synthetic_signal()
    ascending = transform(y, x=x)
    descending = transform(y[::-1], x=x[::-1])
    np.testing.assert_allclose(descending.x, ascending.x)
    np.testing.assert_allclose(descending.map, ascending.map, atol=1e-6)
    assert descending.metadata["axis_reversed"] is True


def test_affine_axis_units_leave_normalized_map_unchanged():
    x, y = synthetic_signal()
    original = transform(y, x=x)
    rescaled = transform(y, x=1000.0 + 4.2 * x)
    np.testing.assert_allclose(rescaled.map, original.map, atol=2e-6)


def test_batch_shape_and_shared_coordinates():
    x, y = synthetic_signal()
    result = transform_batch(np.stack([y, 0.5 * y, -y]), x=x, order_step=0.25)
    assert result.map.shape == (3, 9, 901)
    assert result.metadata["n_signals"] == 3
    np.testing.assert_allclose(result.map[0], result.map[1], atol=2e-6)
    np.testing.assert_allclose(result.map[0], -result.map[2], atol=2e-6)


def test_no_smoothing_and_no_normalization_are_available():
    x, y = synthetic_signal()
    result = transform(y, x=x, sg_window=None, normalize=None, output_dtype=np.float64)
    assert result.map.dtype == np.float64
    assert result.metadata["sg_window"] is None
    assert result.metadata["normalization"] is None
