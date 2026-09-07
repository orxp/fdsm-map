import numpy as np
import pytest

from fdsm import make_orders, transform


def test_make_orders_includes_both_endpoints():
    np.testing.assert_allclose(make_orders(0.0, 2.0, 0.125), np.linspace(0, 2, 17))


def test_make_orders_rejects_nondivisible_step():
    with pytest.raises(ValueError, match="must divide"):
        make_orders(0.0, 2.0, 0.3)


def test_nonuniform_axis_raises_by_default():
    x = np.array([0.0, 1.0, 2.0, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2])
    with pytest.raises(ValueError, match="uniformly spaced"):
        transform(np.sin(x), x=x, sg_window=None)


def test_nonuniform_axis_can_warn_and_proceed():
    x = np.array([0.0, 1.0, 2.0, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2])
    with pytest.warns(UserWarning, match="median spacing"):
        result = transform(np.sin(x), x=x, sg_window=None, nonuniform="warn")
    assert np.all(np.isfinite(result.map))


@pytest.mark.parametrize("window", [2, 4, 12])
def test_invalid_savgol_window_is_rejected(window):
    with pytest.raises(ValueError):
        transform(np.arange(21.0), sg_window=window)


def test_complex_signal_and_axis_are_rejected():
    with pytest.raises(ValueError, match="real-valued"):
        transform(np.arange(21.0) + 1j, sg_window=None)
    with pytest.raises(ValueError, match="real-valued"):
        transform(np.arange(21.0), x=np.arange(21.0) + 1j, sg_window=None)


def test_integer_output_dtype_is_rejected():
    with pytest.raises(TypeError, match="floating-point"):
        transform(np.arange(21.0), sg_window=None, output_dtype=np.int16)


def test_invalid_nyquist_policy_is_rejected():
    with pytest.raises(ValueError, match="retain"):
        transform(np.arange(21.0), sg_window=None, nyquist="invalid")
