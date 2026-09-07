import numpy as np
import pytest

from fdsm import fractional_derivative_fft


def analytic_fractional_multitone(x, components, order):
    output = np.zeros_like(x)
    for amplitude, cycles, phase in components:
        output += amplitude * (2 * np.pi * cycles) ** order * np.sin(
            2 * np.pi * cycles * x + phase + np.pi * order / 2
        )
    return output


@pytest.mark.parametrize("order", [0.5, 1.0, 1.5, 2.0])
def test_fft_operator_matches_analytic_periodic_derivative(order):
    n = 2048
    x = np.arange(n) / n
    components = [(1.0, 5, 0.0), (0.5, 12, 0.7), (0.3, 23, 1.9)]
    y = sum(a * np.sin(2 * np.pi * k * x + p) for a, k, p in components)
    calculated = fractional_derivative_fft(y, delta=1 / n, order=order, pad_ratio=0)
    expected = analytic_fractional_multitone(x, components, order)
    relative_peak_error = np.max(np.abs(calculated - expected)) / np.max(np.abs(expected))
    assert relative_peak_error < 1e-10


def test_order_zero_retains_signal():
    y = np.linspace(-1, 1, 100)
    calculated = fractional_derivative_fft(y, delta=0.1, order=0, pad_ratio=0.25)
    np.testing.assert_allclose(calculated, y, atol=1e-12)


def test_unpaired_nyquist_policies_are_distinct():
    y = (-1.0) ** np.arange(128)
    retained = fractional_derivative_fft(
        y, delta=1.0, order=0.5, pad_ratio=0, nyquist="retain"
    )
    removed = fractional_derivative_fft(
        y, delta=1.0, order=0.5, pad_ratio=0, nyquist="zero_unpaired"
    )

    assert np.max(np.abs(retained)) > 0.1
    assert np.max(np.abs(removed)) < 1e-12
