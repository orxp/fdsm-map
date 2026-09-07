"""Generate and display an FDSM from a synthetic multi-frequency signal."""

import numpy as np

from fdsm import transform
from fdsm.plotting import plot_map

x = np.linspace(0.0, 10.0, 1001)
y = (
    np.sin(2.0 * np.pi * 0.7 * x)
    + 0.25 * np.sin(2.0 * np.pi * 2.1 * x + 0.4)
    + 0.15 * (x / x.max())
)

result = transform(y, x=x, order_step=0.125, sg_window=11, sg_polyorder=3)
print(result.metadata)

figure, _axes = plot_map(result)
figure.show()

