import json

import numpy as np

from fdsm.cli import main


def test_cli_writes_self_describing_archive(tmp_path):
    x = np.linspace(0.0, 5.0, 101)
    y = np.sin(2 * np.pi * x)
    input_path = tmp_path / "signal.csv"
    output_path = tmp_path / "map.npz"
    np.savetxt(input_path, np.column_stack([x, y]), delimiter=",")

    return_code = main(
        [
            str(input_path),
            str(output_path),
            "--x-column",
            "0",
            "--y-column",
            "1",
            "--order-step",
            "0.5",
        ]
    )

    assert return_code == 0
    with np.load(output_path) as archive:
        assert archive["map"].shape == (5, 91)
        assert archive["orders"].shape == (5,)
        metadata = json.loads(str(archive["metadata"]))
    assert metadata["n_input_samples"] == 101
    assert metadata["n_output_samples"] == 91
