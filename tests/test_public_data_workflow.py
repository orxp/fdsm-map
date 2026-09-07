from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "examples" / "public_data_workflow.py"
SPEC = importlib.util.spec_from_file_location("public_data_workflow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


def test_load_mzml_tic_and_convert_seconds(tmp_path: Path) -> None:
    time = np.asarray([0.0, 30.0, 60.0], dtype="<f8")
    intensity = np.asarray([2.0, 5.0, 3.0], dtype="<f4")
    time_encoded = base64.b64encode(time.tobytes()).decode()
    intensity_encoded = base64.b64encode(intensity.tobytes()).decode()
    mzml = tmp_path / "tiny.mzML"
    mzml.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml">
  <chromatogramList count="1">
    <chromatogram id="TIC" defaultArrayLength="3">
      <cvParam accession="MS:1000235" name="total ion current chromatogram"/>
      <binaryDataArrayList count="2">
        <binaryDataArray>
          <cvParam accession="MS:1000523"/>
          <cvParam accession="MS:1000595" unitName="second"/>
          <binary>{time_encoded}</binary>
        </binaryDataArray>
        <binaryDataArray>
          <cvParam accession="MS:1000521"/>
          <cvParam accession="MS:1000515"/>
          <binary>{intensity_encoded}</binary>
        </binaryDataArray>
      </binaryDataArrayList>
    </chromatogram>
  </chromatogramList>
</mzML>""",
        encoding="utf-8",
    )
    loaded_time, loaded_intensity = workflow.load_mzml_tic(mzml)
    np.testing.assert_allclose(loaded_time, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(loaded_intensity, intensity)


def test_load_openspecy_asp(tmp_path: Path) -> None:
    asp = tmp_path / "tiny.asp"
    asp.write_text("3\n4000\n1000\n1\n2\n4\n0.1\n0.5\n0.2\n", encoding="utf-8")
    x, y = workflow.load_openspecy_asp(asp)
    np.testing.assert_allclose(x, [4000, 2500, 1000])
    np.testing.assert_allclose(y, [0.1, 0.5, 0.2])


def test_resample_uniform_reverses_descending_axis() -> None:
    x, y = workflow.resample_uniform(
        np.asarray([3.0, 1.0, 0.0]), np.asarray([9.0, 1.0, 0.0]), n_points=4
    )
    np.testing.assert_allclose(x, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(y, [0.0, 1.0, 5.0, 9.0])
    np.testing.assert_allclose(np.diff(x), np.diff(x)[0])
