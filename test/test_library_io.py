"""Round-trip GS settings and masks without building a GPU engine."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from gs import GS_mask_setup, load_masks, optical_setup, settings_from_json
from gs.utils.library import save_library


def _engine(engine_mode="fibremodes", maskcount=21, w0=2.15):
    optical = optical_setup(
        wavelength=0.6328,
        fin=4.51,
        fout=18,
        masksize=32,
        slm_pixel_size=8,
        MMF_mfd=34,
        MMF_diameter=50,
        MMF_mode_groups=10,
        mode_basis="LG",
    )
    mask = GS_mask_setup(filtering=False, filtering_mask_radius_factor=4)
    phase_masks = np.full((maskcount, 4, 4), 0.25, dtype=np.float32)
    return SimpleNamespace(
        engine_mode=engine_mode,
        opticalsetup=optical,
        masksetup=mask,
        _options={"mode_range": (0, 20)} if engine_mode == "fibremodes" else {"number_masks": maskcount},
        maskcount=maskcount,
        illumination=SimpleNamespace(_calculated_W0=w0, mode="Single Mode Fiber"),
        phase_masks=phase_masks,
        efficiency=np.ones(maskcount, dtype=np.float32),
        iteration_convergence=np.arange(maskcount, dtype=np.int32),
        convergence_history=np.zeros((maskcount, 3), dtype=np.float32),
        efficiency_history=np.zeros((maskcount, 3), dtype=np.float32),
        D_core_coefs=np.zeros((maskcount, 2), dtype=np.complex64),
        mask_passed=np.zeros(maskcount, dtype=np.uint8),
        basis_coupling_snr=np.zeros(maskcount, dtype=np.float32),
        dumping_field=np.zeros(maskcount, dtype=np.float32),
        core_field=np.zeros(maskcount, dtype=np.float32),
        leaked_power=np.zeros(maskcount, dtype=np.float32),
        _pattern_sequence_scan_array=np.arange(maskcount),
    )


def test_save_roundtrip_and_folder_name(tmp_path):
    engine = _engine()
    run_dir = save_library(engine, folder=tmp_path, label="double clad")

    expected = (
        "double_clad_modes_lib_fin_4.51_fout_18_wav_0.6328_MFD_34_"
        "num_masks_21_w0slm_2.15_basis_LG_filtering_False_mask_radius_factor_4"
    )
    assert run_dir.name == expected
    assert run_dir.parent == tmp_path

    optical, mask, engine_mode, options = settings_from_json(run_dir)
    assert engine_mode == "fibremodes"
    assert options["mode_range"] == (0, 20)
    assert optical.fin == 4.51
    assert optical.wavelength == 0.6328
    assert optical.MMF_mfd == 34
    assert optical.mode_basis == "LG"
    assert optical.W0_slm is None
    assert mask.filtering is False
    assert mask.filtering_mask_radius_factor == 4
    assert mask.goal_value == engine.masksetup.goal_value

    payload = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    assert payload["label"] == "double_clad"
    assert payload["meta"]["maskcount"] == 21
    assert payload["meta"]["w0_slm_um"] == 2.15
    assert payload["meta"]["folder_name"] == expected
    assert payload["meta"]["saved_at"]

    loaded = load_masks(run_dir)
    assert loaded["phase_masks"].shape == (21, 4, 4)
    assert np.allclose(loaded["phase_masks"], engine.phase_masks)
    assert "pattern_sequence" in loaded


def test_same_settings_overwrite_the_same_folder(tmp_path):
    engine = _engine()
    first = save_library(engine, folder=tmp_path)
    engine.phase_masks[:] = 1.5
    second = save_library(engine, folder=tmp_path)

    assert first == second
    assert len(list(tmp_path.iterdir())) == 1
    assert np.allclose(load_masks(second)["phase_masks"], 1.5)


def test_field_library_prefix(tmp_path):
    engine = _engine(engine_mode="fields", maskcount=2)
    run_dir = save_library(engine, folder=tmp_path)
    assert run_dir.name.startswith("field_lib_fin_4.51_")
    assert "_num_masks_2_" in run_dir.name

    _, _, engine_mode, options = settings_from_json(run_dir / "settings.json")
    assert engine_mode == "fields"
    assert options["number_masks"] == 2
