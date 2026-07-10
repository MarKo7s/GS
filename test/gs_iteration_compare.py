"""One-off iteration-by-iteration GS comparison (filter off). Does not modify pyGS/pyPhasemasks."""
import sys
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parent
ALGORITHMS = ROOT.parent.parent
GS_DIR = ALGORITHMS / "GS"
sys.path.insert(0, str(ALGORITHMS))
sys.path.insert(0, str(GS_DIR))

from GS import pyPhasemasks
from pyGS import GS
from toolbox.gs_operators import complex_field, power_field, normalize_field, create_field
from toolbox.setup import optical_setup, GS_mask_setup

# Shared optical params (match GS_test.ipynb intent)
KW = dict(
    wavelength=1.3,
    fin=25e3,
    fout=8e3,
    masksize=960,
    slm_pixel_size=9.2,
    MMF_mfd=12.2,
    downscale=2,
    upsample=0,
    MAX_iterations=300,
)

def rel_err(a, b):
    a = cp.asarray(a)
    b = cp.asarray(b)
    num = float(cp.sqrt(cp.sum(cp.abs(a - b) ** 2)))
    den = float(cp.sqrt(cp.sum(cp.abs(b) ** 2)))
    return num / max(den, 1e-30)

def report(name, a, b):
    e = rel_err(a, b)
    print(f"  {name:24s} rel_L2 = {e:.3e}")
    return e

def main():
    data = np.load(ROOT / "data_testing/modal_decompositions/faces.npz")
    sad = data["sad"][0].astype(np.complex64)
    Target = np.empty((1, sad.shape[0]), np.complex64)
    Target[0, :] = sad

    print("=== Building environments (filter OFF, downscale=2) ===")
    old = pyPhasemasks.SetupGenericMaskGSenvioment(
        wavelength=KW["wavelength"],
        fin=KW["fin"],
        fout=KW["fout"],
        masksize=KW["masksize"],
        slm_pixel_size=KW["slm_pixel_size"],
        MMF_mfd=KW["MMF_mfd"],
        MMFdiameter=62.5,
        MaxModegroups=22,
        maskscount=1,
        goal_fidelity=0.99,
        downscale=KW["downscale"],
        upsample=KW["upsample"],
        MAX_iterations=KW["MAX_iterations"],
    )

    optical = optical_setup(
        wavelength=KW["wavelength"],
        fin=KW["fin"],
        fout=KW["fout"],
        masksize=KW["masksize"],
        slm_pixel_size=KW["slm_pixel_size"],
        MMF_mfd=KW["MMF_mfd"],
        MMF_diameter=62.5,
        MMF_mode_groups=22,
        mode_basis="LG",
    )
    masksetup = GS_mask_setup(
        num_masks=1,
        downscale=KW["downscale"],
        upsample=KW["upsample"],
        MAX_iterations=KW["MAX_iterations"],
        filtering=False,
    )
    new = GS(optical, masksetup, engine_mode="fields", number_masks=1)

    print(f"Old SLM_PLANE_N={old.SLM_PLANE_N}, New SLM_PLANE_N={new.slmplane.SLM_PLANE_N}")
    print(f"Old SLM_N={old.SLM_N}, New SLM_N={new.slmplane.SLM_N}")

    B_old = old.ILLUMINATION_gpu
    B_new = new.gpu_memory.get("ILLUMINATION")
    R0_old = old.R0_gpu
    R0_new = new.gpu_memory.get("R0")
    M_old = old.Modes1D_bases
    M_new = new.gpu_memory.get("MODES_ARRAY_1D")

    print("\n=== Static catalog comparison ===")
    e_B = report("ILLUMINATION", B_old, B_new)
    e_R0 = report("R0", R0_old, R0_new)
    e_M = report("MODES_ARRAY_1D", M_old, M_new)

    # Target field init (mask 0)
    coefs = cp.asarray(Target[0])
    tf_old_1d = cp.matmul(coefs, M_old)
    tf_old_1d = tf_old_1d / cp.sqrt(cp.sum(cp.abs(tf_old_1d) ** 2))
    N = old.SLM_PLANE_N
    Target_Mode = cp.reshape(tf_old_1d, (N, N))
    SLM_Target = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(Target_Mode)))
    A_old = SLM_Target / cp.sqrt(cp.sum(cp.abs(SLM_Target) ** 2))

    tf_new_1d = cp.matmul(coefs, M_new)
    tf_new_2d = cp.reshape(tf_new_1d, (N, N))
    target_field = normalize_field(tf_new_2d, power_field(tf_new_2d))
    target_at_slm = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(target_field)))
    A_new = normalize_field(target_at_slm, power_field(target_at_slm))
    Target_new = target_field

    print("\n=== After target init ===")
    e_A0 = report("A initial", A_old, A_new)
    e_T = report("target_field 2D", Target_Mode, Target_new)

    R1_old = cp.negative(R0_old) + 1
    R1_new = cp.negative(R0_new) + 1
    crop = old.SLM_N // 2
    c = old.SLM_center

    max_iters = 80
    print(f"\n=== Per-iteration GS update (filter OFF), {max_iters} steps ===")
    print(f"{'i':>3}  {'rel_A':>10}  {'rel_D':>10}  {'rel_C_crop':>10}  {'rel_mask':>10}  {'eff_old':>8}  {'eff_new':>8}")

    w_old = 1.0
    w_new = 1.0
    first_div = None

    for i in range(max_iters):
        # forward
        C_old = cp.multiply(cp.abs(B_old), cp.exp(1j * cp.angle(A_old)))
        D_old = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(C_old)))
        D_old = D_old / cp.sqrt(cp.sum(cp.abs(D_old) ** 2))

        C_new = complex_field(B_new, A_new)
        D_new = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(C_new)))
        D_new = normalize_field(D_new, power_field(D_new))

        # efficiency on full D
        D1d_old = cp.reshape(D_old, -1).astype(cp.complex64)
        D1d_new = cp.reshape(D_new, -1).astype(cp.complex64)
        eff_old = float(cp.sum(cp.abs(cp.matmul(D1d_old, old.Modes1D_bases_conj)) ** 2))
        eff_new = float(power_field(cp.matmul(D1d_new, new.gpu_memory.get("MODES_ARRAY_1D_HERMITIAN"))))

        mask_old = C_old[c - crop : c + crop, c - crop : c + crop]
        mask_new = C_new[c - crop : c + crop, c - crop : c + crop]

        e_A = rel_err(A_old, A_new) if i > 0 else 0.0
        e_D = rel_err(D_old, D_new)
        e_C = rel_err(C_old, C_new)
        e_m = rel_err(mask_old, mask_new)

        if first_div is None and e_D > 1e-4:
            first_div = i

        print(f"{i:3d}  {e_A:10.3e}  {e_D:10.3e}  {e_C:10.3e}  {e_m:10.3e}  {eff_old:8.4f}  {eff_new:8.4f}")

        # old update (filter off)
        scatter_old = (D_old * R1_old) / cp.sqrt(cp.sum(cp.abs(D_old * R1_old) ** 2))
        scatter2_old = cp.sqrt(1 - w_old) * scatter_old
        E_p = cp.sqrt(cp.sum(cp.abs(Target_Mode * R0_old) ** 2))
        E_old = (Target_Mode * R0_old / E_p) * w_old + scatter2_old
        A_old = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E_old)))
        w_old *= 0.99

        # new update (filter off)
        scatter_new = D_new * R1_new
        scatter_new = normalize_field(scatter_new, power_field(scatter_new))
        scatter_w = cp.sqrt(1 - w_new) * scatter_new
        E_core = Target_new * R0_new
        E_pow = cp.sqrt(power_field(E_core))
        E_new = create_field(E_core, E_pow, scatter_w, w_new)
        A_new = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E_new)))
        w_new *= 0.99

    print("\n=== Summary ===")
    print(f"ILLUMINATION diff: {e_B:.3e}")
    print(f"R0 diff:           {e_R0:.3e}")
    print(f"MODES diff:        {e_M:.3e}")
    print(f"A_init diff:       {e_A0:.3e}")
    print(f"target_field diff: {e_T:.3e}")
    if e_B > 1e-5 or e_R0 > 1e-5 or e_M > 1e-4:
        print("=> Divergence likely from SETUP arrays not matching between engines.")
    elif e_A0 > 1e-5 or e_T > 1e-5:
        print("=> Divergence likely from TARGET INIT path.")
    elif first_div == 0:
        print("=> Forward pass (C,D) already differs at i=0 — check B/A or FFT grid.")
    elif first_div == 1:
        print("=> First GS update (scatter/E/A) introduces divergence at i=1.")
    else:
        print(f"=> Trajectories track until iteration {first_div} (filter-off update path).")

if __name__ == "__main__":
    main()
