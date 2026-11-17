"""phase2_structure_tensor.py - local orientation of the papyrus from the CT (no learning involved).

At every voxel the structure tensor J = K_rho * (grad I_sigma)(grad I_sigma)^T is formed and split into eigenvectors
v1, v2, v3 (eigenvalues l1 >= l2 >= l3):
    v1  the direction of strongest change: the sheet NORMAL
    v3  the direction of least change: the in-sheet direction the tracker follows (the FIBRE direction)
    (l1 - l2) / (l1 + l2 + l3), clipped to 0..1: the tensor's own "sheet measure" (1 = clearly a sheet)

Both scales are in voxels and assume 9.6 um voxels; at another resolution scale them (docs/TUNING.md).
    sigma  smoothing before the gradient (noise vs. detail)          pipeline uses 1.0
    rho    size of the neighbourhood the orientation is averaged over  pipeline uses 6.0 (larger = steadier, but
           blurs where sheets bend sharply or touch)

The normal's sign is set to point from darker to brighter CT at each voxel. That is a LOCAL sign: it flips across the
middle of a sheet, so later steps never rely on it (phase 5 and the tracker align signs themselves).

Results are identical to the earlier version; the eigen-decomposition is now done in slabs to use less memory.
"""
import numpy as np
from scipy import ndimage as ndi


def compute_structure_tensor(vol: np.ndarray, sigma: float = 1.0, rho: float = 6.0, slab: int = 16):
    """vol (z, y, x) float -> evals (z, y, x, 3) descending, evecs (z, y, x, 3, 3) with evecs[..., :, k] = v_(k+1),
    sheet measure (z, y, x), polarity (z, y, x) = gradient strength along the oriented normal."""
    gz = ndi.gaussian_filter(vol, sigma=sigma, order=[1, 0, 0])
    gy = ndi.gaussian_filter(vol, sigma=sigma, order=[0, 1, 0])
    gx = ndi.gaussian_filter(vol, sigma=sigma, order=[0, 0, 1])

    comps = {k: ndi.gaussian_filter(a * b, sigma=rho) for k, (a, b) in
             dict(zz=(gz, gz), yy=(gy, gy), xx=(gx, gx), zy=(gz, gy), zx=(gz, gx), yx=(gy, gx)).items()}

    D, H, W = vol.shape
    evals = np.empty((D, H, W, 3), dtype=np.float32)
    evecs = np.empty((D, H, W, 3, 3), dtype=np.float32)
    for z0 in range(0, D, slab):                                  # same numbers as one big eigh, a fraction of the memory
        z1 = min(z0 + slab, D); s = slice(z0, z1)
        J = np.empty((z1 - z0, H, W, 3, 3), dtype=np.float32)
        J[..., 0, 0] = comps["zz"][s]; J[..., 1, 1] = comps["yy"][s]; J[..., 2, 2] = comps["xx"][s]
        J[..., 0, 1] = J[..., 1, 0] = comps["zy"][s]
        J[..., 0, 2] = J[..., 2, 0] = comps["zx"][s]
        J[..., 1, 2] = J[..., 2, 1] = comps["yx"][s]
        w, v = np.linalg.eigh(J)                                  # ascending
        evals[s] = w[..., ::-1]; evecs[s] = v[..., :, ::-1]       # descending: v1 = normal, v3 = fibre direction
        del J, w, v
    del comps

    l1, l2, l3 = evals[..., 0], evals[..., 1], evals[..., 2]
    sheet_prob = np.clip((l1 - l2) / (l1 + l2 + l3 + 1e-6), 0.0, 1.0)

    # orient the normal from darker to brighter CT (a local sign; see the note at the top)
    dot = gz * evecs[..., 0, 0] + gy * evecs[..., 1, 0] + gx * evecs[..., 2, 0]
    flip = np.where(dot < 0, -1.0, 1.0).astype(np.float32)
    evecs[..., :, 0] *= flip[..., None]
    polarity = dot * flip
    return evals, evecs, sheet_prob, polarity


def build_phase4_field_stack(evecs: np.ndarray, sheet_prob: np.ndarray, polarity: np.ndarray) -> np.ndarray:
    """Pack the fields the later phases read into one (z, y, x, 9) float32 array:
        0-2  fibre direction v3 (z, y, x)
        3-5  normal v1 (z, y, x)
        6    sheet measure      (replaced by the U-Net's sheet probability in phase 3 when the U-Net is used)
        7    polarity           (replaced by the U-Net's recto probability)
        8    zero               (filled with the U-Net's verso probability)"""
    D, H, W = sheet_prob.shape
    fs = np.zeros((D, H, W, 9), dtype=np.float32)
    fs[..., 0:3] = evecs[..., :, 2]
    fs[..., 3:6] = evecs[..., :, 0]
    fs[..., 6] = sheet_prob
    fs[..., 7] = polarity
    return fs
