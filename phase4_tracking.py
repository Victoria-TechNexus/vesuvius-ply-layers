
"""phase4_tracking.py - trace short tracks along the papyrus fibres.

Seeds are bright ridge points of the papyrus; from each seed a track is integrated in both directions along the fibre
direction field (RK4), and stops at the tile edge, at a length limit, where the papyrus fades, where the verso is reached,
or where the track turns too sharply. All settings are passed in from run_pipeline.py (TRACK and SEED there).

Field stack channels read here (z, y, x, channel):
    0-2  fibre direction (phase 2)
    3-5  sheet normal (phase 2)        - seeds look for a brightness peak along it
    6    brightness B, 0..1            - run_pipeline.py puts the CT brightness here before tracking
    7    recto probability (phase 3)
    8    verso probability (phase 3)

Results are identical to the earlier version (phase4_streamline_integration_tractography.py); the paths are now
collected step by step in arrays instead of one Python list per track, which is faster.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial import cKDTree

REASONS = ("length", "edge", "sheet", "recto", "verso", "curve")   # stop reasons 0..5 ("sheet" = brightness fell below tau_sheet)


@dataclass(frozen=True)
class Rules:
    """Tracking settings. Defaults are the values used for all published results."""
    step: float = 0.5           # step length (voxels)
    integrator: str = "rk4"     # "rk4" or "euler"
    max_len: float = 600.0      # longest track, each direction (voxels)
    tau_sheet: float = 0.5      # stop where brightness (channel 6) drops below this
    tau_recto: float = 0.0      # stop where recto probability drops below this (0 = off)
    tau_verso: float = 0.6      # stop where verso probability rises above this
    kappa_max: float = 0.5      # stop where the track turns more than this many radians per voxel
    use_classes: bool = True    # apply the recto / verso stops
    trim: float = 1.0           # remove this length (voxels) from a track end stopped by brightness, recto, verso or turning

    def as_dict(self) -> dict:
        return asdict(self)


def trilinear(arr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """arr (Z, Y, X, C); pts (N, 3) as z, y, x. Returns (N, C). Points are clamped to the volume."""
    hi = np.array(arr.shape[:3], dtype=float) - 1.0
    p = np.clip(pts, 0.0, hi - 1e-6)
    i0 = np.floor(p).astype(np.int64)
    f = (p - i0).astype(np.float32)
    i1 = np.minimum(i0 + 1, hi.astype(np.int64))
    z0, y0, x0 = i0.T
    z1, y1, x1 = i1.T
    fz, fy, fx = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    c00 = arr[z0, y0, x0] * (1 - fx) + arr[z0, y0, x1] * fx
    c01 = arr[z0, y1, x0] * (1 - fx) + arr[z0, y1, x1] * fx
    c10 = arr[z1, y0, x0] * (1 - fx) + arr[z1, y0, x1] * fx
    c11 = arr[z1, y1, x0] * (1 - fx) + arr[z1, y1, x1] * fx
    return (c00 * (1 - fy) + c01 * fy) * (1 - fz) + (c10 * (1 - fy) + c11 * fy) * fz


def ridge_seeds(field_stack, covered_pts=None, thr=0.6, offset=1.5, spacing=2.0, cover_r=2.0, batch=1_000_000):
    """Seed points: voxels with brightness (channel 6) above `thr` that are at least as bright as the points `offset`
    voxels either side along the normal (a ridge across the sheet), more than `cover_r` voxels from `covered_pts`
    (earlier tracks), and at most one per `spacing`-voxel cell."""
    zz, yy, xx = np.nonzero(field_stack[..., 6] > thr)
    f = lambda q: trilinear(field_stack[..., 6:7], q).ravel()
    keep = []
    for s in range(0, len(zz), batch):
        sl = slice(s, s + batch)
        P = np.stack([zz[sl], yy[sl], xx[sl]], 1).astype(float)
        n = field_stack[zz[sl], yy[sl], xx[sl], 3:6].astype(float)
        c = f(P)
        keep.append(P[(c >= f(P + offset * n)) & (c >= f(P - offset * n))])
    if not keep:
        return np.empty((0, 3), dtype=float)
    P = np.vstack(keep)
    if covered_pts is not None and len(covered_pts):
        P = P[cKDTree(covered_pts).query(P)[0] > cover_r]
    if len(P) == 0:
        return np.empty((0, 3), dtype=float)
    _, u = np.unique(np.round(P / spacing).astype(np.int64), axis=0, return_index=True)
    return P[u]


class Tracer:
    def __init__(self, field_stack: np.ndarray, sign_safe: bool = True):
        """sign_safe: interpolate the fibre direction after turning the 8 corner vectors to agree (recommended; the
        direction field has no fixed sign, and plain interpolation lets opposite corners cancel)."""
        self.f = field_stack
        self.hi = np.array(field_stack.shape[:3], dtype=float) - 1.0
        self.sign_safe = sign_safe

    def sample(self, pts: np.ndarray) -> np.ndarray:
        return trilinear(self.f, pts)

    def direction(self, pts: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Unit fibre direction at pts, turned to agree with ref."""
        if not self.sign_safe:
            v = self.sample(pts)[:, 0:3]
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
            s = np.sign(np.sum(v * ref, -1)); s[s == 0] = 1.0
            return v * s[:, None]
        f3 = self.f[..., 0:3]
        hi = np.array(f3.shape[:3]) - 1
        p = np.clip(pts, 0.0, hi - 1e-6); i0 = np.floor(p).astype(np.int64); fr = p - i0
        near = f3[tuple(np.clip(np.round(p).astype(np.int64), 0, hi).T)].astype(np.float64)
        near *= np.where(np.sum(near * ref, -1, keepdims=True) < 0, -1.0, 1.0)
        out = np.zeros((len(p), 3))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    c = f3[i0[:, 0] + dz, i0[:, 1] + dy, i0[:, 2] + dx].astype(np.float64)
                    c *= np.where(np.sum(c * near, -1, keepdims=True) < 0, -1.0, 1.0)
                    w = ((fr[:, 0] if dz else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dx else 1 - fr[:, 2]))
                    out += w[:, None] * c
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)

    def _run(self, seeds: np.ndarray, sense: float, rules: Rules):
        n = len(seeds)
        pos = seeds.astype(float).copy()
        alive = np.ones(n, dtype=bool)
        prev = np.zeros((n, 3))
        reason = np.zeros(n, dtype=np.int64)
        h = rules.step
        FIXED = np.array([0.0, 1.0, 0.0])                       # reference for the very first step (sense picks the way)
        step_ids, step_pts = [np.arange(n)], [pos.copy()]       # the seed is each track's first point

        for it in range(int(np.ceil(rules.max_len / h))):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            x = pos[idx]
            ref = np.broadcast_to(sense * FIXED, x.shape) if it == 0 else prev[idx]
            k1 = self.direction(x, ref)
            if rules.integrator == "rk4":
                k2 = self.direction(x + 0.5 * h * k1, k1)
                k3 = self.direction(x + 0.5 * h * k2, k1)
                k4 = self.direction(x + h * k3, k1)
                dx = (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            elif rules.integrator == "euler":
                dx = h * k1
            else:
                raise ValueError(rules.integrator)
            new = x + dx
            dnew = dx / (np.linalg.norm(dx, axis=1, keepdims=True) + 1e-12)
            vals = self.sample(new)

            stop = np.zeros(idx.size, dtype=np.int64)
            if it > 0:
                turn = np.arccos(np.clip(np.sum(dnew * prev[idx], -1), -1.0, 1.0)) / h
                stop[turn > rules.kappa_max] = 5
            if rules.use_classes:
                stop[(stop == 0) & (vals[:, 8] > rules.tau_verso)] = 4
                stop[(stop == 0) & (vals[:, 7] < rules.tau_recto)] = 3
            stop[(stop == 0) & (vals[:, 6] < rules.tau_sheet)] = 2
            stop[np.any((new < 0) | (new > self.hi), axis=1)] = 1

            ok = stop == 0
            alive[idx[~ok]] = False; reason[idx[~ok]] = stop[~ok]
            step_ids.append(idx[ok]); step_pts.append(new[ok])
            pos[idx[ok]] = new[ok]
            prev[idx[ok]] = dnew[ok]

        ids = np.concatenate(step_ids); pts = np.concatenate(step_pts)
        order = np.argsort(ids, kind="stable")                 # group by track, keeping step order
        counts = np.bincount(ids, minlength=n)
        out = np.split(pts[order], np.cumsum(counts)[:-1])
        if rules.trim > 0:
            for i in np.nonzero(reason >= 2)[0]:
                p = out[i]
                if len(p) < 2:
                    continue
                tail = np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1)[::-1])
                drop = int(np.searchsorted(tail, rules.trim, side="left")) + 1
                out[i] = p[:max(1, len(p) - drop)]
        return out, reason

    def trace(self, seeds: np.ndarray, rules: Rules, both: bool = True):
        """Trace every seed forwards and (if both) backwards and join the two halves. Returns the paths (each (n, 3) as
        z, y, x), the count of each stop reason, and for each path the index of its seed point. self.last_reasons holds
        the stop reason of each half."""
        fwd, r1 = self._run(seeds, 1.0, rules)
        if not both:
            self.last_reasons = (r1, np.zeros_like(r1))
            return fwd, {name: int(np.sum(r1 == i)) for i, name in enumerate(REASONS)}, [0] * len(fwd)
        bwd, r2 = self._run(seeds, -1.0, rules)
        self.last_reasons = (r1, r2)
        paths = [np.concatenate([b[::-1], f[1:]]) for f, b in zip(fwd, bwd)]
        counts = {name: int(np.sum(r1 == i) + np.sum(r2 == i)) for i, name in enumerate(REASONS)}
        return paths, counts, [len(b) - 1 for b in bwd]
