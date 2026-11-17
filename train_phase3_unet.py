
"""train_phase3_unet.py - train the phase 3 U-Net. OPTIONAL: the pipeline uses the trained checkpoint
phase3_unet_checkpoint.pt and does not need this script. Use it to retrain, or to train on other scrolls.

No human labels are used. For each training block the structure tensor (phase 2, at a fine scale, rho = 1.2) gives the
targets the network learns to reproduce, more smoothly and with less noise:
    sheet  = the tensor's sheet measure s
    recto  = s * (share of the fibre direction lying AROUND the scroll axis)
    verso  = s * (share of the fibre direction lying ALONG the scroll axis)
    polarity (strength of the dark-to-bright edge across the sheet, clipped to 0..1), and a fibre-direction vector
Voxels at the edge of papyrus (where sheets meet gaps) count 5x in the loss.

    python train_phase3_unet.py --check     # first: read every block and report how much papyrus it holds; no training
    python train_phase3_unet.py             # train (needs a GPU in practice); writes CHECKPOINT (refuses to overwrite)

Two versions, chosen with EXTRA_BLOCKS in SETTINGS. Both use the original training with the two bug fixes below (1 and 2), a
fixed random seed, and a held-out block that is only measured (item 5):
    v1.1  EXTRA_BLOCKS = False  the original five blocks (125 patches)
    v1.2  EXTRA_BLOCKS = True   plus six more blocks of the original size (item 3; 11 blocks, 275 patches)
Items 4 and 6 were tried for v2 and are OFF in SETTINGS (SWAP_YX, CLIP_POLARITY). v2 used six LARGER extra blocks (128 on a side,
750 of its 875 patches) together with items 4 and 6; it separated layers worse on the clean tiles.

Changes since the published checkpoint was trained:
  1. Every training block is read and scaled the same way (ScrollStreamer). Before, the PHerc. 332 block was rescaled to its
     own min..max while the Paris 4 blocks were divided by the data type's maximum.
  2. The direction loss matches each patch with its own mask; before, the mask's extra channel mixed the masks of a batch.
  3. Six more blocks about 900 voxels from the scroll centre, around the scroll, near the height of the tiles the pipeline
     works on, but clear of the six tiles themselves (they stay a fair test). The first five blocks all came from the core
     or near it.
  4. Random swaps of the y and x axes, on top of the random flips: together they show every sheet at all 8 quarter-turn and
     mirror orientations around the scroll axis. The recto / verso targets do not change under these turns (they only depend
     on how much the fibre lies along z), so the targets stay correct.
  5. A held-out block (never trained on): every 10 epochs its loss is printed, and the plain error of the sheet / recto / verso
     outputs on its papyrus (0 = perfect) next to the error of always answering the average. Held-out error falling well below
     that average = the network learns papyrus in general, not just its training blocks.
  6. The polarity target is clipped to 0..1 (the loss expects targets in that range).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from phase1_stream_ct import ScrollStreamer
from phase2_structure_tensor import compute_structure_tensor
from phase3_unet import Phase2StageBUNet3D, nematic_sign_invariant_loss

# ================================================================ SETTINGS ================================================================
P4 = dict(scroll="PHercParis4", scan="20260411134726-2.400um-0.2m-78keV-masked.zarr")
# Training blocks: scroll + scan (as in phase 1), and z, y, x ranges at LEVEL. The scroll centre of PHerc. Paris 4 is x 4087, y 4087.
# The six pipeline tiles cover z 15382-15878, y 4794-5210, x 3895-4687: no block below overlaps them.
SOURCES = [
    # ---- the original five (64 x 128 x 128): core and near-core, plus another scroll
    dict(name="PHercParis4 - crushed centre (z 5000)", **P4, z=(5000, 5064), y=(3950, 4078), x=(3950, 4078)),
    dict(name="PHercParis4 - inner core mid-height (z 7500)", **P4, z=(7500, 7564), y=(3950, 4078), x=(3950, 4078)),
    dict(name="PHercParis4 - deep core (z 14890)", **P4, z=(14890, 14954), y=(4101, 4229), x=(4572, 4700)),
    dict(name="PHercParis4 - intermediate radius, relaxed wraps", **P4, z=(5000, 5064), y=(3700, 3828), x=(3700, 3828)),
    dict(name="PHerc0332 - core", scroll="PHerc0332", scan="20251211183505-2.399um-0.2m-78keV-masked.zarr",
         z=(4200, 4264), y=(1910, 2038), x=(1910, 2038)),
]
# Six more, the SAME size as the originals (64 x 128 x 128, 25 patches each), about 900 voxels from the scroll centre, all around the
# scroll. The original five all come from the core or near it.
#   EXTRA_BLOCKS = False -> v1.1: the original five blocks only (125 patches)
#   EXTRA_BLOCKS = True  -> v1.2: all eleven blocks (275 patches)
EXTRA_BLOCKS = False
EXTRA_SOURCES = [
    dict(name="PHercParis4 - tiles' side, below them (z 16132)", **P4, z=(16132, 16196), y=(4900, 5028), x=(4000, 4128)),
    dict(name="PHercParis4 - tiles' side, above them (z 14732)", **P4, z=(14732, 14796), y=(4900, 5028), x=(4300, 4428)),
    dict(name="PHercParis4 - opposite side (-y)", **P4, z=(15532, 15596), y=(3100, 3228), x=(4000, 4128)),
    dict(name="PHercParis4 - right side (+x)", **P4, z=(15532, 15596), y=(4000, 4128), x=(4950, 5078)),
    dict(name="PHercParis4 - left side (-x)", **P4, z=(15532, 15596), y=(4000, 4128), x=(3100, 3228)),
    dict(name="PHercParis4 - diagonal (+x, +y)", **P4, z=(15232, 15296), y=(4700, 4828), x=(4700, 4828)),
]
if EXTRA_BLOCKS: SOURCES = SOURCES + EXTRA_SOURCES
# Held-out block: measured every 10 epochs, NEVER trained on, so it does not change the trained model. It shows how the network
# does on papyrus it has not seen. It lies outside every training block and outside the six pipeline tiles. None = off.
VALIDATION = dict(name="PHercParis4 - held out, diagonal (+x, -y)", **P4, z=(15800, 15928), y=(3350, 3478), x=(4700, 4828))
LEVEL = 2
TARGET_TENSOR = dict(sigma=1.0, rho=1.2)          # fine-scale tensor for the targets
PATCH, STRIDE = 64, 16
EDGE_WEIGHT, PAPYRUS_WEIGHT, BACKGROUND_WEIGHT = 5.0, 1.0, 0.1
PAPYRUS_THRESHOLD = 0.35                          # tensor sheet measure above which a voxel counts as papyrus
EPOCHS, BATCH, LR, WEIGHT_DECAY = 60, 4, 1e-3, 1e-4
DIRECTION_WEIGHT = 5.0                            # direction loss counts 5x the class loss
SWAP_YX = False                                   # random y <-> x swaps on top of the flips (tried for v2; off = original training)
CLIP_POLARITY = False                             # clip the polarity target to 0..1 (tried for v2; off = original training)
SEED = 0
CHECKPOINT = "phase3_unet_checkpoint.pt"          # rename the old checkpoint first; this script will not overwrite a file
# =========================================================================================================================================

CHECK_ONLY = "--check" in sys.argv[1:]
if not CHECK_ONLY and os.path.exists(CHECKPOINT):
    sys.exit(f"{CHECKPOINT} already exists. Rename the old checkpoint first (or set CHECKPOINT to a new name); this script will not overwrite it.")
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (training on CPU is very slow)"))


def class_focal_loss(pred_logits, target, mask=None, gamma=2.0, alpha=0.5):
    """Focal loss: concentrates on voxels the network still gets wrong."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    loss = alpha * (1 - torch.exp(-bce)) ** gamma * bce
    if mask is not None:
        m = mask.expand_as(loss)
        return (loss * m).sum() / (m.sum() + 1e-8)
    return loss.mean()


class PatchDataset(Dataset):
    def __init__(self, augment=True):
        self.patches = []; self.augment = augment

    def add(self, vol, vecs, cls, weight):
        D, H, W = vol.shape; n = 0
        for z in range(0, D - PATCH + 1, STRIDE):
            for y in range(0, H - PATCH + 1, STRIDE):
                for x in range(0, W - PATCH + 1, STRIDE):
                    w = weight[z:z + PATCH, y:y + PATCH, x:x + PATCH]
                    if np.mean(w > 0.9) <= 0.05: continue                     # skip patches that are nearly all background
                    s = (slice(z, z + PATCH), slice(y, y + PATCH), slice(x, x + PATCH))
                    self.patches.append((torch.from_numpy(vol[s]).unsqueeze(0).float(), torch.from_numpy(vecs[(slice(None),) + s]).float(),
                                         torch.from_numpy(cls[(slice(None),) + s]).float(), torch.from_numpy(w).unsqueeze(0).float()))
                    n += 1
        return n

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        v, vec, cls, w = [t.clone() for t in self.patches[i]]
        if not self.augment: return v, vec, cls, w
        if SWAP_YX and torch.rand(1).item() > 0.5:                            # quarter turn around the scroll axis: swap y and x
            v, vec, cls, w = (t.transpose(2, 3).contiguous() for t in (v, vec, cls, w)); vec = vec[[0, 2, 1]]
        for ax in (1, 2, 3):                                                   # random flips; the direction flips with the data
            if torch.rand(1).item() > 0.5:
                v, vec, cls, w = (torch.flip(t, [ax]) for t in (v, vec, cls, w)); vec[ax - 1] = -vec[ax - 1]
        return v, vec, cls, w


def targets(vol):
    """structure-tensor targets for one block: fibre direction, the four class targets, and the loss weights"""
    _, evecs, s, polarity = compute_structure_tensor(vol, **TARGET_TENSOR)
    t = np.stack([evecs[..., 0, 2], evecs[..., 1, 2], evecs[..., 2, 2]]).astype(np.float32)   # fibre direction (z, y, x)
    along = np.abs(t[0]); around = np.sqrt(t[1] ** 2 + t[2] ** 2 + 1e-12); tot = along + around + 1e-12
    cls = np.stack([s, s * around / tot, s * along / tot, (np.clip(polarity, 0.0, 1.0) if CLIP_POLARITY else polarity)]).astype(np.float32)   # sheet, recto, verso, polarity
    pap = (s > PAPYRUS_THRESHOLD).astype(np.float32)
    edge = (F.max_pool3d(torch.from_numpy(pap)[None, None], 5, 1, 2)[0, 0].numpy() - pap) == 1.0
    weight = np.full_like(pap, BACKGROUND_WEIGHT); weight[pap == 1.0] = PAPYRUS_WEIGHT; weight[edge] = EDGE_WEIGHT
    return t, cls, weight, pap


# ---------------------------------------------------------------- build the training set (and the held-out set)
print("\n[1] Building the training set" + (" (check only: no training)" if CHECK_ONLY else ""))
data = PatchDataset(augment=True); held = PatchDataset(augment=False); streamers = {}; report = []
ALL = SOURCES + ([VALIDATION] if VALIDATION else [])
for k, src in enumerate(ALL, 1):
    is_val = src is VALIDATION
    print(f"\n[{k}/{len(ALL)}] {src['name']}" + ("   (HELD OUT: measured, never trained on)" if is_val else ""))
    key = (src["scroll"], src["scan"])
    try:
        if key not in streamers: streamers[key] = ScrollStreamer(scroll_name=src["scroll"], scan_folder=src["scan"], level=LEVEL)
        vol = streamers[key].fetch_chunk(*src["z"], *src["y"], *src["x"])
    except Exception as e:
        print(f"  skipped: {e}"); report.append((src["name"], "skipped", 0, 0.0, 0.0)); continue
    t, cls, weight, pap = targets(vol)
    n = (held if is_val else data).add(vol, t, cls, weight)
    scanned = float((vol > 0).mean()); papyrus = float(pap.mean())
    report.append((src["name"] + (" [held out]" if is_val else ""), "ok", n, scanned, papyrus))
    print(f"  {n} patches | inside the scan mask {scanned:.0%} | papyrus (sheet measure > {PAPYRUS_THRESHOLD}) {papyrus:.0%} | "
          f"verso share of papyrus {float(cls[2][pap > 0].mean() / max(cls[0][pap > 0].mean(), 1e-6)) if pap.any() else 0:.0%}")

print("\n  block                                                        status  patches  in scan  papyrus")
for name, st, n, sc, pp in report:
    flag = "   <- little or no papyrus: move this block" if st == "ok" and (n < 5 or pp < 0.05) else ""
    print(f"  {name:<60} {st:>6} {n:>8} {sc:>8.0%} {pp:>8.0%}{flag}")
print(f"  total: {len(data)} training patches, {len(held)} held-out patches")
if CHECK_ONLY: sys.exit(0)
if not len(data): raise SystemExit("no training patches; check the sources")
loader = DataLoader(data, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED))
held_loader = DataLoader(held, batch_size=BATCH, shuffle=False) if len(held) else None
print(f"\n{len(data)} patches, {len(loader)} batches per epoch" + (f" | held out: {len(held)} patches" if len(held) else " | no held-out patches"))

# ---------------------------------------------------------------- train
model = Phase2StageBUNet3D(in_channels=1).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

def losses(v, vec, cls, w):
    with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        pred_vec, pred_cls = model(v)
        l_dir = nematic_sign_invariant_loss(pred_vec, vec, (w[:, 0] == PAPYRUS_WEIGHT).float())   # papyrus voxels only
        l_cls = class_focal_loss(pred_cls, cls, w)
    return l_dir, l_cls

def held_out_loss():
    """on the held-out block: direction loss, class loss, and the plain error of the sheet / recto / verso outputs on papyrus
    (mean |output - target|, both 0..1), next to the error of always answering the block's average (a model that learned nothing)"""
    if held_loader is None: return None
    model.eval(); d = c = 0.0; err = torch.zeros(3, device=device); base = torch.zeros(3, device=device); n = 0.0
    with torch.no_grad():
        for v, vec, cls, w in held_loader:
            v, vec, cls, w = (t.to(device) for t in (v, vec, cls, w))
            l_dir, l_cls = losses(v, vec, cls, w); d += l_dir.item(); c += l_cls.item()
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                _, pred_cls = model(v)
            m = (w > 0.9).float()                                              # papyrus and its edge
            out = torch.sigmoid(pred_cls[:, :3].float()); tgt = cls[:, :3]
            err += ((out - tgt).abs() * m).sum(dim=(0, 2, 3, 4)); base += ((HELD_MEAN.view(1, 3, 1, 1, 1) - tgt).abs() * m).sum(dim=(0, 2, 3, 4)); n += m.sum().item()
    model.train()
    return dict(direction=d / len(held_loader), cls=c / len(held_loader), error=(err / max(n, 1.0)).tolist(), baseline=(base / max(n, 1.0)).tolist())

HELD_MEAN = torch.zeros(3, device=device)
if held_loader is not None:                                                    # the held-out block's average sheet / recto / verso target on papyrus
    tot = torch.zeros(3); cnt = 0.0
    for _, _, cls, w in held.patches:
        m = (w > 0.9).float(); tot += (cls[:3] * m).sum(dim=(1, 2, 3)); cnt += m.sum().item()
    HELD_MEAN = (tot / max(cnt, 1.0)).to(device)

print(f"\n[2] Training for {EPOCHS} epochs"); t0 = time.time(); history = []
for ep in range(1, EPOCHS + 1):
    model.train(); l_dir_sum = l_cls_sum = 0.0; te = time.time()
    for v, vec, cls, w in loader:
        v, vec, cls, w = v.to(device), vec.to(device), cls.to(device), w.to(device)
        opt.zero_grad()
        l_dir, l_cls = losses(v, vec, cls, w); loss = DIRECTION_WEIGHT * l_dir + l_cls
        if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
        l_dir_sum += l_dir.item(); l_cls_sum += l_cls.item()
    sched.step()
    if ep == 1 or ep % 10 == 0:
        h = held_out_loss(); history.append(dict(epoch=ep, train_direction=l_dir_sum / len(loader), train_class=l_cls_sum / len(loader), held_out=h))
        held_txt = "held out: off" if h is None else (f"held out: direction {h['direction']:.4f}, class {h['cls']:.4f} | held-out error sheet / recto / verso "
                   f"{h['error'][0]:.3f} / {h['error'][1]:.3f} / {h['error'][2]:.3f} (answering the average: {h['baseline'][0]:.3f} / {h['baseline'][1]:.3f} / {h['baseline'][2]:.3f})")
        print(f"  epoch {ep:3d}/{EPOCHS} | training: direction {l_dir_sum / len(loader):.4f}, class {l_cls_sum / len(loader):.4f} | {held_txt}"
              + (f" | first epoch took {time.time() - te:.0f}s" if ep == 1 else ""))
print(f"  trained in {time.time() - t0:.0f}s")

torch.save(dict(epoch=EPOCHS, model_state_dict=model.state_dict(), history=history,
                settings=dict(sources=SOURCES, extra_blocks=EXTRA_BLOCKS, validation=VALIDATION, level=LEVEL, target_tensor=TARGET_TENSOR, patch=PATCH, stride=STRIDE,
                              epochs=EPOCHS, batch=BATCH, lr=LR, weight_decay=WEIGHT_DECAY, direction_weight=DIRECTION_WEIGHT,
                              swap_yx=SWAP_YX, polarity_clipped=CLIP_POLARITY, seed=SEED)), CHECKPOINT)
print(f"saved {CHECKPOINT}")
