
"""phase3_unet.py - the phase 3 U-Net: its architecture, the loss used to train it, and running it on a tile.

This file defines the network. The trained weights are in phase3_unet_checkpoint.pt (made by train_phase3_unet.py).

A small 3D U-Net (three levels, 16-32-64 channels, 128 at the bottom) with two outputs per voxel:
    direction head  a unit vector (sign-free): the fibre direction            (not used by the pipeline)
    class head      four logits: sheet, recto, verso, polarity                (sheet, recto, verso are used)
It is trained on targets computed by the structure tensor itself (no human labels): see train_phase3_unet.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """(3x3x3 convolution -> BatchNorm -> ReLU) twice."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(out_c), nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class Phase2StageBUNet3D(nn.Module):
    """3D U-Net trained on 64^3 patches; any input whose sides are multiples of 8 works (see predict_volume)."""
    def __init__(self, in_channels=1):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, 16); self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(16, 32);          self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(32, 64);          self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(64, 128)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2); self.dec3 = DoubleConv3D(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2);  self.dec2 = DoubleConv3D(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2);  self.dec1 = DoubleConv3D(32, 16)
        self.head_nematic = nn.Conv3d(16, 3, kernel_size=1)     # fibre direction (sign-free unit vector)
        self.head_class = nn.Conv3d(16, 4, kernel_size=1)       # logits: sheet, recto, verso, polarity

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return F.normalize(self.head_nematic(d1), p=2, dim=1), self.head_class(d1)


def load_unet(checkpoint_path, device=None):
    """The trained network from a checkpoint file, on the GPU if there is one, ready to use."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Phase2StageBUNet3D(in_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.eval()
    return model, device


def predict_volume(model, vol, device, slab=64):
    """Sheet / recto / verso / polarity probabilities, (4, z, y, x) float32, for a whole tile (numpy, 0..1 CT).
    The tile is run in slabs of `slab` slices along z. Sides that are not multiples of 8 are padded by reflection and the
    padding removed afterwards; for tiles whose sides are already multiples of 8 (such as 192 x 416 x 416) the result is
    exactly what the pipeline has always produced."""
    import numpy as np
    D, H, W = vol.shape
    ph, pw = (-H) % 8, (-W) % 8
    out = np.zeros((4, D, H, W), dtype=np.float32)
    with torch.no_grad():
        for z0 in range(0, D, slab):
            z1 = min(z0 + slab, D); block = vol[z0:z1]
            pd = (-(z1 - z0)) % 8
            if pd or ph or pw:
                block = np.pad(block, ((0, pd), (0, ph), (0, pw)), mode="reflect")
            x = torch.from_numpy(np.ascontiguousarray(block)).unsqueeze(0).unsqueeze(0).float().to(device)
            _, logits = model(x)
            out[:, z0:z1] = torch.sigmoid(logits).squeeze(0).cpu().numpy()[:, :z1 - z0, :H, :W]
    return out


def nematic_sign_invariant_loss(pred_vec, target_vec, mask=None):
    """Direction loss 1 - (pred . target)^2, so v and -v count as the same direction."""
    loss = 1.0 - torch.sum(pred_vec * target_vec, dim=1) ** 2
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()
