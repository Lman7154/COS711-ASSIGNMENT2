"""
Filament segmentation pipeline (COS711 A2, milestone 2: experimental setup).

Reusable building blocks for every experiment, so each run differs only by a Config, not by copy-pasted
code: cached data loading from data_prep/prepared/, models (U-Net family, FCN, DeepLabV3+, PSPNet, FPN),
disk-masked losses, training with early stopping + LR-on-plateau, Panoptic Quality validation in the
ORIGINAL image frame (identical to Kaggle's scoring), sliding-window inference with Gaussian blending and
optional flip TTA, instance extraction, diagnostics, and submission.csv writing.

Grounded in the data-prep findings (data_prep/DATA_PREP_DECISIONS.md):
  - native resolution, 512 crops, sliding-window inference    (PQ ceiling 0.70 at 512)
  - filament-centred crops p_fg=0.7, other crops on the disk  (278:1 class imbalance)
  - one random annotation set per image per sample            (annotators disagree, human PQ ~0.34)
  - validate against ALL annotation sets, pooled              (how Kaggle scores)
  - masks drawn with pycocotools                              (identical to Kaggle GT)

Audit fixes (see EXPERIMENTAL_SETUP_UNDERSTANDING.md, "Audit"):
  1. validation maps predictions back to the original frame before scoring (was: canonical vs original,
     which capped a perfect prediction at PQ 0.864)
  2. connected components use 8-connectivity (4-connectivity split thin diagonal filaments)
  3. the early-stopping "monitor" images are spread over the whole fold in time (was: first 40 = 2011-14)
  4. background crops are sampled on the solar disk (was: 36% centred on the sky)
  5. LR is halved when validation PQ stalls (was: cosine over the cap, so it never annealed)
  6. the loss ignores pixels off the solar disk (stated in the methodology, now implemented)
  7. selection (monitor images) and reporting (report images) use DISJOINT parts of the validation fold
"""
from __future__ import annotations
import json, math, time, random
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_util
from scipy import ndimage

IMG_SIZE = 2048
EIGHT = np.ones((3, 3), bool)                      # 8-connectivity structuring element
MCOLS = [f"M{i}" for i in range(6)]


# --------------------------------------------------------------------------------------------------
# 0. Config
# --------------------------------------------------------------------------------------------------
@dataclass
class Config:
    prepared_dir: str = "../data_prep/prepared"
    cache_dir: str = "cache"
    out_dir: str = "runs"
    name: str = "baseline"
    # data
    patch: int = 512
    p_fg: float = 0.7
    val_fold: int = 0
    samples_per_epoch: int = 800    # one "epoch" = this many random crops
    batch_size: int = 8
    num_workers: int = 0            # 0 in notebooks on Windows; 4 in scripts
    augment: bool = True
    disk_loss: bool = True          # ignore off-disk pixels in the loss
    # model
    arch: str = "smp_unet"          # unet | fcn32 | fcn8 | smp_unet | smp_unetpp | smp_fpn | smp_deeplabv3p | smp_pspnet
    encoder: str = "resnet34"
    base_ch: int = 32               # width of the from-scratch "unet"
    # loss
    loss: str = "bce_dice"          # bce_dice | dice | tversky | focal_dice
    tversky_beta: float = 0.7
    # optimisation
    epochs: int = 30                # MAXIMUM epochs; early stopping usually stops sooner
    patience: int = 5               # stop after this many validations without a min_delta gain (0 = off)
    min_delta: float = 0.002
    lr: float = 3e-4
    lr_patience: int = 2            # halve LR after this many validations without improvement
    weight_decay: float = 1e-4
    amp: bool = True
    monitor_images: int = 40        # validation images used for early stopping / tuning (spread in time)
    # inference / post-processing
    threshold: float = 0.5
    low_threshold: float | None = None   # hysteresis: grow from >=threshold seeds into >=low_threshold
    close_px: int = 0               # morphological closing radius before splitting (0 = off)
    min_area: int = 60              # drop predicted blobs smaller than this (px)
    window_overlap: float = 0.5
    tta: bool = False               # average 4 flips at inference
    infer_batch: int = 8
    seed: int = 42

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _canon_geometry(cfg: Config):
    c = json.loads((Path(cfg.prepared_dir) / "config.json").read_text())
    return (IMG_SIZE - 1) / 2, float(c["canon_r"]), float(c.get("disk_margin", 1.02))


# --------------------------------------------------------------------------------------------------
# 1. Masks, PQ, and diagnostics
# --------------------------------------------------------------------------------------------------
def ann_to_instmap_from_rle(rles, shape=(IMG_SIZE, IMG_SIZE)):
    """List of RLE-count strings -> uint16 instance map (0 = background, 1..N)."""
    inst = np.zeros(shape, np.uint16)
    for k, counts in enumerate(rles, start=1):
        m = mask_util.decode({"size": [shape[0], shape[1]], "counts": counts.encode("ascii")})
        inst[m > 0] = k
    return inst


def encode_mask(binary_mask):
    """Binary mask (HxW) -> RLE counts string for submission.csv."""
    return mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))["counts"].decode("ascii")


def panoptic_quality(gt_inst, pred_inst, iou_thr=0.5):
    """Class-agnostic PQ counts for one image (pooled later, like the organisers' get_pq_score).
    Also counts fragmentation (a GT overlapped by >1 prediction) and merging (a prediction overlapping
    >1 GT), the one-to-many / many-to-one relations the competition reports."""
    gt_inst = gt_inst.astype(np.int64); pred_inst = pred_inst.astype(np.int64)
    gids, gareas = np.unique(gt_inst[gt_inst > 0], return_counts=True)
    pids, pareas = np.unique(pred_inst[pred_inst > 0], return_counts=True)
    out = dict(tp=0, fp=len(pids), fn=len(gids), iou_sum=0.0, frag_gt=0, merge_pred=0)
    if len(gids) == 0 or len(pids) == 0:
        return out
    ga = dict(zip(gids, gareas)); pa = dict(zip(pids, pareas))
    both = (gt_inst > 0) & (pred_inst > 0)
    base = int(pred_inst.max()) + 1
    pairs, inter = np.unique(gt_inst[both] * base + pred_inst[both], return_counts=True)
    gi_all, pj_all = np.divmod(pairs, base)
    tp = 0; iou_sum = 0.0
    for gi, pj, it in zip(gi_all, pj_all, inter):          # IoU > 0.5 matches are unique
        iou = it / (ga[gi] + pa[pj] - it)
        if iou > iou_thr:
            tp += 1; iou_sum += iou
    out.update(tp=tp, fp=len(pids) - tp, fn=len(gids) - tp, iou_sum=float(iou_sum),
               frag_gt=int((np.bincount(gi_all) > 1).sum()), merge_pred=int((np.bincount(pj_all) > 1).sum()))
    return out


def pq_from_counts(counts_list):
    keys = ["tp", "fp", "fn", "iou_sum", "frag_gt", "merge_pred"]
    s = {k: sum(c.get(k, 0) for c in counts_list) for k in keys}
    denom = s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]
    return dict(pq=s["iou_sum"] / denom if denom else 0.0, sq=s["iou_sum"] / s["tp"] if s["tp"] else 0.0,
                rq=s["tp"] / denom if denom else 0.0, tp=s["tp"], fp=s["fp"], fn=s["fn"],
                frag_gt=s["frag_gt"], merge_pred=s["merge_pred"])


def official_pq(gt_df, pred_df):
    """Numpy port of the organisers' self-evaluation get_overlap_df + get_pq_score (for cross-checks).
    gt_df: filament_id = <set_id>_<k>; pred_df: filament_id = <image stem>_<k>; both with segmentation_rle."""
    dec = lambda r: mask_util.decode({"size": [IMG_SIZE, IMG_SIZE], "counts": r.encode("ascii")}).ravel()
    gk = gt_df.filament_id.str.split("_", n=1).str[0]
    pk = pred_df.filament_id.str.rsplit("_", n=1).str[0]
    tp_ious, fp, fn = [], 0, 0
    for ann_img in gk.unique():
        stem = ann_img.split("-", 1)[1]
        g = np.stack([dec(r) for r in gt_df.segmentation_rle[gk == ann_img]]).astype(np.float32)
        pr = pred_df.segmentation_rle[pk == stem]
        if len(pr) == 0:
            fn += len(g); continue
        p = np.stack([dec(r) for r in pr]).astype(np.float32)
        inter = g @ p.T
        union = g.sum(1)[:, None] + p.sum(1)[None] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0)
        hit = iou > 0.5
        tp_ious += iou[hit].tolist(); fp += int((hit.sum(0) == 0).sum()); fn += int((hit.sum(1) == 0).sum())
    d = len(tp_ious) + 0.5 * fp + 0.5 * fn
    return sum(tp_ious) / d if d else 0.0


# --------------------------------------------------------------------------------------------------
# 2. Cache (fast loading) and dataset
# --------------------------------------------------------------------------------------------------
def build_cache(cfg: Config, verbose=True):
    """One-off: decode the prepared PNGs into memory-mappable .npy files + filament pixel indices.
    Crops then read only the ~0.8 MP window they need instead of decoding a 4 MP PNG per sample."""
    root, cache = Path(cfg.prepared_dir), Path(cfg.cache_dir)
    done = cache / "done.json"
    if done.exists():
        return
    (cache / "img").mkdir(parents=True, exist_ok=True); (cache / "inst").mkdir(parents=True, exist_ok=True)
    t = time.time()
    for p in sorted((root / "images").glob("*.png")):
        np.save(cache / "img" / f"{p.stem}.npy", cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
    offsets, chunks, pos = {}, [], 0
    for p in sorted((root / "instances").glob("*.png")):
        inst = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert inst.max() < 256, p
        np.save(cache / "inst" / f"{p.stem}.npy", inst.astype(np.uint8))
        fg = np.flatnonzero(inst).astype(np.int32)
        offsets[p.stem] = [pos, pos + len(fg)]; chunks.append(fg); pos += len(fg)
    np.save(cache / "fg_index.npy", np.concatenate(chunks))
    (cache / "fg_offsets.json").write_text(json.dumps(offsets))
    done.write_text(json.dumps({"seconds": round(time.time() - t)}))
    if verbose:
        print(f"cache built in {time.time() - t:.0f}s")


def load_image(cfg: Config, stem):
    return np.load(Path(cfg.cache_dir) / "img" / f"{stem}.npy", mmap_mode="r")


def _instance_boundaries(inst):
    k = np.ones((3, 3), np.uint8)
    mx = cv2.dilate(inst, k); mn = cv2.erode(inst, k)
    return ((mx != mn) & (mx > 0)).astype(np.uint8)


def _patch_affine(center, patch, angle=0.0, scale=1.0, fh=False, fv=False):
    t = math.radians(angle)
    A = scale * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    A = A @ np.diag([-1.0 if fh else 1.0, -1.0 if fv else 1.0])
    half = (patch - 1) / 2
    b = np.array([half, half]) - A @ np.asarray(center, float)
    return np.hstack([A, b[:, None]]).astype(np.float64)


def _photometric(img, rng):
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    img = (img - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.05, 0.05)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(0.005, 0.03), img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def _warp_window(src, M, patch, interp):
    """warpAffine(src, M) but reading only the source window the patch needs (works on memmaps)."""
    A, b = M[:, :2], M[:, 2]
    Ainv = np.linalg.inv(A)
    corners = np.array([[0, 0], [patch - 1, 0], [0, patch - 1], [patch - 1, patch - 1]], float)
    srcpts = (corners - b) @ Ainv.T
    x0 = max(int(np.floor(srcpts[:, 0].min())) - 2, 0); x1 = min(int(np.ceil(srcpts[:, 0].max())) + 3, src.shape[1])
    y0 = max(int(np.floor(srcpts[:, 1].min())) - 2, 0); y1 = min(int(np.ceil(srcpts[:, 1].max())) + 3, src.shape[0])
    if x1 <= x0 or y1 <= y0:
        return np.zeros((patch, patch), src.dtype)
    win = np.ascontiguousarray(src[y0:y1, x0:x1])
    Mw = np.hstack([A, (b + A @ np.array([x0, y0], float))[:, None]]).astype(np.float32)
    return cv2.warpAffine(win, Mw, (patch, patch), flags=interp, borderValue=0)


class FilamentPatchDataset(Dataset):
    """Training crops: random observation, ONE random annotation set, filament-centred crop with
    probability p_fg (otherwise a random point ON the disk), random rotation/flip/scale + photometric."""
    def __init__(self, cfg: Config, folds, augment=True):
        build_cache(cfg, verbose=False)
        self.cfg = cfg; self.cache = Path(cfg.cache_dir)
        obs = pd.read_csv(Path(cfg.prepared_dir) / "train_observations.csv")
        self.obs = obs[obs.fold.isin(folds)].reset_index(drop=True)
        self.augment = augment
        self.c0, self.R, self.margin = _canon_geometry(cfg)
        self._fg = None; self._off = None

    def __len__(self):
        return self.cfg.samples_per_epoch

    def _fg_idx(self, sid):
        if self._fg is None:                                    # lazily, once per worker process
            self._fg = np.load(self.cache / "fg_index.npy", mmap_mode="r")
            self._off = json.loads((self.cache / "fg_offsets.json").read_text())
        a, b = self._off[sid]
        return self._fg, a, b

    def __getitem__(self, i):
        cfg = self.cfg; P = cfg.patch
        rng = np.random.default_rng() if self.augment else np.random.default_rng((cfg.seed, i))
        o = self.obs.iloc[int(rng.integers(len(self.obs)))]
        sid = rng.choice(o.set_ids.split("|"))
        fg, a, b = self._fg_idx(sid)
        if b > a and rng.random() < cfg.p_fg:
            y, x = divmod(int(fg[int(rng.integers(a, b))]), IMG_SIZE)
            c = (x + rng.uniform(-P / 4, P / 4), y + rng.uniform(-P / 4, P / 4))
        else:                                                   # uniform point on the solar disk
            r = self.R * math.sqrt(rng.random()); th = rng.uniform(0, 2 * math.pi)
            c = (self.c0 + r * math.cos(th), self.c0 + r * math.sin(th))
        if self.augment:
            M = _patch_affine(c, P, rng.uniform(0, 360), rng.uniform(0.95, 1.05), rng.random() < 0.5, rng.random() < 0.5)
        else:
            M = _patch_affine(c, P)
        img = np.load(self.cache / "img" / f"{o.obs}.npy", mmap_mode="r")
        inst = np.load(self.cache / "inst" / f"{sid}.npy", mmap_mode="r")
        im = _warp_window(img, M, P, cv2.INTER_LINEAR).astype(np.float32) / 255.0
        ins = _warp_window(inst, M, P, cv2.INTER_NEAREST)
        # disk mask of this crop (same affine), so the loss can ignore the sky
        Minv = cv2.invertAffineTransform(M.astype(np.float32))
        uu, vv = np.meshgrid(np.arange(P, dtype=np.float32), np.arange(P, dtype=np.float32))
        xs = Minv[0, 0] * uu + Minv[0, 1] * vv + Minv[0, 2]; ys = Minv[1, 0] * uu + Minv[1, 1] * vv + Minv[1, 2]
        disk = ((xs - self.c0) ** 2 + (ys - self.c0) ** 2 <= (self.R * self.margin) ** 2).astype(np.float32)
        if self.augment:
            im = _photometric(im, rng)
        sem = (ins > 0).astype(np.float32)
        bnd = _instance_boundaries(ins).astype(np.float32)
        return (torch.from_numpy(im)[None], torch.from_numpy(sem)[None],
                torch.from_numpy(bnd)[None], torch.from_numpy(disk)[None])


# --------------------------------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small from-scratch U-Net (no pretraining). base controls width."""
    def __init__(self, base=32, out_ch=1):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.d1 = _DoubleConv(1, c[0]); self.d2 = _DoubleConv(c[0], c[1])
        self.d3 = _DoubleConv(c[1], c[2]); self.d4 = _DoubleConv(c[2], c[3])
        self.bott = _DoubleConv(c[3], c[4]); self.pool = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2); self.c4 = _DoubleConv(c[4], c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2); self.c3 = _DoubleConv(c[3], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2); self.c2 = _DoubleConv(c[2], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2); self.c1 = _DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(self.pool(x1)); x3 = self.d3(self.pool(x2)); x4 = self.d4(self.pool(x3))
        b = self.bott(self.pool(x4))
        y = self.c4(torch.cat([self.u4(b), x4], 1)); y = self.c3(torch.cat([self.u3(y), x3], 1))
        y = self.c2(torch.cat([self.u2(y), x2], 1)); y = self.c1(torch.cat([self.u1(y), x1], 1))
        return self.head(y)


class FCN(nn.Module):
    """Bare-backbone CNN for segmentation (FCN, Long et al. 2015): pretrained classifier backbone, 1x1
    score conv, upsample. skips=0 -> FCN-32s (bare CNN); skips=2 -> FCN-8s (+ stride-16/8 scores)."""
    def __init__(self, encoder="resnet34", skips=0):
        super().__init__()
        from segmentation_models_pytorch.encoders import get_encoder
        self.enc = get_encoder(encoder, in_channels=1, depth=5, weights="imagenet")
        ch = self.enc.out_channels; self.skips = skips
        self.score = nn.ModuleList([nn.Conv2d(ch[-1 - i], 1, 1) for i in range(skips + 1)])

    def forward(self, x):
        H, W = x.shape[-2:]; f = self.enc(x)
        y = self.score[0](f[-1])
        for i in range(1, self.skips + 1):
            s = self.score[i](f[-1 - i])
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False) + s
        return F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)


def build_model(cfg: Config):
    if cfg.arch == "unet":
        return UNet(base=cfg.base_ch)
    if cfg.arch in ("fcn32", "fcn8"):
        return FCN(cfg.encoder, skips=0 if cfg.arch == "fcn32" else 2)
    import segmentation_models_pytorch as smp
    kw = dict(encoder_name=cfg.encoder, encoder_weights="imagenet", in_channels=1, classes=1)
    return {"smp_unet": smp.Unet, "smp_unetpp": smp.UnetPlusPlus, "smp_fpn": smp.FPN,
            "smp_deeplabv3p": smp.DeepLabV3Plus, "smp_pspnet": smp.PSPNet}[cfg.arch](**kw)


def load_model(cfg: Config, ckpt, device):
    m = build_model(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    return m


# --------------------------------------------------------------------------------------------------
# 4. Losses (disk-masked; all handle the 278:1 imbalance)
# --------------------------------------------------------------------------------------------------
def _w(target, mask):
    return torch.ones_like(target) if mask is None else mask


def bce_loss(logits, target, mask=None):
    w = _w(target, mask)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum() / w.sum().clamp(min=1)


def dice_loss(logits, target, mask=None, eps=1.0):
    w = _w(target, mask); p = torch.sigmoid(logits) * w; t = target * w
    return (1 - (2 * (p * t).sum((2, 3)) + eps) / (p.sum((2, 3)) + t.sum((2, 3)) + eps)).mean()


def tversky_loss(logits, target, mask=None, beta=0.7, eps=1.0):
    w = _w(target, mask); p = torch.sigmoid(logits) * w; t = target * w
    tp = (p * t).sum((2, 3)); fp = (p * (1 - t)).sum((2, 3)); fn = ((1 - p) * t * w).sum((2, 3))
    return (1 - (tp + eps) / (tp + (1 - beta) * fp + beta * fn + eps)).mean()


def focal_loss(logits, target, mask=None, gamma=2.0, alpha=0.75):
    w = _w(target, mask); p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target); a = alpha * target + (1 - alpha) * (1 - target)
    return (a * (1 - pt) ** gamma * ce * w).sum() / w.sum().clamp(min=1)


def build_loss(cfg: Config):
    if cfg.loss == "dice":
        return lambda o, t, m: dice_loss(o, t, m)
    if cfg.loss == "tversky":
        return lambda o, t, m: tversky_loss(o, t, m, cfg.tversky_beta)
    if cfg.loss == "focal_dice":
        return lambda o, t, m: focal_loss(o, t, m) + dice_loss(o, t, m)
    return lambda o, t, m: bce_loss(o, t, m) + dice_loss(o, t, m)


# --------------------------------------------------------------------------------------------------
# 5. Inference: Gaussian-blended sliding window (+ optional flip TTA) and instance extraction
# --------------------------------------------------------------------------------------------------
def _gauss_window(P):
    g = np.exp(-0.5 * ((np.arange(P) - (P - 1) / 2) / (P / 4)) ** 2)
    w = np.outer(g, g).astype(np.float32)
    return np.maximum(w / w.max(), 1e-3)


@torch.no_grad()
def predict_prob(model, image, cfg: Config, device):
    """image: HxW uint8 or float in canonical frame. Returns HxW float32 filament probability.
    Overlapping tiles are blended with a Gaussian weight (tile centres count more than tile edges, which
    have less context); tiles that are entirely sky are skipped."""
    model.eval()
    img = np.asarray(image, np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    H, W = img.shape; P = cfg.patch; step = max(1, int(P * (1 - cfg.window_overlap)))
    ys = list(range(0, H - P + 1, step)); xs = list(range(0, W - P + 1, step))
    if ys[-1] != H - P: ys.append(H - P)
    if xs[-1] != W - P: xs.append(W - P)
    coords = [(y, x) for y in ys for x in xs if img[y:y + P, x:x + P].max() > 0]
    gw = _gauss_window(P)
    prob = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    for k in range(0, len(coords), cfg.infer_batch):
        cb = coords[k:k + cfg.infer_batch]
        t = torch.from_numpy(np.stack([img[y:y + P, x:x + P] for y, x in cb]))[:, None].to(device)
        with torch.autocast(device.type, enabled=cfg.amp):
            p = torch.sigmoid(model(t)).float()
            if cfg.tta:
                for dims in ([3], [2], [2, 3]):
                    p = p + torch.sigmoid(model(torch.flip(t, dims))).float().flip(dims)
                p = p / 4
        p = p[:, 0].cpu().numpy()
        for (y, x), pi in zip(cb, p):
            prob[y:y + P, x:x + P] += pi * gw; wsum[y:y + P, x:x + P] += gw
    prob = np.where(wsum > 0, prob / np.maximum(wsum, 1e-6), 0).astype(np.float32)
    # The loss ignores off-disk pixels (disk_loss), so the model's output on the sky is meaningless (it is
    # often ~1). Zero it, exactly matching the region the model was trained on. No annotation lies off the
    # disk (data prep: 0% of filament pixels outside the limb), so this can only remove false filaments.
    prob[~_disk_mask(cfg, H, W)] = 0
    return prob


_DISK_CACHE = {}


def _disk_mask(cfg: Config, H=IMG_SIZE, W=IMG_SIZE):
    c0, R, margin = _canon_geometry(cfg)
    key = (H, W, c0, R, margin)
    if key not in _DISK_CACHE:
        yy, xx = np.ogrid[:H, :W]
        _DISK_CACHE[key] = (xx - c0) ** 2 + (yy - c0) ** 2 <= (R * margin) ** 2
    return _DISK_CACHE[key]


def prob_to_instances(prob, cfg: Config):
    """Probability map -> uint16 instance map: threshold (optionally with hysteresis), optional closing,
    8-connected components, then drop blobs smaller than min_area."""
    if cfg.low_threshold is not None and cfg.low_threshold < cfg.threshold:
        low = prob >= cfg.low_threshold
        lab, n = ndimage.label(low, structure=EIGHT)
        seeds = np.unique(lab[(prob >= cfg.threshold) & low])
        keep = np.zeros(n + 1, bool); keep[seeds[seeds > 0]] = True
        mask = keep[lab]
    else:
        mask = prob >= cfg.threshold
    if cfg.close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.close_px + 1, 2 * cfg.close_px + 1))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    lab, n = ndimage.label(mask, structure=EIGHT)
    if n == 0:
        return lab.astype(np.uint16)
    sizes = np.bincount(lab.ravel())[1:]
    ok = sizes >= cfg.min_area
    remap = np.zeros(n + 1, np.uint16); remap[1:][ok] = np.arange(1, ok.sum() + 1)
    return remap[lab]


def to_original(inst_canon, M):
    """Map a canonical-frame instance map back to the original 2048x2048 image grid (nearest)."""
    return cv2.warpAffine(inst_canon, cv2.invertAffineTransform(np.asarray(M, np.float32).reshape(2, 3)),
                          (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST, borderValue=0)


# --------------------------------------------------------------------------------------------------
# 6. Validation (original frame; disjoint monitor / report subsets)
# --------------------------------------------------------------------------------------------------
_GT_CACHE = {}


def _gt_table(cfg):
    key = cfg.prepared_dir
    if key not in _GT_CACHE:
        gt = pd.read_csv(Path(cfg.prepared_dir) / "train_gt_rle.csv")
        gt["set_id"] = gt.filament_id.str.split("_", n=1).str[0]
        _GT_CACHE[key] = {s: d.segmentation_rle.tolist() for s, d in gt.groupby("set_id")}
    return _GT_CACHE[key]


def val_observations(cfg: Config, subset="all"):
    """Validation fold split into two DISJOINT parts:
       monitor - `monitor_images` observations evenly spread in time (early stopping, tuning)
       report  - all the others (the number we report, never used for any choice)"""
    obs = pd.read_csv(Path(cfg.prepared_dir) / "train_observations.csv")
    v = obs[obs.fold == cfg.val_fold].sort_values("time").reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(v) - 1, min(cfg.monitor_images, len(v))).round().astype(int))
    if subset == "monitor":
        return v.iloc[idx].reset_index(drop=True)
    if subset == "report":
        return v.drop(index=idx).reset_index(drop=True)
    return v


def predict_observations(model, cfg: Config, device, obs_df, dtype=np.float16):
    """Probability maps (canonical frame) for a set of observations, for re-scoring many settings."""
    build_cache(cfg, verbose=False)
    return {o.obs: predict_prob(model, load_image(cfg, o.obs), cfg, device).astype(dtype) for o in obs_df.itertuples()}


def score_probs(probs, obs_df, cfg: Config, details=False):
    """PQ of probability maps after post-processing, scored in the ORIGINAL frame against every set."""
    gts = _gt_table(cfg); counts = []; per_img = []
    for o in obs_df.itertuples():
        pred = to_original(prob_to_instances(np.asarray(probs[o.obs], np.float32), cfg), [getattr(o, c) for c in MCOLS])
        c_img = []
        for sid in o.set_ids.split("|"):
            c = panoptic_quality(ann_to_instmap_from_rle(gts[sid]), pred); counts.append(c); c_img.append(c)
        if details:
            m = pq_from_counts(c_img); m.update(obs=o.obs, site=o.site, n_pred=int(pred.max())); per_img.append(m)
    res = pq_from_counts(counts)
    return (res, pd.DataFrame(per_img)) if details else res


def validate_pq(model, cfg: Config, device, subset="monitor", max_images=None):
    obs_df = val_observations(cfg, subset)
    if max_images:
        obs_df = obs_df.iloc[:max_images]
    return score_probs(predict_observations(model, cfg, device, obs_df), obs_df, cfg)


# --------------------------------------------------------------------------------------------------
# 7. Training (early stopping on monitor PQ, LR halved on plateau, best weights returned)
# --------------------------------------------------------------------------------------------------
def train(cfg: Config, device=None, quick=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed); build_cache(cfg)
    run_dir = Path(cfg.out_dir) / cfg.name; run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    tr_ds = FilamentPatchDataset(cfg, [f for f in range(5) if f != cfg.val_fold], augment=cfg.augment)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                       drop_last=True, pin_memory=(device.type == "cuda"), persistent_workers=cfg.num_workers > 0)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=cfg.lr_patience,
                                                       threshold=cfg.min_delta, threshold_mode="abs", min_lr=1e-6)
    loss_fn = build_loss(cfg); scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp)
    history = []; best_pq = -1.0; best_epoch = -1; bad = 0; stopped_early = False; t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train(); run_loss = 0.0; nb = 0; te = time.time()
        for im, sem, bnd, disk in tr_dl:
            im, sem = im.to(device, non_blocking=True), sem.to(device, non_blocking=True)
            disk = disk.to(device, non_blocking=True) if cfg.disk_loss else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=cfg.amp):
                out = model(im)
            loss = loss_fn(out.float(), sem, disk)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item(); nb += 1
            if quick and nb >= 3:
                break
        train_s = time.time() - te
        m = validate_pq(model, cfg, device, "monitor", max_images=(4 if quick else None))
        sched.step(m["pq"])
        rec = dict(epoch=epoch + 1, train_loss=run_loss / max(nb, 1), lr=opt.param_groups[0]["lr"],
                   train_s=round(train_s, 1), **{f"val_{k}": v for k, v in m.items()})
        if m["pq"] > best_pq + cfg.min_delta or best_epoch < 0:
            best_pq, best_epoch, bad = m["pq"], epoch + 1, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
        history.append(rec); pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if quick:
            break
        if cfg.patience and bad >= cfg.patience:
            stopped_early = True; break
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    summary = dict(name=cfg.name, arch=cfg.arch, encoder=cfg.encoder, loss=cfg.loss, lr=cfg.lr, p_fg=cfg.p_fg,
                   seed=cfg.seed, val_fold=cfg.val_fold, params=int(n_params), monitor_pq=float(best_pq),
                   best_epoch=best_epoch, epochs_run=len(history), max_epochs=cfg.epochs,
                   stopped_early=stopped_early, crops_seen=len(history) * len(tr_dl) * cfg.batch_size,
                   train_minutes=round((time.time() - t0) / 60, 2), quick=quick)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return model, summary, pd.DataFrame(history)


# --------------------------------------------------------------------------------------------------
# 8. Submission
# --------------------------------------------------------------------------------------------------
def make_submission(model, cfg: Config, out_csv, device=None, max_images=None, models=None):
    """Writes submission.csv (one row per predicted filament, original 2048x2048 frame, RLE counts).
    `models`: optional list of models whose probabilities are averaged (e.g. a fold ensemble)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_cache(cfg, verbose=False)
    test = pd.read_csv(Path(cfg.prepared_dir) / "test_observations.csv")
    if max_images:
        test = test.iloc[:max_images]
    ms = models or [model]; rows = []
    for o in test.itertuples():
        img = load_image(cfg, o.obs)
        prob = sum(predict_prob(m, img, cfg, device) for m in ms) / len(ms)
        inst = to_original(prob_to_instances(prob, cfg), [getattr(o, c) for c in MCOLS])
        ids = np.unique(inst[inst > 0])
        for n, k in enumerate(ids, start=1):
            rows.append(dict(filament_id=f"{o.obs}_{n}", segmentation_rle=encode_mask(inst == k)))
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(out_csv, index=False)
    return sub
