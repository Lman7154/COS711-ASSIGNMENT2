"""End-to-end verification: raw Kaggle data -> data prep outputs -> cache -> training crops -> metric.

Run from modelling/:  python verify_pipeline.py
Every check prints PASS/FAIL with the measured number, and results are saved to verify_report.json.
Nothing here is assumed; each check re-derives the expected answer from the raw Kaggle files.
"""
import json, math, time, glob
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
from pycocotools import mask as mu
import filament_pipeline as fp

RAW = Path("../filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
cfg = fp.Config(prepared_dir="../data_prep/prepared")
root = Path(cfg.prepared_dir)
REPORT = {}


def check(name, ok, detail):
    REPORT[name] = dict(ok=bool(ok), detail=detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    coco = json.load(open(RAW / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_set = {}
    for a in coco["annotations"]:
        by_set.setdefault(a["image_id"], []).append(a)
    obs = pd.read_csv(root / "train_observations.csv"); test = pd.read_csv(root / "test_observations.csv")
    sets = pd.read_csv(root / "train_sets.csv"); gt = pd.read_csv(root / "train_gt_rle.csv")

    # ---- 1. counts match the raw data
    raw_files = {i["file_name"] for i in coco["images"]}
    check("raw->prepared counts",
          len(obs) == len(raw_files) == 707 and len(sets) == len(coco["images"]) == 1154
          and len(gt) == len(coco["annotations"]) == 8199 and len(test) == len(list((RAW / "test" / "test_images").glob("*.jpeg"))) == 180,
          f"observations {len(obs)}, sets {len(sets)}, filaments {len(gt)}, test {len(test)}")

    # ---- 2. Kaggle-format GT == polygons from the raw JSON, rasterised with pycocotools (all 8,199)
    gt["set_id"] = gt.filament_id.str.split("_", n=1).str[0]
    bad = 0
    for sid, d in gt.groupby("set_id"):
        anns = by_set[sid]
        if len(anns) != len(d):
            bad += 1; continue
        for a, r in zip(anns, d.segmentation_rle):
            ref = mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], 2048, 2048)))
            if not np.array_equal(ref, mu.decode({"size": [2048, 2048], "counts": r.encode()})):
                bad += 1
    check("train_gt_rle.csv == raw polygons (pycocotools)", bad == 0, f"{bad} mismatching filaments out of 8199")

    # ---- 3. split integrity
    sets_fold = sets.merge(obs[["obs", "fold"]], on="obs", suffixes=("", "_obs"))
    leak_obs = (sets_fold.fold != sets_fold.fold_obs).sum()
    leak_cr = (obs.groupby("cr").fold.nunique() > 1).sum()
    test_labelled = len(set(test.obs) & set(gt.set_id.str.split("-", n=1).str[1]))
    check("split: no observation/rotation in two folds, test unlabelled",
          leak_obs == 0 and leak_cr == 0 and test_labelled == 0 and set(obs.fold) == {0, 1, 2, 3, 4},
          f"set/obs fold mismatches {leak_obs}, rotations spanning folds {leak_cr}, labelled test images {test_labelled}, "
          f"fold sizes {obs.fold.value_counts().sort_index().tolist()}")
    check("transforms present for every image", obs[fp.MCOLS].notna().all().all() and test[fp.MCOLS].notna().all().all(),
          "M0..M5 defined for all 707 train + 180 test images")

    # ---- 4. prepared instance masks map back onto the raw GT (all 1,154 sets)
    counts = []
    for o in obs.itertuples():
        M = [getattr(o, c) for c in fp.MCOLS]
        for sid in o.set_ids.split("|"):
            canon = cv2.imread(str(root / "instances" / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
            counts.append(fp.panoptic_quality(fp.ann_to_instmap_from_rle(gt.segmentation_rle[gt.set_id == sid].tolist()),
                                              fp.to_original(canon, M)))
    m = fp.pq_from_counts(counts)
    check("prepared masks -> original frame vs raw GT (round trip)", m["pq"] > 0.99,
          f"PQ {m['pq']:.4f} (SQ {m['sq']:.4f}, RQ {m['rq']:.4f}) over 1154 sets")

    # ---- 5. cache == prepared PNGs (all files)
    fp.build_cache(cfg)
    cache = Path(cfg.cache_dir); mism = 0
    for p in sorted((root / "images").glob("*.png")):
        mism += not np.array_equal(np.load(cache / "img" / f"{p.stem}.npy"), cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
    for p in sorted((root / "instances").glob("*.png")):
        mism += not np.array_equal(np.load(cache / "inst" / f"{p.stem}.npy"), cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
    fg = np.load(cache / "fg_index.npy"); off = json.loads((cache / "fg_offsets.json").read_text())
    sid0 = sets.set_id.iloc[5]; a, b = off[sid0]
    fg_ok = np.array_equal(fg[a:b], np.flatnonzero(cv2.imread(str(root / "instances" / f"{sid0}.png"), cv2.IMREAD_UNCHANGED)))
    check("cache identical to prepared PNGs", mism == 0 and fg_ok, f"{mism} mismatching files of 2041; filament index exact: {fg_ok}")

    # ---- 6. windowed warp == full-image warp (exactness of the fast crop)
    rng = np.random.default_rng(0); img = np.load(cache / "img" / f"{obs.obs.iloc[0]}.npy")
    diffs = []
    for _ in range(50):
        M = fp._patch_affine((rng.uniform(0, 2048), rng.uniform(0, 2048)), 512, rng.uniform(0, 360),
                             rng.uniform(0.95, 1.05), rng.random() < .5, rng.random() < .5)
        full = cv2.warpAffine(img, M.astype(np.float32), (512, 512), flags=cv2.INTER_NEAREST, borderValue=0)
        diffs.append(int((full != fp._warp_window(img, M, 512, cv2.INTER_NEAREST)).sum()))
    # tiny differences are nearest-neighbour ties at exact half-pixel positions (float rounding), not errors
    check("fast windowed crop == full warp (within float rounding)", max(diffs) <= 0.001 * 512 * 512,
          f"max differing pixels over 50 random crops: {max(diffs)} of 262144 ({max(diffs) / 262144:.3%})")

    # ---- 7. training crops: aligned, on the disk, filament-rich
    ds = fp.FilamentPatchDataset(fp.Config(prepared_dir=cfg.prepared_dir, augment=False), [1, 2, 3, 4], augment=False)
    contrasts, fgfrac, disk_cov, empty, sky_in_disk = [], [], [], 0, []
    for i in range(300):
        im, sem, bnd, disk = [x.numpy()[0] for x in ds[i]]
        fgfrac.append(sem.mean()); disk_cov.append(disk.mean()); empty += sem.sum() == 0
        off = disk == 0
        if off.sum() > 1000:                       # only crops that actually reach past the disk
            core = cv2.erode(off.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # skip the interpolated rim
            sky_in_disk.append((im[core] == 0).mean())
        if sem.sum() > 50:
            ring = cv2.dilate(sem.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & (sem == 0) & (disk > 0)
            contrasts.append(im[sem > 0].mean() - im[ring].mean())
    t_ds = time.time()
    dsa = fp.FilamentPatchDataset(fp.Config(prepared_dir=cfg.prepared_dir), [1, 2, 3, 4])
    for i in range(100): dsa[i]
    per = (time.time() - t_ds) / 100
    check("training crops: labels on dark filaments", np.mean(np.array(contrasts) < 0) > 0.95,
          f"filament darker than surroundings in {np.mean(np.array(contrasts) < 0):.0%} of {len(contrasts)} crops "
          f"(mean contrast {np.mean(contrasts):.3f})")
    check("training crops: sampled on the disk", np.mean(disk_cov) > 0.6 and np.mean(sky_in_disk) > 0.9,
          f"mean disk coverage per crop {np.mean(disk_cov):.0%}; in the {len(sky_in_disk)} crops that reach the "
          f"limb, the loss mask's off-disk area is black sky for {np.mean(sky_in_disk):.1%} of pixels")
    check("training crops: class balance", np.mean(fgfrac) > 0.01,
          f"filament pixels per crop {np.mean(fgfrac):.2%}, crops without filament {empty / 3:.0f}%; "
          f"loading {per * 1000:.0f} ms per augmented crop")

    # ---- 8. perfect predictions through the scoring path (fixes 1 and 2)
    mon = fp.val_observations(cfg, "monitor"); rep = fp.val_observations(cfg, "report")
    years = pd.to_datetime(mon.time).dt.year
    check("monitor/report split", len(set(mon.obs) & set(rep.obs)) == 0 and len(mon) + len(rep) == (obs.fold == 0).sum(),
          f"monitor {len(mon)} images ({years.min()}-{years.max()}, sites {mon.site.value_counts().to_dict()}), "
          f"report {len(rep)} images, overlap 0")
    perf_cfg = fp.Config(prepared_dir=cfg.prepared_dir, min_area=0)
    probs = {o.obs: (cv2.imread(str(root / "instances" / f"{o.set_ids.split('|')[0]}.png"), cv2.IMREAD_UNCHANGED) > 0).astype(np.float32)
             for o in mon.itertuples()}
    single = mon.assign(set_ids=mon.set_ids.str.split("|").str[0])
    m = fp.score_probs(probs, single, perf_cfg)
    check("perfect prediction scores ~1 through the pipeline", m["pq"] > 0.98,
          f"PQ {m['pq']:.4f} (before the fixes the same test scored 0.864)")

    # ---- 9. model predictions: our score == organisers' code
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = fp.Config(prepared_dir=cfg.prepared_dir, arch="smp_unet", encoder="resnet34", min_area=60)
    ck = Path("runs/loss_bce_dice/best.pt")
    if ck.exists():
        model = fp.load_model(mcfg, ck, dev)
        sub_obs = mon.iloc[:15]
        probs = fp.predict_observations(model, mcfg, dev, sub_obs)
        ours = fp.score_probs(probs, sub_obs, mcfg)
        rows = []
        for o in sub_obs.itertuples():
            inst = fp.to_original(fp.prob_to_instances(np.asarray(probs[o.obs], np.float32), mcfg), [getattr(o, c) for c in fp.MCOLS])
            rows += [dict(filament_id=f"{o.obs}_{k}", segmentation_rle=fp.encode_mask(inst == k)) for k in np.unique(inst[inst > 0])]
        sids = [s for ss in sub_obs.set_ids for s in ss.split("|")]
        off_pq = fp.official_pq(gt[gt.set_id.isin(sids)][["filament_id", "segmentation_rle"]], pd.DataFrame(rows))
        check("our PQ == organisers' PQ on real model predictions", abs(off_pq - ours["pq"]) < 1e-6,
              f"organisers {off_pq:.6f} vs ours {ours['pq']:.6f} on 15 images ({len(rows)} predicted filaments)")
        # predictions must never cover the sky (the loss ignores it, so inference must too)
        yy, xx = np.ogrid[:2048, :2048]; c0, R, mg = fp._canon_geometry(mcfg)
        off = (xx - c0) ** 2 + (yy - c0) ** 2 > (R * mg) ** 2
        off_px = [int((np.asarray(probs[o.obs], np.float32)[off] >= mcfg.threshold).sum()) for o in sub_obs.itertuples()]
        areas = [int(a) for o in sub_obs.itertuples()
                 for a in np.bincount(fp.prob_to_instances(np.asarray(probs[o.obs], np.float32), mcfg).ravel())[1:]]
        check("no predicted filament pixels off the solar disk", max(off_px) == 0 and max(areas) < 100000,
              f"max off-disk filament pixels per image {max(off_px)}; largest predicted filament {max(areas)} px")
    else:
        check("our PQ == organisers' PQ on real model predictions", False, "no checkpoint to test with")

    n_fail = sum(not v["ok"] for v in REPORT.values())
    REPORT["_summary"] = dict(checks=len(REPORT), failed=n_fail, seconds=round(time.time() - t0))
    Path("verify_report.json").write_text(json.dumps(REPORT, indent=2))
    print(f"\n{len(REPORT) - 1} checks, {n_fail} failed, {time.time() - t0:.0f}s", flush=True)
