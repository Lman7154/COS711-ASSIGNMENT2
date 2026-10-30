"""Post-processing tuning, TTA, error diagnostics, timing and a submission for the chosen model.
Run from modelling/ after run_experiments.py:  python run_postprocess.py

Tuning uses ONLY the 40 monitor images of fold 0; every reported number is on the 99 report images.
Probability maps are stored as uint8 (0-255) to fit in RAM.
"""
import json, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch
import filament_pipeline as fp
from run_experiments import log

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def probs_u8(model, cfg, obs_df):
    out, t = {}, time.time()
    for o in obs_df.itertuples():
        out[o.obs] = np.round(fp.predict_prob(model, fp.load_image(cfg, o.obs), cfg, DEV) * 255).astype(np.uint8)
    return out, (time.time() - t) / len(obs_df)


def score(probs_u8_, obs_df, cfg, details=False):
    probs = {k: v.astype(np.float32) / 255 for k, v in probs_u8_.items()}
    return fp.score_probs(probs, obs_df, cfg, details=details)


if __name__ == "__main__":
    res = pd.read_csv("runs/results.csv")
    sweep = res[~res.name.str.startswith(("seed2_", "cv_"))]
    best = sweep.sort_values("monitor_pq", ascending=False).iloc[0]
    cfg = fp.Config(prepared_dir="../data_prep/prepared", name=best["name"], arch=best.arch, encoder=best.encoder,
                    loss=best.loss, lr=float(best.lr))
    model = fp.load_model(cfg, Path("runs") / best["name"] / "best.pt", DEV)
    mon, rep = fp.val_observations(cfg, "monitor"), fp.val_observations(cfg, "report")
    log("model:", best["name"], "| monitor", len(mon), "| report", len(rep))
    out = {"model": best["name"]}

    # ---- 1. threshold x min_area on monitor (no TTA)
    pm, _ = probs_u8(model, cfg, mon)
    grid = []
    for thr, ma in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], [60, 150, 300, 500, 800, 1200]):
        cfg.threshold, cfg.min_area, cfg.low_threshold, cfg.close_px = thr, ma, None, 0
        grid.append(dict(threshold=thr, min_area=ma, **score(pm, mon, cfg)))
    g1 = pd.DataFrame(grid).sort_values("pq", ascending=False); g1.to_csv("runs/pp_grid_threshold_area.csv", index=False)
    b = g1.iloc[0]; thr, ma = float(b.threshold), int(b.min_area)
    log(f"best threshold/min_area on monitor: {thr}/{ma} PQ {b.pq:.4f}")

    # ---- 2. hysteresis + closing on monitor (keeping the best threshold / min_area)
    grid = []
    for low, close in itertools.product([None, thr - 0.1, thr - 0.2], [0, 2, 4]):
        if low is not None and low <= 0.05:
            continue
        cfg.threshold, cfg.min_area, cfg.low_threshold, cfg.close_px = thr, ma, low, close
        grid.append(dict(low_threshold=low if low is not None else -1, close_px=close, **score(pm, mon, cfg)))
    g2 = pd.DataFrame(grid).sort_values("pq", ascending=False); g2.to_csv("runs/pp_grid_hysteresis_close.csv", index=False)
    b2 = g2.iloc[0]; low = None if b2.low_threshold < 0 else float(b2.low_threshold); close = int(b2.close_px)
    log(f"best hysteresis/closing on monitor: low={low} close={close} PQ {b2.pq:.4f}")
    del pm

    # ---- 3. report images: default vs tuned, without and with TTA (+ timing, diagnostics)
    settings = {"default (0.5, 60)": dict(threshold=0.5, min_area=60, low_threshold=None, close_px=0),
                "tuned": dict(threshold=thr, min_area=ma, low_threshold=low, close_px=close)}
    table = []
    for tta in [False, True]:
        cfg.tta = tta
        pr, sec = probs_u8(model, cfg, rep)
        for sname, s in settings.items():
            for k, v in s.items(): setattr(cfg, k, v)
            m, per = score(pr, rep, cfg, details=True)
            n_gt = (rep.set_ids.str.split("|").apply(len)).sum()
            table.append(dict(setting=sname, tta=tta, seconds_per_image=round(sec, 2), **m,
                              pred_per_image=round(per.n_pred.mean(), 1)))
            if sname == "tuned" and tta == table[-1]["tta"]:
                per.to_csv(f"runs/pp_per_image_tuned_tta{int(tta)}.csv", index=False)
        del pr
    t3 = pd.DataFrame(table); t3.to_csv("runs/pp_report.csv", index=False)
    log("\n" + t3.to_string(index=False))

    # GT filaments per image for reference
    sets = pd.read_csv("../data_prep/prepared/train_sets.csv")
    gt_per_img = sets[sets.obs.isin(rep.obs)].groupby("obs").n_inst.mean().mean()

    best_row = t3.sort_values("pq", ascending=False).iloc[0]
    out.update(threshold=thr, min_area=ma, low_threshold=low, close_px=close, use_tta=bool(best_row.tta),
               report_table=t3.to_dict("records"), gt_filaments_per_image=round(float(gt_per_img), 1),
               params=int(sum(p.numel() for p in model.parameters())))
    Path("runs/postprocess.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- 4. submission with the chosen settings
    for k, v in settings["tuned"].items(): setattr(cfg, k, v)
    cfg.tta = bool(best_row.tta)
    t = time.time()
    sub = fp.make_submission(model, cfg, "submission.csv", device=DEV)
    dt = time.time() - t
    out.update(submission_rows=len(sub), submission_images=int(sub.filament_id.str.rsplit("_", n=1).str[0].nunique()),
               submission_seconds=round(dt), submission_seconds_per_image=round(dt / 180, 2))
    Path("runs/postprocess.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"submission.csv: {len(sub)} rows, {out['submission_images']} images, {dt:.0f}s ({dt / 180:.2f}s/img)")
