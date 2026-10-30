"""All experimental-setup runs, with the audited pipeline. Run from modelling/:  python run_experiments.py

Resumable: a run whose runs/<name>/result.json exists is skipped. Every run uses early stopping (patience
5, cap 30 epochs of 800 crops), LR halved on plateau, disk-masked loss, and validation in the original frame.

Choices are made ONLY on the 40 "monitor" images of fold 0. The 99 "report" images of fold 0 are scored
for every run but never used to choose anything, so the reported numbers are not inflated by selection.

Stages
  1. loss sweep on the U-Net baseline (bce_dice, dice, tversky, focal_dice)      -> best loss (monitor PQ)
  2. architecture sweep with the best loss                                        -> top-2 architectures
  3. second seed for the top-2 (is the gap bigger than run-to-run noise?)
  4. learning-rate check for the winner (1e-4, 1e-3 vs 3e-4)
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import filament_pipeline as fp

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = dict(prepared_dir="../data_prep/prepared", samples_per_epoch=800, epochs=30, patience=5,
            min_delta=0.002, num_workers=4, batch_size=8)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(name, **kw):
    out = Path("runs") / name / "result.json"
    if out.exists():
        r = json.loads(out.read_text()); log(f"skip {name} (done): monitor {r['monitor_pq']:.4f} report {r['report_pq']:.4f}")
        return r
    cfg = fp.Config(name=name, **{**BASE, **kw})
    log(f"start {name}: {kw}")
    model, s, _ = fp.train(cfg, device=DEV)
    rep = fp.validate_pq(model, cfg, DEV, "report")
    r = {**s, **{f"report_{k}": v for k, v in rep.items()}}
    out.write_text(json.dumps(r, indent=2))
    log(f"done  {name}: best epoch {s['best_epoch']}/{s['epochs_run']}, monitor {s['monitor_pq']:.4f}, "
        f"report {rep['pq']:.4f} (SQ {rep['sq']:.3f} RQ {rep['rq']:.3f}), {s['train_minutes']} min")
    del model; torch.cuda.empty_cache()
    return r


if __name__ == "__main__":
    t0 = time.time(); stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    # 1. loss sweep
    losses = {ls: run(f"loss_{ls}", arch="smp_unet", loss=ls) for ls in ["bce_dice", "dice", "tversky", "focal_dice"]}
    best_loss = max(losses, key=lambda k: losses[k]["monitor_pq"])
    log("best loss (monitor):", best_loss)

    # 2. architecture sweep with that loss
    archs = {"smp_unet": losses[best_loss]}
    for arch, enc in [("fcn8", "resnet34"), ("smp_unetpp", "resnet34"), ("smp_fpn", "resnet34"),
                      ("smp_deeplabv3p", "resnet34"), ("smp_pspnet", "resnet34"),
                      ("smp_unet", "efficientnet-b3"), ("unet", "none"), ("fcn32", "resnet34")]:
        key = arch if enc in ("resnet34", "none") else f"{arch}_{enc}"
        archs[key] = run(f"arch_{key}", arch=arch, encoder=(enc if enc != "none" else "resnet34"), loss=best_loss)
    top2 = sorted(archs, key=lambda k: -archs[k]["monitor_pq"])[:2]
    log("top-2 architectures (monitor):", top2)

    # 3. second seed for the top-2
    for k in top2:
        r = archs[k]
        run(f"seed2_{k}", arch=r["arch"], encoder=r["encoder"], loss=best_loss, seed=7)

    # 4. learning rate for the winner
    w = archs[top2[0]]
    for lr in [1e-4, 1e-3]:
        run(f"lr_{lr:g}_{top2[0]}", arch=w["arch"], encoder=w["encoder"], loss=best_loss, lr=lr)

    rows = [json.loads(p.read_text()) for p in sorted(Path("runs").glob("*/result.json"))]
    pd.DataFrame(rows).to_csv("runs/results.csv", index=False)
    log(f"ALL DONE in {(time.time() - t0) / 3600:.1f} h; best loss {best_loss}; top-2 {top2}")
