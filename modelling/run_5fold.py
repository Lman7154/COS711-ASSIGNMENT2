"""5-fold cross-validation of the chosen configuration. Run from modelling/:  python run_5fold.py

Reads the winning loss/architecture/learning rate from runs/results.csv (chosen on monitor PQ only), then
trains one model per held-out fold with early stopping. Each fold's model is scored on that fold's REPORT
images (never used for early stopping), giving PQ mean +/- std across folds. Fold 0 reuses the sweep run.
Resumable: finished folds are skipped.
"""
import json, time
from pathlib import Path
import pandas as pd, torch
import filament_pipeline as fp
from run_experiments import run, log

if __name__ == "__main__":
    res = pd.read_csv("runs/results.csv")
    sweep = res[~res.name.str.startswith(("seed2_", "cv_"))]
    best = sweep.sort_values("monitor_pq", ascending=False).iloc[0]
    kw = dict(arch=best.arch, encoder=best.encoder, loss=best.loss, lr=float(best.lr))
    log("winner:", best["name"], kw)
    rows = [dict(fold=0, name=best["name"], report_pq=best.report_pq, best_epoch=int(best.best_epoch))]
    for f in range(1, 5):
        r = run(f"cv_fold{f}", val_fold=f, **kw)
        rows.append(dict(fold=f, name=r["name"], report_pq=r["report_pq"], best_epoch=r["best_epoch"]))
    df = pd.DataFrame(rows); df.to_csv("runs/cv_5fold.csv", index=False)
    Path("runs/cv_5fold.json").write_text(json.dumps(dict(config=kw, winner_run=best["name"],
        mean=float(df.report_pq.mean()), std=float(df.report_pq.std()), folds=rows), indent=2))
    log(f"5-fold report PQ = {df.report_pq.mean():.4f} +/- {df.report_pq.std():.4f}")
