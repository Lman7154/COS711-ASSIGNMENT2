"""Re-score every saved best checkpoint with the corrected inference (sky zeroed). Run from modelling/.

Why: the loss ignores off-disk pixels, so the model's sky output is meaningless; before this fix it could
form one huge false "filament" per image. Checkpoints are kept as they are (no retraining): each run's
best.pt is re-scored on its monitor and report images. The original numbers are kept as *_v1 fields.
Note: early stopping picked each checkpoint using the old scoring; that choice is not redone.
Resumable: runs already carrying "rescored": true are skipped.
"""
import json, time, glob
from pathlib import Path
import pandas as pd, torch
import filament_pipeline as fp

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    t0 = time.time()
    for p in sorted(glob.glob("runs/*/result.json")):
        r = json.loads(Path(p).read_text())
        if r.get("rescored"):
            continue
        d = Path(p).parent
        c = json.loads((d / "config.json").read_text())
        cfg = fp.Config(**{k: v for k, v in c.items() if k in fp.Config.__dataclass_fields__})
        m = fp.load_model(cfg, d / "best.pt", DEV)
        mon = fp.validate_pq(m, cfg, DEV, "monitor"); rep = fp.validate_pq(m, cfg, DEV, "report")
        for k in list(r):
            if k.startswith("report_") or k == "monitor_pq":
                r[f"{k}_v1"] = r[k]
        r["monitor_pq"] = mon["pq"]
        r.update({f"report_{k}": v for k, v in rep.items()}); r["rescored"] = True
        Path(p).write_text(json.dumps(r, indent=2))
        print(f"{r['name']:36s} monitor {r['monitor_pq_v1']:.4f} -> {mon['pq']:.4f} | report {r['report_pq_v1']:.4f} -> {rep['pq']:.4f}", flush=True)
        del m; torch.cuda.empty_cache()
    rows = [json.loads(Path(p).read_text()) for p in sorted(glob.glob("runs/*/result.json"))]
    pd.DataFrame([r for r in rows if not r["name"].startswith("cv_")]).to_csv("runs/results.csv", index=False)
    cv = json.loads(Path("runs/cv_5fold.json").read_text())
    by = {r["name"]: r for r in rows}
    for f in cv["folds"]:
        f["report_pq_v1"] = f["report_pq"]; f["report_pq"] = by[f["name"]]["report_pq"]
    s = pd.Series([f["report_pq"] for f in cv["folds"]])
    cv.update(mean_v1=cv["mean"], std_v1=cv["std"], mean=float(s.mean()), std=float(s.std()))
    Path("runs/cv_5fold.json").write_text(json.dumps(cv, indent=2))
    pd.DataFrame(cv["folds"]).to_csv("runs/cv_5fold.csv", index=False)
    print(f"5-fold report PQ {cv['mean_v1']:.4f} +/- {cv['std_v1']:.4f} -> {cv['mean']:.4f} +/- {cv['std']:.4f}")
    print(f"done in {(time.time() - t0) / 60:.0f} min", flush=True)
