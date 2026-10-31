"""Kaggle-improvement experiments. Run from modelling/:  python run_improve.py [stage ...]
Stages (default: all, in this order; each is resumable and writes runs/improve/<stage>.json):

  cv        Fair 5-fold estimate: every fold model scored on ITS OWN report images, with the default and the
            tuned post-processing, with and without flip TTA. Post-processing is re-tuned on the pooled
            monitor images of all 5 folds (200 images), never on report images.
  small     Confidence-aware small-blob rule (keep a blob of small_area..min_area px if its mean probability
            >= small_conf). Tuned on pooled monitor, reported on pooled report.
  ensemble  Does averaging several models help? Only models validated on fold 0 are used (none of them saw a
            fold-0 image), so the test is fair. Post-processing tuned on fold-0 monitor, reported on report.
  cldice    Same recipe as the chosen model, loss bce_dice + 0.5 clDice (connectivity), early stopping on
            fold-0 monitor; compared under the identical protocol.
  alldata   (a) recipe check: fixed-epoch training on folds 1-4 (no early stopping), scored on fold 0;
            (b) the same recipe on ALL 707 images (no local score is possible for it).
  subs      Candidate submission files in submissions/, only for changes that helped locally.

Probability maps are cached as uint8 PNGs in runs/improve/probs/<model>_tta<0|1>/<obs>.png.
"""
import json, sys, time, itertools
from dataclasses import fields
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
import filament_pipeline as fp
from run_experiments import run, log, BASE

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREP = "../data_prep/prepared"
OUT = Path("runs/improve"); PROBS = OUT / "probs"; SUBS = Path("submissions")
OUT.mkdir(parents=True, exist_ok=True); SUBS.mkdir(exist_ok=True)
FOLD_MODELS = {0: "arch_smp_unet_efficientnet-b3", 1: "cv_fold1", 2: "cv_fold2", 3: "cv_fold3", 4: "cv_fold4"}
# every model trained with fold 0 held out (none saw a fold-0 image)
FOLD0_MODELS = ["arch_smp_unet_efficientnet-b3", "seed2_smp_unet_efficientnet-b3", "lr_0.0001_smp_unet_efficientnet-b3",
                "loss_bce_dice", "arch_smp_unetpp", "loss_focal_dice", "arch_smp_deeplabv3p"]
DEFAULT = dict(threshold=0.5, min_area=60)
TUNED0 = dict(threshold=0.6, min_area=500)            # tuned earlier on fold-0 monitor (runs/postprocess.json)
GRID = [dict(threshold=t, min_area=a) for t, a in itertools.product([0.4, 0.5, 0.6, 0.7], [60, 300, 500, 800, 1200])]
PP9 = [dict(threshold=t, min_area=a) for t, a in itertools.product([0.5, 0.6, 0.7], [300, 500, 800])]
FIELDS = {f.name for f in fields(fp.Config)}


def key(s):
    return ",".join(f"{k}={v}" for k, v in s.items())


def cfg_of(name, **kw):
    c = json.loads((Path("runs") / name / "config.json").read_text()); c.update(kw)
    return fp.Config(**{k: v for k, v in c.items() if k in FIELDS})


def gpu_temp():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def ensure_probs(name, stems, tta=True):
    """Cache probability maps (uint8 PNG) of model `name` for the given observation stems."""
    d = PROBS / f"{name}_tta{int(tta)}"; d.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not (d / f"{s}.png").exists()]
    if todo:
        cfg = cfg_of(name, tta=tta, prepared_dir=PREP)
        model = fp.load_model(cfg, Path("runs") / name / "best.pt", DEV)
        t = time.time()
        for s in todo:
            p = fp.predict_prob(model, fp.load_image(cfg, s), cfg, DEV)
            cv2.imwrite(str(d / f"{s}.png"), np.round(p * 255).astype(np.uint8))
        sec = (time.time() - t) / len(todo)
        (d / "_timing.json").write_text(json.dumps(dict(seconds_per_image=round(sec, 3), n=len(todo))))
        log(f"probs {name} tta={tta}: {len(todo)} images, {sec:.2f}s/img, GPU {gpu_temp()}C")
        del model; torch.cuda.empty_cache()
    return d


def sec_per_image(d):
    p = Path(d) / "_timing.json"
    return json.loads(p.read_text())["seconds_per_image"] if p.exists() else None


def load_prob(d, s):
    return cv2.imread(str(Path(d) / f"{s}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255


def getter(dir_of):
    """dir_of: {stem: [dirs]} -> get_prob(stem) averaging those models' maps."""
    return lambda s: sum(load_prob(d, s) for d in dir_of[s]) / len(dir_of[s])


def grid(get_prob, obs_df, settings, **extra):
    base = fp.Config(prepared_dir=PREP)
    named = {key(s): {**s, **extra} for s in settings}
    return fp.score_grid(get_prob, obs_df, base, named)


def fold_obs(f):
    c = fp.Config(prepared_dir=PREP, val_fold=f)
    return fp.val_observations(c, "monitor"), fp.val_observations(c, "report")


def done(stage):
    p = OUT / f"{stage}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(stage, obj):
    (OUT / f"{stage}.json").write_text(json.dumps(obj, indent=2, default=float))


def best_of(res):
    k = max(res, key=lambda k: res[k]["pq"]); return k, res[k]


def parse(k):
    out = {}
    for kv in k.split(","):
        a, b = kv.split("="); out[a] = int(b) if b.isdigit() else float(b)
    return out


# ------------------------------------------------------------------------------------------------ cv
def stage_cv():
    if done("cv"):
        return done("cv")
    mons, reps, rows = [], [], []
    dirs = {0: {}, 1: {}}
    for f, name in FOLD_MODELS.items():
        m, r = fold_obs(f); mons.append(m); reps.append(r)
        stems = pd.concat([m, r]).obs.tolist()
        for tta in (0, 1):
            d = ensure_probs(name, stems, bool(tta))
            dirs[tta].update({s: [d] for s in stems})
    mon_all = pd.concat(mons, ignore_index=True)
    # tune on the pooled monitor images (TTA maps)
    g = grid(getter(dirs[1]), mon_all, GRID)
    kbest, mbest = best_of(g); tuned_cv = parse(kbest)
    log(f"pooled-monitor best: {kbest} PQ {mbest['pq']:.4f}")
    settings = {"default (0.5/60)": DEFAULT, "fold0-tuned (0.6/500)": TUNED0, f"cv-tuned ({kbest})": tuned_cv}
    for f, r in zip(FOLD_MODELS, reps):
        for tta in (0, 1):
            res = fp.score_grid(getter(dirs[tta]), r, fp.Config(prepared_dir=PREP), settings)
            for sname, mres in res.items():
                rows.append(dict(fold=f, tta=tta, setting=sname, **mres))
        log(f"fold {f} report done")
    df = pd.DataFrame(rows); df.to_csv(OUT / "cv_report_by_fold.csv", index=False)
    summ = df.groupby(["setting", "tta"]).pq.agg(["mean", "std"]).reset_index()
    pooled = []
    for (sname, tta), d in df.groupby(["setting", "tta"]):
        s = d[["tp", "fp", "fn"]].sum(); iou = (d.sq * d.tp).sum()
        pooled.append(dict(setting=sname, tta=tta, pooled_pq=iou / (s.tp + 0.5 * s.fp + 0.5 * s.fn)))
    summ = summ.merge(pd.DataFrame(pooled), on=["setting", "tta"]); summ.to_csv(OUT / "cv_summary.csv", index=False)
    log("\n" + summ.to_string(index=False))
    out = dict(monitor_grid=g, tuned_cv=tuned_cv, summary=summ.to_dict("records"),
               seconds_per_image={t: sec_per_image(PROBS / f"{FOLD_MODELS[0]}_tta{t}") for t in (0, 1)})
    save("cv", out); return out


# ------------------------------------------------------------------------------------------------ small
def stage_small(cv):
    if done("small"):
        return done("small")
    base = cv["tuned_cv"]; mons, reps, dir_of = [], [], {}
    for f, name in FOLD_MODELS.items():
        m, r = fold_obs(f); mons.append(m); reps.append(r)
        d = PROBS / f"{name}_tta1"; dir_of.update({s: [d] for s in pd.concat([m, r]).obs})
    mon_all, rep_all = pd.concat(mons, ignore_index=True), pd.concat(reps, ignore_index=True)
    settings = {"off": {**base}}
    for sa, sc in itertools.product([100, 200, 300], [0.7, 0.75, 0.8, 0.85, 0.9]):
        if sa < base["min_area"]:
            settings[f"small_area={sa},small_conf={sc}"] = {**base, "small_area": sa, "small_conf": sc}
    g = fp.score_grid(getter(dir_of), mon_all, fp.Config(prepared_dir=PREP), settings)
    kbest = max(g, key=lambda k: g[k]["pq"])
    log(f"small-blob monitor: off {g['off']['pq']:.4f} | best {kbest} {g[kbest]['pq']:.4f}")
    rep = fp.score_grid(getter(dir_of), rep_all, fp.Config(prepared_dir=PREP),
                        {"off": settings["off"], kbest: settings[kbest]})
    log(f"small-blob report: off {rep['off']['pq']:.4f} | {kbest} {rep[kbest]['pq']:.4f}")
    out = dict(base=base, monitor_grid=g, best=kbest, best_setting=settings[kbest], report=rep,
               helps_on_monitor=bool(g[kbest]["pq"] > g["off"]["pq"] + 0.002))
    save("small", out); return out


# ------------------------------------------------------------------------------------------------ ensemble
def eval_set(models, mon, rep, report=True):
    """Tune post-processing on monitor for this model set; score the best setting on report."""
    dir_of = {s: [PROBS / f"{m}_tta1" for m in models] for s in pd.concat([mon, rep]).obs}
    g = grid(getter(dir_of), mon, PP9); k, mres = best_of(g)
    out = dict(models=list(models), pp=parse(k), monitor_pq=mres["pq"],
               seconds_per_image=round(sum(sec_per_image(PROBS / f"{m}_tta1") or 0 for m in models), 2))
    if report:
        r = fp.score_grid(getter(dir_of), rep, fp.Config(prepared_dir=PREP), {"x": parse(k)})["x"]
        out.update(report_pq=r["pq"], report_sq=r["sq"], report_rq=r["rq"], report_tp=r["tp"], report_fp=r["fp"],
                   report_fn=r["fn"], report_frag_gt=r["frag_gt"])
    return out


def stage_ensemble():
    if done("ensemble"):
        return done("ensemble")
    mon, rep = fold_obs(0); stems = pd.concat([mon, rep]).obs.tolist()
    for m in FOLD0_MODELS:
        ensure_probs(m, stems, True); time.sleep(20)
    rows = []
    for m in FOLD0_MODELS:
        rows.append(dict(kind="single", **eval_set([m], mon, rep))); log("single", m, f"{rows[-1]['report_pq']:.4f}")
    effb3 = FOLD0_MODELS[:3]
    for kind, ms in [("2 seeds effb3", effb3[:2]), ("3 effb3", effb3), ("effb3 + r34 U-Net", effb3[:1] + ["loss_bce_dice"]),
                     ("3 effb3 + r34 U-Net", effb3 + ["loss_bce_dice"])]:
        rows.append(dict(kind=kind, **eval_set(ms, mon, rep))); log(kind, f"{rows[-1]['report_pq']:.4f}")
    # greedy forward selection on MONITOR only (report scored only for the final pick)
    chosen, best_mon, trace = [], -1, []
    while len(chosen) < 5:
        cands = [(eval_set(chosen + [m], mon, rep, report=False), m) for m in FOLD0_MODELS if m not in chosen]
        r, m = max(cands, key=lambda c: c[0]["monitor_pq"])
        if r["monitor_pq"] <= best_mon + 0.002:
            break
        chosen.append(m); best_mon = r["monitor_pq"]; trace.append(dict(added=m, monitor_pq=r["monitor_pq"]))
        log("greedy +", m, f"monitor {best_mon:.4f}")
    rows.append(dict(kind="greedy (monitor)", **eval_set(chosen, mon, rep)))
    df = pd.DataFrame(rows); df.to_csv(OUT / "ensemble_fold0.csv", index=False)
    log("\n" + df[["kind", "models", "pp", "monitor_pq", "report_pq", "seconds_per_image"]].to_string(index=False))
    out = dict(table=df.to_dict("records"), greedy_trace=trace)
    save("ensemble", out); return out


# ------------------------------------------------------------------------------------------------ cldice
def stage_cldice():
    if done("cldice"):
        return done("cldice")
    name = "cldice_smp_unet_efficientnet-b3"
    r = run(name, arch="smp_unet", encoder="efficientnet-b3", loss="bce_dice_cldice", lr=3e-4)
    mon, rep = fold_obs(0); ensure_probs(name, pd.concat([mon, rep]).obs.tolist(), True)
    rows = [dict(kind="bce_dice seed 42", **eval_set([FOLD0_MODELS[0]], mon, rep)),
            dict(kind="bce_dice seed 7", **eval_set([FOLD0_MODELS[1]], mon, rep)),
            dict(kind="bce_dice + clDice", best_epoch=r["best_epoch"], train_minutes=r["train_minutes"],
                 **eval_set([name], mon, rep))]
    df = pd.DataFrame(rows); df.to_csv(OUT / "cldice_fold0.csv", index=False)
    log("\n" + df[["kind", "pp", "monitor_pq", "report_pq", "report_frag_gt"]].to_string(index=False))
    helps = rows[2]["monitor_pq"] > max(rows[0]["monitor_pq"], rows[1]["monitor_pq"])
    out = dict(table=rows, helps_on_monitor=bool(helps)); save("cldice", out); return out


# ------------------------------------------------------------------------------------------------ alldata
def fixed_run(name, folds, epochs, loss):
    res = Path("runs") / name / "summary.json"
    if (Path("runs") / name / "best.pt").exists() and res.exists():
        log("skip", name, "(done)"); return json.loads(res.read_text())
    cfg = fp.Config(name=name, **{**BASE, "arch": "smp_unet", "encoder": "efficientnet-b3", "loss": loss, "lr": 3e-4})
    log(f"start {name}: folds {folds}, {epochs} epochs, loss {loss}")
    _, s = fp.train_fixed(cfg, folds, epochs, device=DEV); torch.cuda.empty_cache()
    log(f"done  {name}: {s['train_minutes']} min, GPU {gpu_temp()}C"); return s


def stage_alldata(cld):
    if done("alldata"):
        return done("alldata")
    # candidate epoch counts: the median best epoch of the 5 bce_dice fold runs (12), and the best epoch of
    # the clDice run on fold 0 (clDice learned more slowly). Chosen on fold-0 MONITOR PQ.
    median_bd = int(np.median([f["best_epoch"] for f in json.loads(Path("runs/cv_5fold.json").read_text())["folds"]]))
    loss = "bce_dice_cldice" if cld["helps_on_monitor"] else "bce_dice"
    ref = FOLD0_MODELS[0] if loss == "bce_dice" else "cldice_smp_unet_efficientnet-b3"
    ref_best = json.loads((Path("runs") / ref / "result.json").read_text())["best_epoch"]
    mon, rep = fold_obs(0)
    rows = [dict(kind=f"early stopping ({ref}, best epoch {ref_best})", epochs=None, **eval_set([ref], mon, rep))]
    for e in sorted({median_bd, ref_best}):
        chk = f"fixed{e}_f1234_{loss}"
        s1 = fixed_run(chk, [1, 2, 3, 4], e, loss)
        ensure_probs(chk, pd.concat([mon, rep]).obs.tolist(), True)
        rows.append(dict(kind=f"fixed {e} epochs, cosine LR", epochs=e, train_minutes=s1["train_minutes"],
                         **eval_set([chk], mon, rep)))
        time.sleep(30)
    log("\n" + pd.DataFrame(rows)[["kind", "pp", "monitor_pq", "report_pq"]].to_string(index=False))
    fixed = [r for r in rows if r["epochs"]]
    epochs = max(fixed, key=lambda r: r["monitor_pq"])["epochs"]
    full = f"alldata_fixed{epochs}_{loss}"
    s2 = fixed_run(full, [0, 1, 2, 3, 4], epochs, loss)
    chosen = next(r for r in fixed if r["epochs"] == epochs)
    out = dict(epochs=epochs, loss=loss, recipe_check=rows, alldata_run=full, alldata_minutes=s2["train_minutes"],
               recipe_gap_report=chosen["report_pq"] - rows[0]["report_pq"])
    save("alldata", out); return out


# ------------------------------------------------------------------------------------------------ subs
def write_sub(fname, models, pp, note):
    path = SUBS / fname
    if path.exists():
        log("skip", fname, "(exists)"); return None
    cfg = cfg_of(models[0], prepared_dir=PREP, tta=True, **pp)
    ms = [fp.load_model(cfg_of(m, prepared_dir=PREP), Path("runs") / m / "best.pt", DEV) for m in models]
    t = time.time(); sub = fp.make_submission(None, cfg, path, device=DEV, models=ms); dt = time.time() - t
    info = dict(file=fname, models=models, post_processing=pp, tta=True, rows=len(sub),
                images=int(sub.filament_id.str.rsplit("_", n=1).str[0].nunique()), seconds_per_image=round(dt / 180, 2), note=note)
    log(f"{fname}: {info['rows']} rows, {info['images']} images, {info['seconds_per_image']}s/img")
    del ms; torch.cuda.empty_cache(); return info


def stage_subs(cv, small, ens, cld, alld):
    prev = done("subs") or {"files": []}
    have = {f["file"] for f in prev["files"]}
    files = list(prev["files"])
    tbl = pd.DataFrame(ens["table"])
    single = tbl[(tbl.kind == "single") & (tbl.models.apply(lambda m: m[0]) == FOLD0_MODELS[0])].iloc[0]
    e3 = tbl[tbl.kind == "3 effb3"].iloc[0]
    ens_helps = e3.monitor_pq > single.monitor_pq + 0.002
    pp = dict(cv["tuned_cv"])
    if small["helps_on_monitor"]:
        pp.update(small_area=small["best_setting"]["small_area"], small_conf=small["best_setting"]["small_conf"])
    plan = [("sub_alldata.csv", [alld["alldata_run"]], pp, "one model trained on all 707 images")]
    if ens_helps:
        epp = dict(e3.pp)
        if small["helps_on_monitor"]:
            epp.update(small_area=pp["small_area"], small_conf=pp["small_conf"])
        plan.append(("sub_5fold_ensemble.csv", list(FOLD_MODELS.values()), epp, "average of the 5 fold models"))
        plan.append(("sub_alldata_plus_5fold.csv", [alld["alldata_run"]] + list(FOLD_MODELS.values()), epp,
                     "all-data model + 5 fold models"))
    for fname, models, p, note in plan:
        if fname in have:
            continue
        info = write_sub(fname, models, p, note)
        if info:
            files.append(info); save("subs", dict(files=files, ensemble_helps_on_monitor=bool(ens_helps)))
        time.sleep(30)
    out = dict(files=files, ensemble_helps_on_monitor=bool(ens_helps)); save("subs", out); return out


if __name__ == "__main__":
    want = set(sys.argv[1:]) or {"cv", "small", "ensemble", "cldice", "alldata", "subs"}
    t0 = time.time()
    cv = stage_cv() if "cv" in want or want & {"small", "subs"} else None
    small = stage_small(cv) if want & {"small", "subs"} else None
    ens = stage_ensemble() if want & {"ensemble", "subs"} else None
    cld = stage_cldice() if want & {"cldice", "alldata", "subs"} else None
    alld = stage_alldata(cld) if want & {"alldata", "subs"} else None
    if "subs" in want:
        stage_subs(cv, small, ens, cld, alld)
    log(f"finished {sorted(want)} in {(time.time() - t0) / 60:.0f} min")
