# Understanding the Data (MAGFiLO) — one complete read

This is the single document to read for the data side. It is self-contained: it starts from what a
filament is, explains every term and every code choice in plain language, and walks through what the
notebook `data_prep.ipynb` found, section by section. You should not need to open another file.

(Related docs, only if you want them: `DATA_PREP_DECISIONS.md` = the decisions in table form with
sources; `../PROJECT_UNDERSTANDING.md` = the problem, the goal, the marking and the rules.)

---

## Contents
1. What is a solar filament?
2. What is an H-alpha image, and where the images come from
3. What is an "annotator", and why their labels differ
4. The words we use: observation vs annotation set vs filament
5. The code pieces explained (pycocotools, polygon, RLE, the SITES dictionary, cv2)
6. What made our masks "identical to Kaggle's" (the bug and the fix)
7. Walk-through of the notebook, section by section, with the numbers we found
8. One-screen summary

---

## 1. What is a solar filament?

A **solar filament** is a dense "cloud" of cooler gas held above the Sun's surface by magnetic field
lines. In the images we use it looks like a **dark, thin, curved thread** lying on the bright face of
the Sun. Each filament has:
- a **spine** — its long main axis, and
- **barbs** — tiny side branches coming off the spine (their direction encodes the magnetic "chirality",
  left- or right-handed).

The same structure is called a **filament** when it's a dark thread on the disk, and a **prominence**
when it's seen glowing at the edge of the Sun against black space.

Why anyone cares: filaments sit at the core of solar eruptions — **CMEs, solar flares, and SEP storms** —
which can damage power grids, GPS, aircraft and astronauts. Outlining them precisely helps space-weather
forecasting. Our job is to produce one outline (mask) per filament.

---

## 2. What is an H-alpha image, and where the images come from

**H-alpha (Hα)** is one exact colour of red light (656.3 nm) emitted by hydrogen. An **H-alpha image** is
a photo of the Sun taken through a filter that passes only that one red colour. Filaments are made of
hydrogen gas that **absorbs** H-alpha light, so in these images they show up as **dark threads** on the
bright disk. In ordinary white light they're basically invisible — that's why the whole dataset is
H-alpha. "The light we use" therefore means the H-alpha slice of sunlight, not full white sunlight.

We did **not** choose H-alpha; it's a given of the dataset. The brief says the data is MAGFiLO; the Kaggle
Data page says the images are H-alpha from GONG; the paper's title is literally *...from H-Alpha
Observations*.

The images come from **GONG** (Global Oscillation Network Group), **six telescopes** spread around the
world so the Sun is watched 24/7. The site is encoded in every file name (see the SITES table in §5).

---

## 3. What is an "annotator", and why their labels differ

An **annotator** is a trained person who was given a solar image and asked to **draw an outline around
every filament** they could see, and mark each one's spine and chirality. These hand-drawn outlines are
the **ground truth** — the correct answers our model learns from. There is no automatic truth here; the
truth is whatever expert humans drew.

The MAGFiLO team used a **double-blind protocol**: groups of **three annotators each independently
labelled the same image**, without seeing each other's work (MAGFiLO paper). Kaggle says this directly:
> Kaggle Data page: *"all filaments present within one image may be annotated multiple times by different
> annotators, independently ... e.g. `010101-...` and `010102-...` (note that they are different by only
> one digit). You may treat those images simply as 'different' images."*

**Why their labels differ.** Filaments are faint, fuzzy, edgeless shapes. Two experts honestly disagree on:
- **whether** a faint smudge is a filament at all (so they find different *numbers*), and
- **exactly where** the edge is (so their outlines differ in shape).

This is genuine uncertainty, not carelessness. We measured it (notebook §6): two annotators of the same
image agree at only **PQ ≈ 0.34** and differ by about **2 filaments on average**. In one example three
annotators found **16, 8, and 7** filaments on the *same* image.

**Why it matters for us:**
- The training labels are genuinely **noisy** — don't treat any one annotator as perfect.
- It sets a realistic **ceiling**: matching human-level agreement (~0.34) is already strong. The
  organisers say any PQ above 0.35 is valuable.
- It drives two decisions: use **one random annotator's labels per training sample**, and **validate
  against all of them** (which is how Kaggle scores).

---

## 4. The words we use (so the group stays consistent)

- **observation** = one actual JPEG photo of the Sun. (707 in training, 180 in test.)
- **annotation set** = one annotator's complete labelling of one observation. (1,154 in training.)
  Its id is `<annotator/batch code>-<file stem>`, e.g. `010101-20141027025054Uh`.
- **filament / annotation** = one outlined filament inside a set. (8,199 in training.)

So the chain is: **707 observations → 1,154 annotation sets → 8,199 filaments.** 296 observations were
labelled by 2–3 annotators, which is where the extra sets come from.

> Important consequence: because the same photo can appear as 2–3 "annotation sets", a careless split
> could put copies of one photo in both training and validation. Section 7 (the split) handles this.

---

## 5. The code pieces explained

These are the lines in the notebook that aren't obvious, in plain language.

### `from pycocotools import mask as mask_util   # official GT format`

**pycocotools** is the official Python toolkit for the **COCO format** — the standard way computer-vision
datasets store object outlines. Our labels are in COCO format. The one tool we use from it, `mask`
(renamed `mask_util` in our code), converts a mask between two forms:

- **Polygon** — a connect-the-dots outline of the filament: a list of corner points
  `[x0, y0, x1, y1, …]` where the last point joins back to the first (a closed loop). Compact, and it's
  **how the labels are stored** in the JSON.
  > Kaggle Data page: *"A segmentation … is a list of floats representing a closed path. Each path forms
  > a polygon capturing a filament's shape."*

- **RLE (Run-Length Encoding)** — a compressed version of the **filled pixel mask**. Instead of storing a
  value for all ~4.2 million pixels, it stores *runs*: "off 500, on 30, off 1200, …". Tiny, because a mask
  is mostly long stretches of the same value. This is the format Kaggle's **`submission.csv`** needs.
  > Kaggle Data page: *"instead of binary masks, we store segmentations as polygons in RLE format. The
  > conversion (RLE ↔ Binary Masks) are lossless."*
  > Kaggle Submission page: *"Column `segmentation_rle` contains RLE counts each encoding the mask
  > corresponding to one filament."*

So the flow is: **polygon (label) → filled pixel mask → RLE (submission)**, and pycocotools does every
step. Kaggle itself tells us to use this exact package:
> Kaggle Data page: *"The annotations are structured into the COCO-style data format … you can take
> advantage of many common operations using the `pycocotools` python package."*
> Kaggle Submission page: *"To convert your output … to the expected format (RLE Counts), please use the
> `pycocotools` Python package."*

The comment `# official GT format` in the code is a reminder that this import is what keeps our masks and
our scoring identical to Kaggle's. See §6 for what "identical" means and why it was not automatic.

### The `SITES` dictionary

Every file name encodes **when and where** the image was taken: `YYYYMMDDHHMMSSII.jpeg`. The letter just
before `.jpeg` (the `B` in `20140609195854Bh.jpeg`) is the **telescope site**.
> Kaggle Data page: *"Each observation's capture time and instrument are encoded in its file name …
> `20260901165702Bh.jpeg` corresponds to … the Big Bear (Bh) Solar Observatory."*

`SITES` just turns that one letter into a readable name:

| Code | Telescope |
|---|---|
| B | Big Bear (USA) |
| C | Cerro Tololo (Chile) |
| L | Learmonth (Australia) |
| M | Mauna Loa (USA) |
| T | El Teide (Spain) |
| U | Udaipur (India) |

**Why we need it.** The six telescopes see slightly different conditions (weather, optics), so "site" is
a real variable. We use it twice: (1) to check image quality **per site** (does the test set look
different from training? — no, it doesn't), and (2) to **stratify the validation split by site** so every
fold contains all six instruments. The code needs each image's site to do either, and the dictionary makes
plots and tables readable ("Big Bear (USA)" instead of "B"). `print("DATA_ROOT:", …)` simply shows which
folder the data loaded from, so you can confirm the paths are correct.

### `cv2` and the "No module named 'cv2'" error

`cv2` is **OpenCV**, the library we use to read and process the images. If a cell fails with
`ModuleNotFoundError: No module named 'cv2'`, it just means the notebook is running on a Python where the
packages aren't installed (a wrong-kernel situation, not a bug). Fix: pick the Python that has the
packages as the kernel, or run this at the top of the notebook to install into the exact Python in use:
```python
import sys
!{sys.executable} -m pip install opencv-python-headless numpy pandas matplotlib seaborn tqdm scikit-learn pycocotools nbformat
```
On Kaggle most are pre-installed; you may still need `pip install pycocotools`.

---

## 6. What made our masks "identical to Kaggle's" (the bug and the fix)

**The problem.** A filament is stored as a **polygon** (an outline). To train and to score, we have to
turn that outline into a **filled pixel mask** (every pixel inside the outline = 1). There is more than
one way to "fill in" an outline, and different tools round the edges slightly differently.

- Our **first** version filled the outline with **OpenCV's `fillPoly`**.
- Kaggle builds its ground-truth masks with **pycocotools**, and scores by decoding **pycocotools RLE**.

When we compared the two on 1,500 real filaments, the OpenCV masks matched the pycocotools masks at only
**IoU 0.878 on average (worst 0.589)**. "IoU" here just means how much the two filled shapes overlap;
1.0 would be a perfect match. So the two tools disagreed on roughly 12% of each filament's pixels — almost
all of it along the thin edge. Because filaments are only **~11 pixels wide**, a one-pixel edge difference
is a big fraction of the whole object. Result: our local score would have been **systematically wrong**,
and we'd have been tuning the model against the wrong target.

**The fix.** We switched to drawing every mask with **pycocotools** (`mask_util.frPyObjects`), exactly the
same tool and the same rounding Kaggle uses. "Identical to Kaggle's" means, concretely:
1. Our mask **areas match the label's `area` field exactly** (maximum difference 0.0 pixels across the
   whole dataset). Kaggle computes that `area` with pycocotools too, which is why they now agree.
2. Our local **Panoptic Quality equals the organisers' own scoring code exactly: 0.34327 vs 0.34327**
   (notebook §10b runs both on the same 93 annotation sets and asserts they agree).

That's what the earlier claim meant. The single import `from pycocotools import mask` is what lets us
trust a local PQ number without spending one of Kaggle's five daily submissions to check it.

(Note: later we also made the pycocotools version **fast** — it originally drew each mask on the full
2048×2048 image, which was slow — by only filling inside each filament's bounding box. We verified this
fast version produces the **exact same pixels**, so it's a speed change only, not an accuracy change.)

---

## 7. Walk-through of the notebook, section by section

Each code cell in the notebook also has a short "What & why" above it and an "Interpretation" below it;
this is the same story in one place.

### §1 — What is in the dataset?
*Why:* know what one training example is before modelling, and count observations vs annotation sets
separately (they're easy to confuse, and confusing them would break the split).
*Found:* 707 observations, 1,154 annotation sets, 8,199 filaments; 180 test images. 411 images labelled
once, 145 twice, 151 three times. Chirality classes Left/Right/Unidentifiable (Ambiguous = 0). Every
filament is **one polygon**, `iscrowd = 0` (so we never merge polygons). No files missing.

### §3 — Image properties per site, and noisy images
*Why:* check the six telescopes don't produce systematically different images, and that the test set isn't
different from training.
*Detail:* we find the Sun by fitting a circle to the true **solar limb** (edge), tracing 720 rays. A naive
threshold would lock onto the bright **halo ring** outside the Sun instead — our first version did exactly
that and wrongly reported radius ≈ 1000 px.
*Found:* the Sun is already **centred and uniform** — limb radius **905 px ± 1 px**, centre within ~1 px of
the image centre. **Train and test look the same per site → no distribution shift.** 9 images had a shaky
limb fit and fall back to the default geometry.
*Noisy images:* **87 of 887 flagged** (66 train, 21 test) — mostly dark regions, uneven lighting, a few
blurry. They're real artefacts (cloud/haze) but the filaments are still visible, and the **test set has
the same artefacts at the same rate**. Decision: **keep them all**.

### §5 — Filament size, class imbalance, instance separation
*Found:* filaments are **thin** — median width **~11 px**, thinnest 5% under ~5.5 px. **Huge class
imbalance:** filaments are only **0.36% of a picture → ~278 background pixels per filament pixel.**
**Touching filaments are rare** (0.2%), and a *perfect* mask split into blobs already scores **PQ 0.997**.
*Meaning:* the hard part is **not** separating touching filaments — it's keeping each one in **one piece**
(not fragmenting) and not inventing extras.

### §6 — How much annotators disagree (the human ceiling)
*Found:* human-vs-human **PQ ≈ 0.34**; they mostly disagree on *whether* something is a filament (not just
edges), differing by ~2 filaments per image. A PQ of 0.34–0.35 is roughly human-level.

### §7 — How much resolution can we lose?
*Test:* shrink perfect labels, enlarge back, score.
*Found:* 2048 → PQ 1.000, 1024 → 0.884, **512 → 0.700.** Shrinking to the usual 512 throws away ~30% of
the achievable score before the model even runs. Decision: **keep native resolution**, train on 512×512
crops, predict with a sliding window.

### §8 — A validation split we can trust
*Why:* the brief forbids splitting one observation's sets across train/val, and warns that images close in
time or from the same instrument aren't independent (filaments live for days and the Sun rotates in ~27
days, so nearby images show the *same* filaments).
*What we do:* group by **Carrington rotation** (solar-rotation number) and **stratify by site**; 5 folds;
assert no observation or rotation is shared.
*Found:* balanced folds (138–145 images each), all sites in every fold, **leakage checks pass**, median
val-to-train time gap 12–15 days.

### §9 — Preprocessing
*What:* centre the Sun + scale to radius 905 (nearly a no-op since GONG already did this), stretch
brightness to [0,1], set the sky to 0. Store the transform so predictions map back to the original grid.
*Found:* the forward-then-back **round trip scores PQ 0.9997** → mapping predictions back loses nothing.
The part that actually helps is **brightness normalisation**.

### §10 — Augmentation and class imbalance
*What:* rotation (round disk), flips (note: a flip swaps Left/Right chirality), small scale, brightness/
contrast/blur/noise — all physically justified. **No elastic deformation** (it would distort barbs).
Crops are **centred on a filament** with probability `p_fg` to fight the 278:1 imbalance.
*Found:* at `p_fg = 0.7`, filament pixels per crop rise to 1.83% and empty crops fall from 27% → **7%**.

### §10b — Matching Kaggle's scoring exactly
*Found:* our PQ = organisers' code PQ = **0.34327** (identical); the mask→RLE submission encoder is
verified lossless. So we can score locally and trust it.

### §11–13 — Export, PyTorch dataset, summary
We save everything the modelling milestone needs into `prepared/` (images, per-set instance + boundary
masks, soft masks, fold assignments, and the ground truth in Kaggle format), wrap it in a ready-to-use
PyTorch `Dataset`, and print a one-screen summary. The folder is regenerable by re-running the notebook.

---

## 8. One-screen summary

| Thing | Value |
|---|---|
| Observations / annotation sets / filaments (train) | 707 / 1,154 / 8,199 |
| Test observations (no labels) | 180 |
| Images labelled by 2 / 3 annotators | 145 / 151 |
| Image | 2048×2048 grayscale H-alpha, 6 GONG sites, 2011–2022 |
| Solar disk radius | 905 ± 1 px (already centred by GONG) |
| Median filament width | ~11 px |
| Class imbalance | ~278 background : 1 filament pixel |
| Touching filaments | 0.2% (separation is NOT the hard part) |
| Human-vs-human PQ (ceiling) | ~0.34 |
| PQ ceiling at 512 / 1024 / native | 0.70 / 0.88 / 1.00 |
| Flagged noisy images (train / test) | 66 / 21 — all kept |
| Split | 5-fold, grouped by solar rotation, stratified by site, no leakage |
| Local PQ vs Kaggle's code | 0.34327 = 0.34327 (identical) |

**Bottom line:** we understand the data, our masks and scoring match Kaggle exactly, the split is
leakage-free, and the prepared dataset is ready for modelling.
