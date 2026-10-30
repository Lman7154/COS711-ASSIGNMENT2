# Data Preparation: Decision Log

Each decision is written as: **Decision → Why → Source → Evidence (where to find it in `data_prep.ipynb`)**.

Source key:
- **[BRIEF]** COS711 Assignment 2 PDF (`COS711-A2 (1).pdf`)
- **[KAGGLE]** official competition pages (Data, Evaluation, Leaderboard Ranking, Submission File), saved in `kaggle_pages/pages.md`
- **[SELF-EVAL]** the organisers' scoring code, saved in `kaggle_pages/self-evaluation-notebook.ipynb`
- **[PAPER]** Ahmadzadeh et al., *MAGFiLO*, Scientific Data 11:1031 (2024)
- **[DATA]** our own measurement in the notebook (section number given)

All numbers come from the final verified run (all 49 cells passed, 12 min), which uses Kaggle-exact pycocotools masks.

---

## 0. What the data is (facts every decision builds on)

| Fact | Value | Source / Evidence |
|---|---|---|
| Label file | One COCO-style JSON: `images`, `annotations`, `categories` | [KAGGLE] Data page; [DATA] §1 |
| A COCO "image" is an **annotation set**, not an observation | id = `<annotator code>-<file stem>`, e.g. `010101-20141027025054Uh` | [KAGGLE] Data page: "may have two sets of annotations … treat those images simply as different images"; [DATA] §1 assertion `set_id == set_code + "-" + obs` |
| Train size | 707 observations, 1,154 annotation sets, 8,199 filaments | [DATA] §1 output |
| Annotation sets per observation | 1 set: 411, 2 sets: 145, 3 sets: 151 | [DATA] §1 output |
| Why multiple sets exist | Double-blind protocol: groups of 3 annotators labelled copies of the same batch | [PAPER] "Manual Annotation" section |
| Test size | 180 observations, no labels | [DATA] §1 |
| Images | 2048×2048, 8-bit, **grayscale** JPEG, full solar disk in H-alpha | [KAGGLE] Data page ("should not be processed as color images"); [DATA] mode `L`, 2048×2048 for all |
| Instruments | 6 GONG sites, letter in file name: B, C, L, M, T, U | [KAGGLE] Data page (`YYYYMMDDHHMMSSII`); [PAPER] site list |
| Years | 2011–2022, same mix in train and test | [DATA] §1 bar plots |
| Annotation contents | One polygon per filament (no holes, no islands), bbox, area, spine polyline, chirality label | [KAGGLE] Data page; [PAPER] "segmentation is in one piece … contains no holes" |
| Categories | Left 2,535, Right 2,590, Unidentifiable 3,074, Ambiguous 0 | [DATA] §1; [PAPER] "no filaments are labeled as ambiguous" |
| Limb darkening | Already removed by GONG with a Fourier filter | [PAPER] "large scale variations of intensity (e.g., solar limb darkening) are removed" |

---

## 1. How we are scored (this drives the local metric and output format)

**Decision:** implement PQ exactly as Kaggle does, and export a submission encoder in Kaggle's format.

- **Why:** The brief requires a local PQ implementation ("Implement the Panoptic quality metric locally"). A local metric that differs from Kaggle's would mislead every later experiment.
- **Source:** [BRIEF] §3.1; [KAGGLE] Leaderboard Ranking + Submission File; [SELF-EVAL] `get_pq_score`, `get_overlap_df`.
- **Evidence (what Kaggle actually does, read from their code):**
  1. Every **annotation set** is scored separately against all predictions for that image ([SELF-EVAL] `process_entry` splits `annotator-image`).
  2. A match is IoU **> 0.5**. TP IoUs, FP and FN are **pooled over all sets**, then PQ = ΣIoU / (TP + ½FP + ½FN) ([SELF-EVAL] `get_pq_score`).
  3. Class-agnostic: the categories are never used ([SELF-EVAL]).
  4. Submission: one row per predicted filament, `filament_id = <stem>_<k>`, `segmentation_rle` = pycocotools RLE counts string ([KAGGLE] Submission File).
  5. Public leaderboard = ~50% of the test set ([KAGGLE] Submission File).
- **Verification in the notebook (§10b):** on 93 real annotation sets, organisers' code PQ = **0.34327** and our PQ = **0.34327** (identical). RLE encode → decode is asserted lossless. `prepared/train_gt_rle.csv` holds all 8,199 GT filaments in Kaggle format.

### Bug found and fixed: mask rasterisation
- **Problem:** We first drew polygons with OpenCV `fillPoly`. On 1,500 random filaments its masks matched pycocotools at only **IoU 0.878 mean (min 0.589)**. Kaggle's ground truth is made with pycocotools ([KAGGLE] Data page: "area … computed using pycocoapi"; [SELF-EVAL] decodes pycocotools RLE).
- **Fix:** All masks are now drawn with `pycocotools.mask.frPyObjects` (`ann_mask` in §2). pycocotools areas match the JSON `area` field exactly (max difference 0.0).
- **Why it matters:** Thin filaments (~11 px wide) lose a large fraction of their pixels to a 1-px edge disagreement, so the old local PQ would have been biased.

---

## 2. D1 — Train/validation split

**Decision:** 5-fold `StratifiedGroupKFold`. Groups = **Carrington rotation** (one full solar rotation, ~27.3 days); stratified by **site**. Fold 0 is the default hold-out; all 5 folds are used for mean ± std.

| Requirement | Source | How it's handled | Evidence |
|---|---|---|---|
| Annotation sets of one observation never split | [BRIEF] §3.1 | The split unit is the observation; all its sets inherit its fold | §8 assertion "an observation spans folds" passes |
| Observations close in time are not independent | [BRIEF] §3.1 | Group by solar rotation, so the same filament seen on several days stays in one fold | §8 assertion "a rotation spans folds" passes |
| Why time matters physically | [PAPER]; solar physics | Filaments live days to weeks and cross the disk in ~13.6 days | [DATA] §1: 119 consecutive train observations < 24 h apart; 31 test days also appear in train |
| Same instrument is not independent | [BRIEF] §3.1 | Every fold contains all 6 sites in the same proportions as the test set | §8 crosstab: each site 14–19% in every fold |

- **Evidence of balance (§8 table):** 138–145 observations per fold, 24–28 rotations per fold, mean coverage 0.30–0.39%.
- **Evidence of separation (§8 gap table):** median gap from a validation image to the nearest training image is **11.8–15.0 days**.
- **Known limitation, stated openly:** 3–8% of validation images are < 3 days from a training image, at rotation boundaries (minimum gap 0.55 days). This could leak a little if a filament crosses a rotation boundary. Mitigation if CV and leaderboard disagree: drop training images within N days of each validation block.
- **Alternatives rejected:**
  - Random split by observation: leaks the same filament across days.
  - Split by year: the 2018–2022 years are small (14–42 images), so folds would be very unbalanced.
  - Leave-one-site-out as the main split: the test set contains all sites, so it would not match the test distribution. It stays an optional robustness check.

---

## 3. D2 — Multiple annotators (label noise)

**Decision:** During training, each sample uses **one randomly chosen annotation set** of that observation. Validation scores against **every** set (like Kaggle). A soft mean-of-annotators mask is also exported.

- **Why:** Kaggle scores against every set ([SELF-EVAL]), so validation must too. Picking one set per sample keeps instance labels internally consistent, unlike merging polygons from different people. The variation between annotators then works as label-noise regularisation.
- **Evidence — human agreement (§6, 598 annotator pairs from 296 observations):**
  - PQ between two humans: **0.341 ± 0.165** per pair; **0.340** pooled the Kaggle way; per site 0.325 (Mauna Loa) to 0.367 (Big Bear)
  - Segmentation quality (SQ): 0.597 · Recognition quality (RQ): 0.538 · pixel Dice: 0.614
  - Filament count differs by **2.0 on average** between annotators
- **What this means:** The humans disagree mostly on *whether* a faint structure is a filament (low RQ), and also on edges (SQ). ~0.34 is a realistic ceiling. It matches the organisers' note that "any PQ score of greater than 0.35 is of great value" ([KAGGLE] Announcements, Aug 20 2026).
- **Source for the protocol:** [PAPER] double-blind groups of 3; Cohen's kappa reported for chirality.
- **Alternatives kept available:** soft target (`prepared/soft/`), majority vote (`soft ≥ 0.5`).

---

## 4. D3 — Working resolution

**Decision:** Keep **native resolution** (solar disk radius ~905 px). Train on **512×512 patches**, infer with an overlapping sliding window.

- **Why:** Filaments are thin, and downsampling destroys them before the model even sees them.
- **Source:** [BRIEF] / [KAGGLE] both name "fine-scale structures" as a main difficulty.
- **Evidence (§5):** median filament width ≈ **11 px** (area ÷ spine length), 5th percentile ≈ 5.5 px.
- **Evidence (§7, PQ of *perfect* labels after down- then up-sampling, 150 sets):**

| Input size | PQ ceiling | Median width at that size |
|---|---|---|
| 2048 | 1.000 | 11.1 px |
| 1536 | 0.846 | 8.3 px |
| 1024 | 0.884 | 5.5 px |
| 768 | 0.712 | 4.2 px |
| 512 | 0.700 | 2.8 px |

  A model trained on 512-resized full disks would lose ~30% PQ before making any mistake. (1536 scores below 1024 because 2048→1536 is not an integer ratio, so nearest-neighbour resampling distorts thin shapes more. If we downsample at all, an integer factor is preferred.)
- **Trade-off:** Native resolution costs more compute, and Kaggle also scores end-to-end efficiency ([KAGGLE] abstract; [BRIEF] §2). This goes into the compression milestone. 1024 is a possible middle ground.

---

## 5. D4 — Class imbalance

**Decision:** **Filament-biased patch sampling** (70% of patches centred on a filament pixel, `p_fg = 0.7`). Combine with an overlap-based loss (Dice/Tversky/Lovász, chosen in methodology). Compute loss only on the disk.

- **Why:** At this ratio, plain pixel-wise BCE learns to predict "background everywhere".
- **Source:** [BRIEF] presentation must cover "class imbalance"; lecture on data prep (L3).
- **Evidence (§5):** foreground coverage **0.36%** mean (median 0.30%, max 1.67%) → **≈278 : 1** background : foreground.
- **Evidence (§10, one observation, 300 patches each):**

| p_fg | Foreground per 512 patch | Empty patches |
|---|---|---|
| 0.0 (uniform) | 0.71% | 27% |
| 0.5 | 1.34% | 16% |
| 0.7 (chosen) | 1.83% | 7% |
| 0.9 | 2.05% | 4% |

- **Why not 0.9:** It barely adds foreground over 0.7, and the model sees less pure background. Background matters, because false positives cost ½ each in PQ. `p_fg` is a hyperparameter to tune on fold 0.

---

## 6. D5 — Noisy / anomalous observations

**Decision:** **Keep all images by default.** Flag suspicious ones automatically for visual review; `EXCLUDE_TRAIN` lists any the group decides to drop.

- **Why keep:**
  1. MAGFiLO already removed strongly anomalous observations before annotation, using an 8×8-cell intensity detector plus manual reporting ([PAPER] "Anomaly Detection and Removal").
  2. The **test set has the same kinds of artefacts**, so training on mild noise builds robustness. [KAGGLE] names "background noise and image quality" as a main difficulty.
  3. Dropping images also drops expert labels we can't replace.
- **Detector (§3):** robust z-scores (median/MAD) on limb radius, centre offset, sharpness (Laplacian variance), uneven illumination (std of 8×8 cell means, as in [PAPER]) and dark-region fraction. A flag is also raised when the limb fit fails. Saturation uses an absolute rule (> 0.1% saturated disk pixels).
- **Bugs found and fixed:** (1) The first saturation check flagged 316 images, because most disks have 0 saturated pixels, so the MAD was 0 and the z-scores exploded. (2) The first disk detector measured the instrument halo ring instead of the Sun (see D6).
- **Result:** **87 of 887 images flagged (66 train, 21 test)**: dark regions 71, uneven 31, off-centre 15, limb fit failed 9, odd radius 4, blurry 3, saturated 1. The gallery shows real cloud/haze shading and a few odd exposures, but filaments stay visible in all of them. The dark-region check is sensitive on purpose, since large filaments also create dark area.
- **Why this supports keeping them:** the test set is flagged at the same rate (12% vs 9%), so the model will meet these artefacts at scoring time.
- **Action for the group:** look at the §3 gallery. The default is to keep everything.

---

## 7. D6 — Preprocessing

**Decision:** Per image: (1) detect the solar limb (circle fit to the strongest outward brightness drop along 720 rays), (2) one affine warp that centres the limb and scales its radius to the median 905 px, (3) normalise intensity to [0,1] using on-disk 1st–99th percentiles, (4) set the sky outside the disk to 0. Store the transform and invert it for the submission.

- **Why:** Remove instrument differences that carry no filament information, so the model only learns what's on the disk.
- **Evidence (§3, all 887 images):**
  - **the Sun is already centred and equally sized by GONG:** limb radius **905 px ± 1 px**, centre within **0.9 px** median (99th pct 8 px) of the image centre. Site differences are under 0.3%.
  - the bright **halo ring** around the Sun varies a lot (997 ± 27 px). Our first detector measured this ring by mistake and reported "±2.5% radius differences" that don't exist. Fixed and verified: 0% of annotated pixels lie outside the detected limb.
  - 9 limb fits were unreliable and fall back to image centre + median radius (accurate to a few px, given the above).
  - on-disk brightness and contrast differ mildly between sites and days. Train and test have the same per-site statistics (§3 table), so there is no train/test shift.
- **Source:** [KAGGLE] grayscale requirement; [PAPER] limb darkening already removed (so we don't remove it again).
- **Evidence it is safe (§9):** round trip on ground-truth masks gives **PQ 0.9997 (min 0.998)**.
- **What actually matters:** geometry is nearly an identity, so the useful part is the **intensity normalisation**. Intensity-only normalisation without the warp would be an equally valid choice.

---

## 8. D7 — Augmentation

**Decision:** Rotation 0–360°, horizontal and vertical flips, scale ±5%, gamma 0.8–1.25, contrast ±15%, brightness ±0.05, Gaussian blur (p=0.3), Gaussian noise (p=0.3). **No elastic deformation.** Geometric augmentations are done as one affine warp per patch.

| Augmentation | Why (physical justification) | Source |
|---|---|---|
| Any-angle rotation | After centring, the disk is rotation-symmetric; filaments occur at all orientations | [DATA] §9 centred disk |
| Flips | Valid for class-agnostic masks. **Mirrors chirality** (Left↔Right) if chirality is ever used | [PAPER] chirality definition |
| Scale ±5% | Small size variation of filaments (site radius differences are < 0.3%, so not needed for that) | [DATA] §3 |
| Gamma / contrast / brightness | Site exposure and sky transparency differences | [DATA] §3 histograms |
| Blur | Bad seeing; blurry images exist in both train and test | [DATA] §3 sharpness; [PAPER] anomaly list |
| Noise | "Background noise" is a named difficulty | [KAGGLE] Description |
| No elastic deformation | Would distort barbs, the fine structure we're scored on | [KAGGLE] "Fine-scale structures" |

- **Evidence:** §10 gallery of augmented patches with their instance boundaries, which shows the labels stay aligned.

---

## 9. D8 — Instance separation (what the next milestones should focus on)

**Decision:** Export an **instance-boundary target** with every annotation set. Data prep shows that touching filaments are *not* the main instance problem; fragmentation and merging of close neighbours are.

- **Evidence (§5):**
  - only **0.2%** (20 of 8,199) of filaments touch another; 5 of 1,154 sets have overlapping polygons
  - connected components on a *perfect* semantic mask give **PQ 0.997 ± 0.016**
  - **1.4%** (116) of filaments lie within 5 px of another, and these are the cases a slightly blurry prediction will merge
- **Source:** [KAGGLE] and [BRIEF] both name "structural continuity" (fragments and "islands") and penalise one-to-many / many-to-one matches.
- **Implication:** The enhancement milestone should target **continuity**: boundary head, morphological closing or gap bridging, minimum-area filtering. A heavy instance-segmentation architecture isn't needed just to split touching instances.

---

## 10. What is handed over (`data_prep/prepared/`)

| File | Contents |
|---|---|
| `images/<obs>.png` | Canonical, normalised image (train + test) |
| `instances/<set_id>.png` | uint16 instance map per annotation set |
| `boundaries/<set_id>.png` | Instance boundary target |
| `soft/<obs>.png` | Mean over annotators (×255) |
| `train_observations.csv` | obs, site, time, rotation, **fold**, sets, anomaly flags, affine M |
| `train_sets.csv` | Per annotation set: fold, #instances, coverage, touching stats |
| `test_observations.csv` | Test obs + affine M (needed to map predictions back) |
| `train_gt_rle.csv` | Ground truth in Kaggle submission format + fold, for local scoring |
| `config.json` | Every preprocessing parameter |

Code in the notebook for later milestones: `FilamentPatchDataset` (§12), `panoptic_quality` / `global_pq` (§2), `encode_mask` / `instances_to_rows` / `official_pq` (§10b), `from_canonical` (§9).

## 11. Open questions for the group meeting
1. `EXCLUDE_TRAIN`: after viewing the §3 gallery, drop any of the 87 flagged images? (default: keep all)
2. Keep the (almost identity) limb-centring warp, or only normalise intensity? (round trip costs < 0.1% PQ)
3. Native resolution vs 1024: how much compute are we willing to spend, given the efficiency score?
4. Is the 3–8% near-boundary validation leakage acceptable, or do we add a time buffer?
