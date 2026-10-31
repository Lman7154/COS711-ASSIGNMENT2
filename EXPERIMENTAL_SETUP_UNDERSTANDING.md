# Understanding the Experimental Setup — one complete read

The single document for the modelling side, the counterpart to `data_prep/DATA_UNDERSTANDING.md`. It
explains in plain language how we went from the prepared data to a chosen model and a `submission.csv`,
why each piece is built the way it is, what we found, what went wrong along the way, and what's still open.

Files it describes (all in `modelling/`):

| File | What it is |
|---|---|
| `filament_pipeline.py` | All reusable code: data loading, models, losses, training, scoring, inference, submission |
| `verify_pipeline.py` | 14 checks from the raw Kaggle files through data prep to the metric |
| `run_experiments.py` | The 16 sweep runs (losses, architectures, second seeds, learning rates) |
| `run_5fold.py` | The chosen configuration on all 5 folds |
| `rescore_runs.py` | Re-scores every saved model after the sky fix (§9, fault 11) |
| `run_postprocess.py` | Post-processing tuning, test-time augmentation, timing, `submission.csv` |
| `experimental_setup.ipynb` | Reads all saved results, plots and interprets them ("What & why" + "Interpretation" per cell) |
| `submission.csv` | The Kaggle file from the final model |

---

## 1. The headline results

| | Value |
|---|---|
| **Chosen model** | U-Net with a pretrained **EfficientNet-B3** encoder, BCE+Dice loss, learning rate 3×10⁻⁴ |
| **Final held-out PQ** | **0.396** (99 report images, never used for any choice), with threshold 0.6, min_area 500, flip TTA |
| **5-fold PQ** (before post-processing tuning) | **0.298 ± 0.011** |
| Human agreement (annotator vs annotator) | ~0.34. The final model is above it on held-out images |
| Model size / speed | 13.2M parameters (53 MB); 2.1 s per test image with TTA (0.6 s without); 21 min training |
| Pipeline correctness | 14/14 checks pass; our PQ equals the organisers' scoring code to 6 decimals |
| `submission.csv` | 1,474 filaments, 179 of 180 test images, passes every Kaggle format rule |

---

## 2. Two PQ numbers that are easy to confuse

| Number | What it is | What it is NOT |
|---|---|---|
| **~0.34** (data prep) | One annotator's outlines scored against the *other* annotators. It shows how much humans agree, a realistic ceiling. The same comparison verified our PQ code against the organisers' (0.34327 = 0.34327). | **Not a model score.** No model was involved. |
| **0.396** (this milestone) | How well our final model's filaments match the expert outlines on held-out images. | Not the Kaggle leaderboard score; that needs an upload. |

### What the model detects, and what PQ measures
For every pixel the model gives a probability of "filament". We threshold that into a mask, split the mask
into connected blobs, drop blobs smaller than `min_area`, and **each remaining blob is one detected
filament**. PQ then scores two things at once, **PQ = SQ × RQ**:
- **RQ (recognition):** did we find each real filament as its own object, without inventing extras?
  A real filament counts as found only if one prediction overlaps it with IoU > 0.5.
  RQ = TP / (TP + ½FP + ½FN).
- **SQ (segmentation):** for the filaments we found, how well does our outline overlap the expert's?

Final model: SQ 0.673 × RQ 0.588 = PQ 0.396. Outlines were good from the start; most of the improvement
came from **counting** better (fewer invented filaments).

---

## 3. Why there were "candidates", and why the baseline was a U-Net

The brief says choices must be "informed by the data and by experiments". So we didn't pick an architecture
on paper: we listed sensible **candidates**, trained them all under identical rules, and let local PQ
choose. "Candidate" means "an option we will test". Having candidates is how you *avoid* guessing.

**Framing.** PQ needs one mask per filament. We produce that as a *semantic* mask (filament / not
filament) split into blobs, rather than with a heavy instance model, because the data prep showed it's
enough: only 0.2% of filaments touch, and a perfect semantic mask split into blobs already scores PQ 0.997.
It's also cheaper, and efficiency is marked.

**Why a U-Net baseline.** A baseline is the simple, sensible model every later idea must beat. U-Net was
chosen because of the data, not habit: filaments are thin (~11 px wide, with barbs), and U-Net's skip
connections carry fine detail from the early layers straight to the output; it's light; and it's the
standard reference for fine structures in scientific images. But a fair setup must also be able to prove
the baseline wrong, so the candidate list includes models that are **not** U-Nets.

### CNN vs FCN vs U-Net
- **CNN** (Convolutional Neural Network) is the whole family. U-Net, FCN, DeepLab, PSPNet, ResNet are all CNNs.
- A **plain CNN** like ResNet is built for **classification**: it ends in a classifier with one answer per
  image. On its own it can't draw a mask.
- An **FCN** (Fully Convolutional Network, Long et al. 2015) is the minimal way to segment with that plain
  CNN: replace the classifier with a 1×1 convolution that scores every location, and stretch the result back
  to full size. **"Bare CNN for segmentation" = FCN-32s.** FCN-8s adds score maps from two earlier layers.
- **U-Net** = the same backbone + a full decoder with skip connections at every level.

### Every standard segmentation model, considered
Checked against the GeeksforGeeks overview "Image Segmentation Models" (23 Jul 2025):

| Model family | In our setup? | Result / reason |
|---|---|---|
| CNN backbone (bare) | Yes: FCN-32s | **Worst, PQ 0.107**: its 32×32-pixel output grid is too coarse for ~11 px filaments |
| FCN | Yes: FCN-8s | 0.268, below the U-Nets |
| U-Net | Yes: ResNet34, EfficientNet-B3, from scratch, U-Net++ | **Winner family** |
| SegNet | No | Like U-Net but passes only pooling positions to the decoder, not the features; U-Net keeps more detail |
| DeepLabV3+ | Yes | 0.294, but unstable across seeds |
| PSPNet | Yes | 0.246; pyramid pooling targets large regions and averages away thin structures |
| Mask R-CNN | Not built | Direct instances; only worth its cost if errors were merged neighbours (merging is rare: 15 cases) |
| ViT / Swin Transformer | Not yet | Usually need far more than 707 images and more compute |
| Traditional (thresholding, Otsu, watershed, k-means) | Not as models | The brief requires deep learning; thresholding is used inside our post-processing |

On metrics: IoU and Dice are inside our evaluation (PQ matches by IoU > 0.5; Dice is part of the loss).
**Pixel accuracy** fails here, as the article warns for imbalanced data: an empty prediction scores 99.6%
and finds zero filaments. That's why Kaggle uses PQ.

---

## 4. How every run is trained and judged (the protocol)

- **Data:** 568 training images (folds 1–4). One **epoch** = 800 random 512×512 crops. 70% are centred on a
  filament, the rest on a random point **of the solar disk**. Each crop gets a random rotation, flip, small
  scale change, brightness/contrast/blur/noise, and a **randomly chosen annotator's** labels.
- **Epochs ≠ annotators.** An epoch is a number of crops, not "one pass per annotator". One real pass over
  the 568 images is ~5,700 crops (each disk holds ~10 crops' worth of area). Re-drawing the annotator each
  time is how we treat annotator disagreement (human PQ ~0.34) as useful noise.
- **Loss:** computed only on the solar disk (the sky carries no information).
- **Optimiser:** AdamW, learning rate 3×10⁻⁴, **halved when validation PQ stalls for 2 epochs**; mixed precision.
- **Early stopping:** validate after every epoch; keep the best checkpoint; stop after 5 epochs without a
  gain of at least 0.002; at most 30 epochs. No fixed epoch count: the number is measured per run.
- **Choosing vs reporting.** Validation fold 0 (139 images) is split in two:
  - **40 monitor images**, spread evenly over 2011–2022 and all six sites: used for **every** choice
    (early stopping, which loss, which architecture, post-processing).
  - **99 report images**: scored for every run but **never** used to choose anything. All reported numbers
    come from these, so they aren't inflated by selection.
  - Check: ranking all 16 runs by monitor PQ agrees with ranking them by report PQ (Spearman 0.93), and
    report PQ is on average slightly higher (+0.009), so choices on 40 images generalise. Differences under
    ~0.01 PQ are within noise.
- **Scoring:** predictions are mapped back to the **original** 2048×2048 frame and scored against **every**
  annotator's labels, pooled, exactly like the organisers' code.

---

## 5. What the experiments chose, and why

### Loss (U-Net ResNet34, all else equal)

| Loss | Monitor PQ | Report PQ | Note |
|---|---|---|---|
| **BCE+Dice** | **0.307** | **0.317** | Chosen: best on both sets |
| Focal+Dice | 0.286 | 0.304 | 30% slower to train |
| Dice | 0.265 | 0.269 | Peaks early, then degrades |
| Tversky (β 0.7) | 0.258 | 0.279 | Fewest fragments but misses the most filaments |

Pure overlap losses (Dice, Tversky) give a weak, unstable signal on crops with no filament (13% of crops);
the BCE term adds a steady per-pixel signal.

### Architecture (BCE+Dice, all else equal)

| Model | Params | Monitor PQ | Report PQ |
|---|---|---|---|
| **U-Net EfficientNet-B3** | **13.2M** | **0.312** | 0.309 |
| U-Net ResNet34 | 24.4M | 0.307 | 0.317 |
| DeepLabV3+ ResNet34 | 22.4M | 0.294 | 0.303 |
| U-Net++ ResNet34 | 26.1M | 0.284 | 0.304 |
| FCN-8s ResNet34 | 21.3M | 0.268 | 0.266 |
| FPN ResNet34 | 23.1M | 0.265 | 0.269 |
| PSPNet ResNet34 | 21.4M | 0.246 | 0.237 |
| U-Net from scratch | 7.8M | 0.232 | 0.241 |
| FCN-32s (bare CNN) | 21.3M | 0.107 | 0.119 |

- The two U-Nets lead and are tied within noise (their order flips between the two image sets).
  **EfficientNet-B3 wins the tie by size: 46% fewer parameters**, and efficiency is marked.
- **A decoder is necessary** (bare CNN 0.107). **Pretraining helps** (+0.06 over the same U-Net from scratch).
  **More decoder complexity didn't help** (U-Net++, FPN below plain U-Net and slower).
- The short 4-epoch tests early on suggested FCN-8s was best. That was an artefact of short training: FCN-8s
  has almost nothing to learn beyond its pretrained encoder, so it starts fast but plateaus low.

### Reproducibility (second seed for the top two)
- **U-Net EfficientNet-B3:** seeds give 0.312 / 0.310 (monitor) and 0.309 / 0.314 (report). Stable.
- **DeepLabV3+:** seed 7 dropped to 0.233 (report 0.245), 0.06 below seed 42. Its validation PQ swings by up
  to 0.13 between epochs. Its good first result was partly luck.

### Learning rate (U-Net EfficientNet-B3)
- **3×10⁻⁴ kept** (0.312, best epoch 12, 21 min).
- 1×10⁻⁴: report PQ level (0.315) but lower on monitor (0.295), ran all 30 epochs, 50% slower.
- 1×10⁻³: too high, unstable (0.251).

### 5-fold cross-validation (chosen configuration)
Report PQ per fold: 0.309, 0.302, 0.298, 0.299, 0.279 → **0.298 ± 0.011**. Best epochs per fold: 12, 6, 15,
9, 13, so a fixed epoch count would have been wrong for some folds. Fold 0 (where all the choices were made)
is the best fold, a reminder that single-fold numbers are slightly optimistic.

### Post-processing and test-time augmentation (tuned on monitor, reported on report images)

| Setting | PQ | False filaments | Filaments per image |
|---|---|---|---|
| default (threshold 0.5, min_area 60) | 0.309 | 1,466 | 14.0 |
| tuned (threshold 0.6, min_area 500) | 0.381 | 670 | 8.6 |
| **tuned + flip TTA** | **0.396** | **620** | 8.3 |

- The **size filter** does most of the work: it removes more than half the false filaments. 500 px sits
  inside the tested grid (60–1,200), so it isn't an edge value.
- **Hysteresis thresholding and morphological closing did not help**: they joined neighbouring blobs more
  than they repaired broken filaments.
- **TTA** (averaging over 4 flips) adds +0.015 for 3.4× the inference time. Kept for accuracy; first thing to
  drop if efficiency matters more.
- The annotators draw 7.5 filaments per image; the final model predicts 8.3 (before tuning: 14.0).

---

## 6. What errors remain (hand-off to the Enhancement milestone)

On the 99 report images: **732 of 1,136 real filaments found (64%)**, **54% of predictions are real**,
620 false filaments, 404 missed. Fragmentation affects 69 real filaments (6%); merging only 15.

- **Missed faint filaments** are the largest error. Part of this is inherent: annotators disagree on which
  faint structures count.
- **Filament-like false detections** (dark patches) are the second.
- **Long, thin filaments broken into pieces:** the worst image shows a long winding filament detected as
  several short segments. No piece covers half of it, so it's missed *and* the pieces count as false. This
  is the "structural continuity" difficulty the competition names, and simple gap-filling didn't fix it.
- **By telescope:** all sites score 0.35–0.44. Mauna Loa is hardest (0.350), as it was for the human
  annotators; Udaipur is easiest (0.441).

Targets for Enhancement: keep long thin filaments in one piece, recover faint ones, then compress the model
(the 13.2M-parameter model and its timing are the baseline).

---

## 7. From model to submission

1. Each test image is preprocessed exactly as in training (the transform is stored per image).
2. A 512×512 window slides over the image with 50% overlap; overlapping predictions are blended with a
   Gaussian weight (tile centres count more than tile edges, which have less context); tiles that are
   entirely sky are skipped. With TTA, each tile is also predicted flipped 3 ways and averaged.
3. The sky is set to 0 (the model was never trained on it; see fault 11).
4. Threshold 0.6 → mask → 8-connected blobs → drop blobs under 500 px.
5. Each blob is mapped back to the original 2048×2048 frame and RLE-encoded with pycocotools.
6. `submission.csv`: one row per filament, `filament_id = <image>_<k>`.

The file passes every format rule (unique ids, only test images, RLE decodes to 2048×2048, no quotes).
One test image (20200310113230Ch) has no predicted filament, which is allowed.

---

## 8. Verification: raw data to metric (`verify_pipeline.py`, 14/14 pass)
1. prepared counts match the raw Kaggle files (707 images, 1,154 annotation sets, 8,199 filaments, 180 test);
2. all 8,199 ground-truth masks are identical to the raw polygons drawn with pycocotools;
3. no image or solar rotation is in two folds; test images carry no labels;
4. every image has its preprocessing transform;
5. prepared masks map back onto the raw ground truth (PQ 0.9997);
6. the fast cache is byte-identical to the prepared files;
7. the fast crop equals a full warp (0.016% of pixels differ, rounding ties only);
8. training labels sit on darker pixels (on real filaments) in 100% of tested crops;
9. crops sit on the solar disk (92% disk coverage) and the loss mask's off-disk area is pure sky;
10. crops are filament-rich (1.39% filament pixels; 13% without filament); 15–22 ms per crop;
11. monitor and report images are disjoint and the monitor set spans 2011–2022, all sites;
12. a perfect prediction scores 0.997 through our scoring code;
13. our PQ equals the organisers' code on real model predictions (0.322735 = 0.322735);
14. no predicted filament pixel lies off the solar disk.

---

## 9. The audit: what was wrong, and what we fixed

A critical review of the first version found faults that made our scores untrustworthy or held them down.

| # | Fault | Fix | Effect |
|---|---|---|---|
| 1 | Validation compared predictions in the preprocessed frame with ground truth in the original frame | Map predictions back before scoring | A perfect prediction scored only 0.864; now 0.997 |
| 2 | Blobs split with 4-connectivity, cutting thin diagonal filaments | 8-connectivity | Perfect-mask ceiling 0.980 → 0.997 |
| 3 | Early stopping used the first 40 fold-0 images, all from 2011–2014 | 40 monitor images spread over 2011–2022 | Representative selection |
| 4 | 36% of background crops centred on black sky | Background crops sampled on the disk | Useful training crops |
| 5 | LR schedule sized for 30 epochs, but runs stopped at ~15, so the LR barely decreased | Halve LR when PQ stalls | The LR actually anneals |
| 6 | The loss included the sky | Disk-masked loss | Matches the methodology |
| 7 | The same images chose the checkpoint and gave the reported score | Monitor / report split | Honest reported numbers |
| 8 | One run per configuration | Second seed for the top two | Exposed DeepLabV3+'s instability |
| 9 | Loss and architecture sweeps never actually trained | All 16 runs trained to early stopping | Real evidence for every choice |
| 10 | Slow loading (a 4-megapixel PNG decoded per crop) | Memory-mapped cache | 15–22 ms per crop |
| 11 | **Found after the sweep:** with the sky excluded from the loss, some models output ~1 on the sky, which became one giant false "filament" per image | Inference zeroes the sky; every saved checkpoint re-scored | +0.013–0.016 PQ for the 6 affected models |

Same model before and after the audit (U-Net ResNet34, BCE+Dice): **0.246 → 0.317**.

Fault 11 is worth remembering: it was caught by **looking at predicted masks**, not by any metric. A plot
of the best and worst images showed no visible predictions, because each image's prediction was dominated by
one sky-sized blob. Checking the metric alone would never have revealed it.

---

## 10. Stated limitations
- Early stopping chose each checkpoint before fault 11 was found; the checkpoints were re-scored, not
  re-chosen (that would need retraining). The ranking it produced still holds.
- Only the top two candidates got a second seed, and U-Net ResNet34 (equally accurate) got only one.
- The final 0.396 uses post-processing tuned on fold 0; the 5-fold number (0.298) is before tuning. With the
  tuned post-processing (0.5/500) and TTA the 5-fold number is **0.368 ± 0.019** (`improvements.ipynb` §2):
  fold 0 is the best fold.
- Kaggle public leaderboard for `submission.csv`: **0.33** (rank 789), within about two fold standard
  deviations of the 5-fold estimate.
- Validation PQ is noisy between epochs (40 monitor images); patience 5 can cut short a volatile model.
- Not explored: crop size, `p_fg`, augmentation strength, batch size (defaults from data prep).

---

## 11. How to reproduce

From `modelling/` (each script is resumable: finished runs are skipped):
```
python verify_pipeline.py      # ~5 min; also builds the 8 GB loading cache on first run
python run_experiments.py      # ~5 h: loss + architecture sweeps, seeds, learning rates
python run_5fold.py            # ~1.5 h
python rescore_runs.py         # ~30 min (only needed once, for runs trained before the sky fix)
python run_postprocess.py      # ~25 min: tuning, TTA, timing, submission.csv
python run_improve.py          # ~4 h: fair 5-fold estimate, small-blob rule, ensembles, clDice,
                               #        all-data model, candidate files in submissions/
```
Then `improvements.ipynb` reads `runs/improve/` (~5 min).
Then open `experimental_setup.ipynb` and run all cells (~1 min; it reads the saved results).

Notes: the scripts use 4 data-loading workers, which only works from a script on Windows (not inside a
notebook). The `cache/` folder (8 GB) can be deleted and is rebuilt automatically. `runs_prefix/` holds the
pre-audit results and the old notebook, for reference. If a cell says `No module named 'cv2'`, the notebook
is on a Python without the packages: `pip install opencv-python-headless segmentation-models-pytorch pycocotools scikit-image`.

---

## 12. Open decisions for the group
1. ~~Upload `submission.csv`~~ Done: 0.33. Next, upload the candidates in `modelling/submissions/` in the
   order given in `improvements.ipynb` §7.
2. Keep TTA (+0.015 PQ, 3.4× slower inference), or drop it for efficiency?
3. Run a second seed for U-Net ResNet34 before final selection, or accept EfficientNet-B3 on size?
4. Enhancement priorities: filament continuity, faint filaments, or compression first?
