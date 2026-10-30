# Experimental Setup & Methodology (plan)

This is the plan for milestone 2 (worth 20 marks) and it sets up milestones 3–4 (Results 30, Enhancement
30). It is a **proposal grounded in what the data prep found**, for the group to react to and then run.
Nothing here is final: the brief requires every choice to be **justified by experiments**, so this
document says *what we will try and why*, and the experiments will decide the rest.

Read alongside: `data_prep/DATA_UNDERSTANDING.md` (what the data is) and
`data_prep/DATA_PREP_DECISIONS.md` (the prep decisions). Facts cited as "(§N)" are from the data-prep
notebook.

> **Status: this plan has been run.** Results, decisions and limitations are in
> `EXPERIMENTAL_SETUP_UNDERSTANDING.md` and `modelling/experimental_setup.ipynb`. In short: the experiments
> chose **U-Net + EfficientNet-B3, BCE+Dice, LR 3×10⁻⁴** (5-fold PQ 0.298 ± 0.011); with tuned
> post-processing (threshold 0.6, min_area 500) and flip TTA the held-out PQ is **0.396**. Hysteresis and
> closing were tried and didn't help; the boundary head and Mask R-CNN were not built (merging is rare).

---

## 1. How we frame the problem

Kaggle scores **Panoptic Quality (PQ)**, which needs **one mask per individual filament** (instances),
not just a filament-vs-background map. So our output must be *instances*. There are two honest ways to get
there, and the data prep tells us which to try first.

- **Semantic segmentation + split into instances** (our baseline). Predict a filament/background mask,
  then cut it into separate objects with connected components (+ a boundary cue for the few that touch).
- **Direct instance/panoptic model** (our stretch goal). A network that outputs instances directly
  (Mask R-CNN, Mask2Former, etc.).

**Why we start with the first:** the data prep showed touching filaments are rare (0.2%, §5) and that a
*perfect* semantic mask split by connected components already scores **PQ 0.997** (§5). So the instance
problem is mostly "get the mask right and keep each filament in one piece", not "pull apart overlapping
objects". That means a strong semantic model + good post-processing can score well for far less compute —
and the brief also marks **efficiency**, so starting light is the right call.

**Working representation (decided in data prep, §7, §9):** native resolution, train on **512×512 crops**,
run inference with an **overlapping sliding window**, then map predictions back to the original
2048×2048 grid with the stored inverse transform. Downsampling to 512 would cap PQ at ~0.70 before the
model makes a single error, so we don't.

---

## 2. Tools

- **Language/framework:** Python + **PyTorch** (already installed; CUDA works on the RTX 4060 locally).
- **Models:** `segmentation-models-pytorch` (U-Net / U-Net++ / FPN with pretrained encoders) for the
  baseline; `torchvision` or `detectron2`/`mmdetection` if we go to a direct instance model.
- **Metric:** our **own PQ implementation**, already verified identical to the organisers' code
  (0.34327 = 0.34327, §10b). We never burn a Kaggle submission just to measure PQ.
- **Data:** the `prepared/` folder. Each sample gives the model: the preprocessed image, a semantic mask,
  an instance map, and a boundary mask, via the `FilamentPatchDataset` already written (§12).
- **Compute:** local RTX 4060 for development; **Kaggle notebooks / Google Colab** for longer runs and
  the final training (the brief suggests these).
- **Tracking:** log every run's config + PQ to a CSV (or Weights & Biases) so comparisons are reproducible.

---

## 3. Candidate architectures (what we will compare)

| Tier | Model | Why try it | Cost |
|---|---|---|---|
| Bare CNN | **FCN-32s / FCN-8s** (ResNet34 backbone, no decoder / two skip score maps) | The simplest deep segmentation model; tests whether a decoder is needed. **Tested (full training): FCN-32s 0.107, FCN-8s 0.268 vs U-Net 0.307** | Low |
| Pyramid pooling | **PSPNet** (ResNet34) | Standard family for global context; targets large regions | Low–med |
| Baseline | **U-Net** with a pretrained encoder (ResNet34 / EfficientNet-B3) | Standard for thin fine structures; skip connections keep barb detail (filaments are ~11 px wide, §5); fast | Low |
| Baseline+ | **U-Net++ / FPN**, same encoders | Better multi-scale fusion; filament area spans ~10²–10⁴ px (§5), so multi-scale helps | Low–med |
| Different family | **DeepLabV3+** (ResNet34) | Not a U-Net: dilated convolutions for wide context, light decoder. Lets the experiments genuinely reject the U-Net design | Low–med |
| Instance head | U-Net semantic + **boundary head** (we already export boundary targets) | Separates the 1.4% of filaments within 5 px of a neighbour (§5) without a heavy detector | Low–med |
| Stretch | **Mask R-CNN / Mask2Former** (direct instances) | Predicts instances directly; may win on fragmentation/merging | High |

Note: U-Net **is** a CNN. The real axis is "semantic + post-processing" vs "direct instance model", and
how fancy the encoder is. We start simple and only add complexity that experiments show is worth it (the
brief rewards going beyond out-of-the-box, but only when justified).

**Pretrained vs from scratch:** start from ImageNet-pretrained encoders (faster, stronger on a small
707-image set), and note that this is allowed — the no-external-data rule is about **test labels**, not
model weights.

---

## 4. Loss function

The data prep found a **278:1 background-to-filament imbalance** (§5). Plain pixel-wise BCE would collapse
to "predict background everywhere", so:

- **Primary:** a region-overlap loss that ignores the imbalance — **Dice** or **Tversky** (Tversky lets us
  weight false negatives higher, useful since missing a filament costs in PQ).
- **Combined:** **BCE/Focal + Dice** is the common, robust choice; Focal down-weights easy background.
- **If we use a boundary head:** add a boundary loss (BCE/Dice on the boundary target).
- Compute the loss **only inside the solar disk** (off-disk is trivially background, §9).

We will try BCE+Dice first, then Tversky/Focal, and let the validation PQ decide.

---

## 5. Training setup

- **Split:** use the **`fold` column** already in `prepared/`. Default = train on folds 1–4, validate on
  fold 0. It's grouped by solar rotation and stratified by site, with leakage asserted (§8). For final
  numbers, train all 5 folds for **mean ± std** (the brief and Results rubric ask for this).
- **Sampling / augmentation:** filament-centred crops at **`p_fg ≈ 0.7`** (empty crops 27%→7%, §10), with
  the physically-justified augmentations from §10 (rotation, flips, scale, brightness/contrast/blur/noise;
  **no elastic**). Note: a flip swaps Left/Right chirality — only matters if we ever train chirality.
- **Multi-annotator targets:** pick **one random annotation set per image per epoch** (§6 / decision D2);
  validate against **all** sets, pooled, the way Kaggle scores. (Soft-average targets are exported as a
  fallback to experiment with.)
- **Optimiser:** AdamW, LR halved when validation PQ stalls for 2 epochs; mixed precision (AMP).
- **Early stopping on validation PQ** (not on loss): validate every epoch on 40 monitor images, keep the
  best checkpoint, stop after 5 epochs without a ≥0.002 gain, cap 30 epochs. No fixed epoch count.
  **Measured:** the chosen model's best epoch was 6–15 depending on the fold; each epoch = 800 crops.

---

## 6. Inference pipeline (how we turn a model into a submission)

1. Preprocess each test image with the **same** transform used in training (stored per image).
2. Run the **sliding window** over the 2048 canonical image (overlapping 512 tiles), average overlaps.
3. Threshold to a filament mask; **split into instances** (connected components; watershed/boundary cue
   for touching ones).
4. Drop tiny specks below a minimum area (tuned on fold 0) to cut false positives (each FP costs ½ in PQ).
5. Map each instance back to the original 2048×2048 grid with the **inverse transform**.
6. Encode each instance to RLE with **pycocotools** and write `submission.csv` (one row per filament) —
   using the verified encoder from §10b.

---

## 7. Hyperparameters we will tune (on fold 0, by local PQ)

Mask threshold, minimum instance area, `p_fg`, learning rate, loss weights (Dice vs BCE, Tversky β),
encoder choice, crop size, and sliding-window overlap. The brief explicitly requires relevant
hyperparameters to be tuned and the choices substantiated.

---

## 8. How we evaluate and compare

- **Primary:** PQ locally (fold 0, and 5-fold mean ± std), cross-checked occasionally on the Kaggle
  leaderboard — they should track each other since there's no train/test shift (§3).
- **Diagnostics** (from the organisers' self-eval code we already ported): the IoU and Dice distributions,
  and the one-to-many / many-to-one counts — these tell us whether errors are **fragmentation**,
  **merging**, or **misses**, which directly guides the Enhancement milestone.
- **Report** (Results rubric): best and worst predicted masks, error types, and PQ with variability over
  runs. **Also report model size and train/inference time** (brief requirement) for the compression work.

---

## 9. Milestone-2 experiment plan (concrete order)

1. **Wire up the data** → confirm `FilamentPatchDataset` feeds a model (already loads one batch, §12).
2. **Baseline:** U-Net (ResNet34) + BCE+Dice, `p_fg 0.7`, fold 0. Get a first local PQ and a first Kaggle
   submission to anchor everything.
3. **Loss sweep:** BCE+Dice vs Tversky vs Focal+Dice.
4. **Encoder/architecture sweep:** ResNet34 vs EfficientNet-B3; U-Net vs U-Net++/FPN vs **DeepLabV3+**
   (a non-U-Net family, so the sweep can genuinely reject U-Net).
5. **Post-processing:** connected components vs + boundary head vs watershed; tune min-area + threshold.
6. **Decide** whether a direct instance model (Mask R-CNN / Mask2Former) beats the best semantic+post
   pipeline enough to justify its cost → hand off to the Enhancement milestone.
7. Lock the best config, run **5 folds for mean ± std**, and record size + timing.

---

## 10. Open decisions for the group
1. Baseline encoder: ResNet34 (fast) or EfficientNet-B3 (stronger)?
2. How far do we push the "direct instance model" route given the efficiency score?
3. Where do we run final training — Kaggle, Colab, or the local 4060?
4. Do we use chirality as an auxiliary signal, or ignore it (the task is class-agnostic)?
