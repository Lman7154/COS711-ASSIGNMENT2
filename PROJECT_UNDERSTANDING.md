# COS711 Assignment 2: Understanding the Project

A plain-language summary of what we are doing and why, for the group and for the presentation.
Sources: the assignment brief (COS711-A2), the official Kaggle pages (Description, Evaluation,
Leaderboard Ranking, Submission File), and the MAGFiLO paper (Ahmadzadeh et al., *Scientific Data* 11:1031, 2024).

---

## 1. What is a solar filament?

A **solar filament** is a dense "cloud" of cooler solar material held up above the Sun's surface
by magnetic field lines, sitting along magnetic **neutral lines** (where the magnetic field flips direction).

In H-alpha images (the light we use here) a filament looks like a **dark, thin, thread-like ribbon**
lying on the bright solar disk. Key shape features:
- a **spine**: the long main axis of the filament.
- **barbs**: small thread-like branches coming off the sides of the spine. Their direction tells
  scientists about the filament's magnetic **chirality** (left- or right-handed field).

Note on terminology: the *same* structure is called a **filament** when seen as a dark feature
on the disk, and a **prominence** when seen glowing at the edge (limb) of the Sun against dark space.

---

## 1b. What is an "H-alpha image"? (and where does H-alpha come from)

**H-alpha (Hα)** is one specific colour (wavelength) of red light, at **656.3 nm**. It is emitted by
hydrogen atoms when an electron drops from the 3rd to the 2nd energy level (the first line of hydrogen's
Balmer series; "H" = hydrogen, "alpha" = first line). It is a fixed, known shade of red.

An **H-alpha image** is a photo of the Sun taken through a very narrow filter that blocks all light
**except** that one Hα wavelength. So it is not an ordinary colour photo of the Sun; it is the Sun seen
in only that single slice of red light.

**Why that wavelength?** Filaments are relatively cool, dense hydrogen gas floating above the surface.
At the Hα wavelength this gas **absorbs** the light coming from the brighter surface behind it, so a
filament appears as a **dark thread** on the bright disk. In ordinary white light, filaments are
essentially invisible. The Hα filter is what makes them stand out, which is exactly why it is used for
studying them.

**So "the light we use here"** means the **Hα (red) part of sunlight**, the narrow band the filter lets
through, not the full white sunlight your eye sees.

**Where did we get that we should use H-alpha?** We did **not** choose it; it is a given of the problem:
- The COS711 brief says the data come from the **MAGFiLO** dataset.
- The Kaggle **Data** page states the images are **H-alpha observations from GONG**, converted from FITS
  to JPEG, 2048x2048, grayscale.
- The MAGFiLO paper's title is literally *Manually Annotated GONG Filaments from **H-Alpha** Observations*.
- The data confirm it: every file name carries a GONG site code, and the annotation JSON lists GONG
  H-alpha image URLs.

**Where the images come from (GONG).** The images are captured by **GONG** (the Global Oscillation
Network Group), a network of **six telescopes** around the world:
Big Bear (USA), Cerro Tololo (Chile), Learmonth (Australia), Mauna Loa (USA), El Teide (Spain),
Udaipur (India). Spreading them around the globe means at least one site always has the Sun in daylight,
so the Sun is watched around the clock. These are the six site codes (B, C, L, M, T, U) in our file names.

---

## 2. What are filaments associated with? (why they matter)

Filaments matter because they sit at the **core of solar eruptions**. When a filament becomes
unstable and erupts, it can drive the three main space-weather events:

1. **Coronal Mass Ejections (CMEs)** - huge clouds of plasma and magnetic field thrown out from the Sun.
2. **Solar flares** - sudden, intense bursts of radiation.
3. **Solar Energetic Particle (SEP) storms** - streams of high-energy particles.

An Earth-directed eruption can:
- damage the **electric power grid**,
- disrupt **GPS** and satellite systems,
- create **radiation hazards** for aircraft crews on polar routes,
- be **lethal to astronauts** outside Earth's protective magnetosphere.

This is why filaments are a **critical event type in space-weather forecasting**: watching them is
part of predicting dangerous solar activity before it reaches Earth.

---

## 3. The scientific goal (the "why")

Automatically and precisely outline **every individual filament** on the Sun from telescope images.

Good per-filament outlines let solar physicists measure each filament's **size, shape, length,
orientation and fine structure (barbs)**. Doing this automatically across huge image archives is
what makes large-scale filament study and space-weather forecasting practical. Today this outlining
is still done largely by hand, which does not scale.

---

## 4. The assignment goal (the "what we are marked on")

Build a **deep-learning model** that:
- takes a single H-alpha image of the Sun as input, and
- outputs **one segmentation mask per individual filament** (instance segmentation),
- scored by the **Panoptic Quality (PQ)** metric on the Kaggle leaderboard.

Important scope note: the assignment stops at producing good per-filament masks on **single images**.
Tracking a filament's growth or evolution over time is the downstream *science use* of these masks,
but it is **not** part of this task. We are not asked to link filaments across time.

The images and expert labels come from the **MAGFiLO** dataset, hosted on the Kaggle
"Solar Filament Segmentation Challenge 2026". We must use only the Kaggle data (no external labels
for the test images).

### What we must actually deliver
- All of our **code** (Python; Jupyter notebook or scripts).
- The final **`submission.csv`** uploaded to Kaggle.
- A **video presentation** (<= 10 min, <= 50 MB) with every member on camera.
- The **PowerPoint slides** used in the video.

### How it is marked (100 marks)
| Category | Marks |
|---|---|
| Format (presentation requirements) | 5 |
| **Data Preparation** | 10 |
| Experimental Setup & Methodology | 20 |
| Segmentation Results | 30 |
| Segmentation Enhancement (instance separation, post-processing, compression, etc.) | 30 |
| Conclusions | 5 |
| Not including code + submission file | **-50%** of total |
| Bonus: top 100 on the Kaggle leaderboard | up to +5 |

The biggest marks are in **Results (30)** and **Enhancement (30)**. "Enhancement" is where we go beyond
a basic pipeline: separating filaments cleanly, keeping them in one piece, capturing barbs, and
cutting the compute cost (pruning / distillation / compression).

---

## 5. Why segmentation, not classification?

| Approach | What it tells you | Why it is not enough here |
|---|---|---|
| **Image classification** | "Is there a filament in this image?" (yes/no) | No location, no shape, no count. Useless for measuring filaments. |
| **Object detection** (bounding boxes) | A rectangle around each filament | A filament is a thin, curved, branching thread. A box is mostly background and loses the shape entirely. |
| **Segmentation** (pixel masks) | The exact pixels of each filament | Captures real **morphology**: area, length, orientation, barbs. This is what the science needs. |

So the reason for segmentation is not only "classification can't tell us the size". It is that only a
**pixel-level mask** captures the shape of something this thin and irregular, and shape is exactly
what matters for a filament (its length, its barbs, whether it is growing or becoming unstable).

---

## 6. Why Panoptic Quality (PQ), not a plain segmentation score like Dice?

This switch is the heart of the task. The competition **originally used the Dice score, then changed
to PQ** (the Evaluation page shows the old Dice text struck out). Understanding why is key.

**Plain segmentation metrics (Dice / IoU) are *semantic*.** They ask: "what fraction of filament
pixels did you get right?" over the whole image, treating all filaments as one single "filament vs
background" blob. A model can score well on Dice while being **scientifically wrong**, because Dice
does not care about *individual* filaments. Two failures it misses:

- **Over-merging:** two separate filaments predicted as one. The pixels are still "filament", so Dice
  is barely affected, but you have now **miscounted** and merged two distinct structures into one.
- **Fragmentation ("islands"):** one real filament broken into several pieces. Dice is mostly fine,
  but you have invented several fake filaments from one real one.

**PQ is an *instance* (panoptic) metric.** It matches each predicted filament to each true filament
one-to-one, and a match only counts if their overlap (IoU) is greater than 0.5. Then:

```
          sum of IoU over matched (true, predicted) pairs
PQ  =  ----------------------------------------------------
                  TP  +  0.5 * FP  +  0.5 * FN
```

- **Numerator** rewards tracing each individual filament's shape accurately (segmentation quality).
- **Denominator** punishes every **extra** filament invented (FP) and every **real** one missed (FN)
  (recognition / counting quality).

So in words: **PQ = "did you find each filament as a separate object?" x "did you trace its shape well?"**

That is exactly why the brief and Kaggle keep stressing **structural continuity** (no fragments, no
"islands") and penalise **one-to-many / many-to-one** matches. PQ directly measures the things that
make a result scientifically useful: correct **count**, correct **separation**, and correct **shape**.
Plain Dice ignores count and separation entirely.

### Evidence from our own data prep
- A **perfect** filament/background mask, split into blobs, scores **PQ 0.997**. So when the mask is
  right, instances are easy. The danger is *imperfect* masks that merge or fragment.
- Independent **human annotators agree only at PQ ~0.34**. Instance-level agreement is hard even for
  experts, which tells us a PQ around 0.34-0.35 is already strong (the organisers themselves say any
  PQ above 0.35 is valuable).

---

## 7. The three named challenges (what makes this hard)

The competition explicitly names three difficulties, and our data prep confirmed all three:

1. **Fine-scale structures (barbs).** Filaments are thin (median width ~11 px) with small side
   branches. Downsampling destroys them, which is why we keep native resolution.
2. **Background noise and image quality.** Ground-based telescopes, so clouds, haze, blur and uneven
   lighting appear in both train and test images.
3. **Structural continuity.** Keeping each filament as **one connected object**, not fragments or
   merged "islands". This is the error PQ punishes most.

---

## 7a. Key terms and code choices explained

This section explains the technical pieces behind our data-prep code, in plain language, with the
exact Kaggle wording that forces each choice.

### "PQ" = Panoptic Quality
Throughout these docs, **PQ means Panoptic Quality**, the instance-level metric Kaggle scores us with
(explained in §6). When we say "our PQ matches Kaggle's", we mean our local Panoptic Quality calculation
gives the same number as the organisers' scoring code.

### What is a polygon, and what is RLE?
A filament mask can be stored two ways:

- **Polygon** — a list of corner points tracing the *outline* of the filament, like connect-the-dots:
  `[x0, y0, x1, y1, ..., xn, yn]`, where the last point joins back to the first (a closed loop). This
  is compact and is how the **labels are stored** in the JSON.
  > Kaggle Data page: *"A segmentation in `annotation["segmentation"]` is a list of floats representing a
  > closed path. Each path forms a polygon capturing a filament's shape. A polygon made with n points is
  > represented as `[x_0, y_0, ... , x_(n-1), y_(n-1)]` where `x_0 = x_(n-1)` and `y_0 = y_(n-1)`."*

- **RLE (Run-Length Encoding)** — a compressed way to store the actual *filled pixel mask*. Instead of
  saving a value for all 2048x2048 = ~4.2 million pixels, RLE stores **runs**: "off for 500 pixels, on
  for 30, off for 1200, ...". Because a mask is mostly long stretches of the same value, this is tiny.
  This is the format Kaggle's **`submission.csv`** requires.
  > Kaggle Data page: *"instead of binary masks, we store segmentations as polygons in RLE format. The
  > conversion (RLE <-> Binary Masks) are lossless."*
  > Kaggle Submission File page: *"Column `segmentation_rle` contains RLE counts each encoding the mask
  > corresponding to one filament ... The Size is fixed for all images: 2048 X 2048 pixels."*

So the pipeline is: **polygon (label) -> filled pixel mask -> RLE (submission)**. We need a tool that
converts between all three forms *exactly* the way Kaggle does.

### Why `from pycocotools import mask as mask_util`
`pycocotools` is the official Python toolkit for the **COCO data format** (the standard format for
object-mask datasets). Our labels are in COCO format, and Kaggle explicitly points us to this package:
> Kaggle Data page: *"The annotations are structured into the COCO-style data format. That means, you can
> take advantage of many common operations using the `pycocotools` python package."*
> Kaggle Submission File page: *"To convert your output ... to the expected format (RLE Counts), please
> use the `pycocotools` Python package."*

We import its `mask` module (named `mask_util` in our code) to convert polygon <-> pixel mask <-> RLE.

**Why it must be this library and not our own code.** Kaggle builds its ground-truth masks with
pycocotools and scores by decoding pycocotools RLE. We first drew masks with OpenCV's `fillPoly`, and on
1,500 random filaments those masks matched pycocotools at only **IoU 0.878 (worst 0.589)**. For filaments
that are only ~11 px wide, a 1-pixel edge disagreement is a large fraction of the object, so our local
scores would have been wrong. After switching to pycocotools:
- our masks match the official area field **exactly** (max difference 0.0), and
- our local PQ equals the organisers' scoring code **exactly: 0.34327 vs 0.34327** (notebook §10b, which
  ports the organisers' self-evaluation code and asserts the two agree within 0.005).

That is what the code comment `# official GT format` refers to: this import is what keeps our local
scoring honest, so we can trust a local PQ without spending a Kaggle submission.

### Why the `SITES` dictionary and the site codes
Every file name encodes **when** and **where** the image was taken: `YYYYMMDDHHMMSSII.jpeg`. The letter
`II` just before `.jpeg` (e.g. the `B` in `20140609195854Bh.jpeg`) is the **GONG telescope site**.
> Kaggle Data page: *"Each observation's capture time and instrument are encoded in its file name as
> follows: `YYYYMMDDHHMMSSII`. For example, `20260901165702Bh.jpeg` corresponds to ... the Big Bear (Bh)
> Solar Observatory."*

`SITES` is a small lookup table turning that one-letter code into a readable name:

| Code | Telescope |
|---|---|
| B | Big Bear (USA) |
| C | Cerro Tololo (Chile) |
| L | Learmonth (Australia) |
| M | Mauna Loa (USA) |
| T | El Teide (Spain) |
| U | Udaipur (India) |

**Why we need it.** The six telescopes see slightly different conditions (weather, optics), so site is a
real variable in the data. We use it twice: (1) we analyse image quality **per site** to check the test
set isn't different from training, and (2) we **stratify the validation split by site**, so every fold
contains all six instruments in the same proportion as the test set. To do either, the code must know each
image's site, and the dictionary makes plots and tables readable ("Big Bear (USA)" instead of "B").
`print("DATA_ROOT:", ...)` simply shows which folder the data was loaded from, so we can confirm the
paths resolved correctly.

### A note on `cv2` / "No module named 'cv2'"
`cv2` is **OpenCV**, the image library used to read and process the images. If a cell fails with
`ModuleNotFoundError: No module named 'cv2'`, the packages are not installed in the Python that the
notebook is currently using (a common "wrong interpreter / wrong kernel" situation; it is not a bug in
the code). Fix: select the Python that has the packages as the notebook kernel, or run this at the top of
the notebook to install into the exact interpreter in use:
```python
import sys
!{sys.executable} -m pip install opencv-python-headless numpy pandas matplotlib seaborn tqdm scikit-learn pycocotools nbformat
```
On Kaggle most of these are pre-installed; you may still need `pip install pycocotools`.

---

## 7b. Rules and constraints we must not break

These come straight from the brief (§3.1 Notes, §4 Marking) and the Kaggle Rules page. Breaking the
starred ones is the difference between a mark and a disqualification / misconduct case.

| # | Rule | Source | Are we compliant? |
|---|---|---|---|
| 1 | **Team of 2-3 students.** Single-student submissions are disqualified. | Brief §3.1 | Group task - **confirm 2-3 members**. |
| 2 | ***No external labels for the test images.*** MAGFiLO is published elsewhere *with* test labels; downloading/using that is **academic misconduct**. Use only the Kaggle Data page. | Brief §3.1 | **Yes - verified below.** |
| 3 | **Python only**, submitted as a Jupyter notebook or scripts. | Brief §3.1 | Yes - `data_prep.ipynb`. |
| 4 | **Accept and abide by the Kaggle competition rules.** (incl. max 5 submissions/day, 2 final submissions, one Kaggle account per person, no private code sharing outside the team.) | Brief §3.1 / Kaggle Rules | Each member must accept the rules on Kaggle before submitting. |
| 5 | **Implement PQ locally** so you don't waste Kaggle submissions validating. | Brief §3.1 | **Yes** - our local PQ matches the organisers' code exactly (0.34327 vs 0.34327). |
| 6 | **Validation split:** annotation sets of the same observation must never be split across train/val; observations close in time or from the same instrument are not independent. | Brief §3.1 | **Yes** - split grouped by solar rotation, stratified by site; leakage asserted. |
| 7 | **Report model size and train/inference time** (end-to-end); experiment with compression. | Brief §3.1 | To be done in the modelling/enhancement milestones. |
| 8 | **Submit code + `submission.csv`** in the zip, or lose **50%** of the total mark. | Brief §4 | To be done before upload. |
| 9 | **Video:** <= 10 min, <= 50 MB, every member on camera (head-and-shoulders, upper-right), each member's role stated. | Brief §4 | To be done in the presentation milestone. |
| 10 | Augmentation/cleaning should be **justified by the data and experiments**; tune hyperparameters. | Brief §3.1 / §4 | In progress - every data-prep choice is evidence-backed (`DATA_PREP_DECISIONS.md`). |

### Compliance check on our data prep (rule 2 - the misconduct one)

We ran an audit of the pipeline and its outputs. Results:
- **0 of the 180 test images have a ground-truth label** anywhere in our prepared data.
- **0 instance masks or boundary masks** reference a test image. Masks exist only for the 707 training
  images (1,154 annotation sets).
- `train_gt_rle.csv` contains labels for the **707 training images only** (8,199 filaments); none for test.
- The notebook reads data **only** from the Kaggle folder (`.../MAGFiLO_1.0_Kaggle_2026/`). There are
  **no downloads, URLs, or external files** of any kind in the code.
- The **test images are used only as unlabelled inputs** (reading their pixels) for two legitimate
  purposes: measuring image quality/geometry, and producing the preprocessed PNGs the model will run on.
  No test label is ever read, because we don't have any and must not obtain any.

**Conclusion: the data-preparation stage breaks none of the rules.** In particular, we never touch
test-image labels, which is the one that would count as academic misconduct.

### A note on the MAGFiLO paper
We read the MAGFiLO paper for **context and methodology** (how the data was annotated, how anomalies
were handled, how PQ is defined). That is allowed and good practice. What is **forbidden** is using it,
or any full version of MAGFiLO, as a **source of test-image masks**. We have not done that and must not.

---

## 8. One-paragraph summary (for the slide)

Solar filaments are dark, thread-like clouds of plasma on the Sun, held up by magnetic fields. They
matter because they drive the three big space-weather events, CMEs, solar flares and SEP storms, which
can damage power grids, GPS, aircraft and astronauts. The goal of this assignment is to build a deep-
learning model that produces a precise **mask for each individual filament** in an H-alpha image
(instance segmentation), using the MAGFiLO dataset on Kaggle. We use **segmentation** rather than
classification because only pixel masks capture a filament's true shape (length, orientation, barbs),
and we are scored with **Panoptic Quality** rather than plain Dice because PQ rewards getting each
filament's shape right **and** punishes miscounting, merging and fragmenting, which is what makes the
result scientifically usable.
