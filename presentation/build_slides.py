"""Builds COS711_A2_Presentation.pptx from the course template. Run from this folder: python build_slides.py
Every number on the slides comes from the saved results in ../modelling/runs and ../data_prep.
The team slide (slide 2) is left exactly as in the template."""
import copy, re, shutil
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

HERE = Path(__file__).parent
TEMPLATE = Path.home() / "Downloads" / "COS711_A2_Presentation_Template.pptx"
OUT = HERE / "COS711_A2_Presentation.pptx"
FIG = HERE / "figures"
BLUE, RED, GREY = RGBColor(0x00, 0x5B, 0xAA), RGBColor(0xD7, 0x1C, 0x33), RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xE8, 0xF0, 0xF8)

prs = Presentation(TEMPLATE)
S = list(prs.slides)


# ------------------------------------------------------------------ helpers
def runs_from(text):
    """'**bold** rest' -> [(text, bold)]"""
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    return [(p[2:-2], True) if p.startswith("**") else (p, False) for p in parts if p]


BU_TAGS = ("a:buClrTx", "a:buClr", "a:buSzTx", "a:buSzPct", "a:buSzPts", "a:buFontTx", "a:buFont",
           "a:buNone", "a:buAutoNum", "a:buChar", "a:buBlip")


def _set_bullet(pPr, char, lvl):
    """Explicit bullet (or none) with a consistent hanging indent, children in schema order."""
    for tag in BU_TAGS:
        for e in pPr.findall(qn(tag)): pPr.remove(e)
    after = next((pPr.find(qn(t)) for t in ("a:tabLst", "a:defRPr", "a:extLst") if pPr.find(qn(t)) is not None), None)
    new = []
    if char:
        clr = etree.Element(qn("a:buClr")); c = etree.SubElement(clr, qn("a:srgbClr")); c.set("val", "D71C33")
        sz = etree.Element(qn("a:buSzPct")); sz.set("val", "100000")
        fnt = etree.Element(qn("a:buFont")); fnt.set("typeface", "Arial")
        ch = etree.Element(qn("a:buChar")); ch.set("char", char)
        new = [clr, sz, fnt, ch]
        pPr.set("marL", str(int(Inches(0.25 + 0.3 * lvl)))); pPr.set("indent", str(int(-Inches(0.22))))
    else:
        new = [etree.Element(qn("a:buNone"))]
        pPr.set("marL", str(int(Inches(0.3 * lvl)))); pPr.set("indent", "0")
    for e in new:
        if after is not None: after.addprevious(e)
        else: pPr.append(e)


def fill(tf, items, size=14, color=None, bullets=True, space=4):
    """items: list of str or (str, level); a str starting with '#' is a heading (bold, no bullet).
    bullets=True: red bullets; False: plain paragraphs."""
    tf.word_wrap = True
    tf.clear()
    first = True
    for it in items:
        text, lvl = (it, 0) if isinstance(it, str) else it
        head = text.startswith("#")
        if head:
            text = f"**{text.lstrip('#').strip()}**"
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        p.level = 0
        for t, b in runs_from(text):
            r = p.add_run(); r.text = t; r.font.size = Pt(size + (1 if head else 0)); r.font.bold = b
            if color is not None:
                r.font.color.rgb = color
        p.space_after = Pt(space); p.space_before = Pt(6 if head and p is not tf.paragraphs[0] else 0)
        _set_bullet(p._p.get_or_add_pPr(), ("•" if lvl == 0 else "–") if bullets and not head else None, lvl)
    return tf


def box(slide, x, y, w, h, items, size=13, color=BLUE, bullets=True, space=4):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = Inches(0.02)
    fill(tf, items, size, color, bullets, space)
    return tb


def place(ph, x, y, w, h):
    ph.left, ph.top, ph.width, ph.height = Inches(x), Inches(y), Inches(w), Inches(h)


def pic(slide, name, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(FIG / name), Inches(x), Inches(y),
                                    Inches(w) if w else None, Inches(h) if h else None)


def table(slide, x, y, w, rows, col_w, size=11, row_h=0.3):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = t.cell(i, j); cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame; tf.word_wrap = True; tf.clear()
            p = tf.paragraphs[0]
            for txt, b in runs_from(str(val)):
                r = p.add_run(); r.text = txt; r.font.size = Pt(size); r.font.bold = b or i == 0
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else RGBColor(0x1A, 0x1A, 0x1A)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if i == 0 else (LIGHT if i % 2 else RGBColor(0xFF, 0xFF, 0xFF))
    return t


def placeholder_box(slide, x, y, w, h, text):
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor(0xF4, 0xF4, 0xF4)
    s.line.color.rgb = RED; s.line.width = Pt(1.5); s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    tf = s.text_frame; tf.word_wrap = True
    fill(tf, text.split("\n"), 12, GREY, bullets=False, space=2)
    for p in tf.paragraphs: p.alignment = 2
    return s


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def title_of(slide, text):
    slide.shapes.title.text_frame.text = text


def new_slide_after(index, layout=0):
    """Add an OBJECT-layout slide and move it to sit right after slide number `index` (1-based)."""
    s = prs.slides.add_slide(prs.slide_layouts[layout])
    lst = prs.slides._sldIdLst; el = lst[-1]; lst.remove(el); lst.insert(index, el)
    return s


# ------------------------------------------------------------------ figures
FIG.mkdir(exist_ok=True)
src = {"fold_timeline.png": "../data_prep/fold_timeline.png",
       "augmentation_example.png": "../modelling/figures/augmentation_example.png",
       "early_stopping_example.png": "../modelling/figures/early_stopping_example.png",
       "architecture_sweep.png": "../modelling/_slidefigs/exp_17_0.png",
       "approach_comparison.png": "../modelling/_slidefigs/seg_8_0.png",
       "error_kinds.png": "../modelling/_slidefigs/seg_14_0.png",
       "val_best_worst.png": "../modelling/_slidefigs/seg_17_0.png",
       "test_predictions.png": "../modelling/_slidefigs/seg_20_0.png"}
for dst, s in src.items():
    if (HERE / s).exists():
        shutil.copy(HERE / s, FIG / dst)

# ================================================================== 1. title
for sh in S[0].shapes:
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                r.text = r.text.replace("Assignment 3", "Assignment 2")
notes(S[0], "Title slide. Add the team names and student numbers in the placeholder line.")

# ================================================================== 2. team: untouched

# ================================================================== 3. dataset preparation
s = S[2]
body = [ph for ph in s.placeholders if ph.placeholder_format.idx == 1][0]
for ph in list(s.placeholders):
    if ph.name == "Text Placeholder 2":
        ph._element.getparent().remove(ph._element)
place(body, 0.92, 1.85, 6.0, 4.05)
fill(body.text_frame, [
    "**Data:** 707 labelled GONG H-alpha images (2048×2048, 6 telescopes, 2011–2022), 1,154 annotation sets, 8,199 filaments; 180 test images",
    "**Preprocessing:** centre every Sun and scale it to radius 905 px; intensity scaled to [0, 1] from the disk's 1st–99th percentiles; sky set to 0",
    "**Resolution:** kept full size (shrinking to 512 caps even perfect labels at PQ 0.70); train on 512×512 crops, which hold 99% of filaments whole",
    "**Class imbalance:** filaments are 0.36% of pixels (278 : 1); 70% of crops centred on a filament (empty crops 27% → 7%); loss only on the disk",
    "**Noisy samples:** 66 train / 21 test images flagged (cloud, uneven light), all kept; experts agree only at PQ 0.34, so each crop uses a random annotator",
    "**Augmentation:** rotation 0–360°, flips, ±5% scale (mask moves too); brightness, contrast, blur, noise (mask unchanged); no elastic warps",
    "**Split:** 5-fold StratifiedGroupKFold, grouped by solar rotation (no leakage), stratified by telescope",
], size=12, space=3)
pic(s, "augmentation_example.png", 7.1, 2.3, w=5.25)
box(s, 7.1, 5.15, 2.35, 0.4, ["One real training crop, augmented"], size=10, color=GREY, bullets=False)
pic(s, "fold_timeline.png", 0.92, 6.0, w=8.4)
notes(s, """We have 707 labelled H-alpha images from six GONG telescopes, with 1,154 annotation sets because many images were drawn by two or three experts.
Every image is centred and scaled so the Sun has the same 905-pixel radius, and brightness is normalised to 0-1 using the disk's own percentiles.
We do NOT shrink the images: shrinking to 512 makes filaments about 3 pixels wide, and even the perfect labels then score only 0.70. Instead we train on 512 by 512 crops cut from the full-size image; 99% of filaments fit whole in one crop.
Filaments are only 0.36% of the pixels, so 70% of crops are centred on a filament, and the loss ignores the sky.
Experts disagree a lot (they agree at PQ 0.34), so each crop uses one randomly chosen annotator.
Augmentation: rotations, flips and small scale changes move the mask with the image; brightness, blur and noise only change the image.
The split is 5-fold StratifiedGroupKFold: all images from one solar rotation stay in one fold, so the same filament never appears in training and validation, and every fold has the same mix of telescopes. The timeline at the bottom shows the folds as blocks of consecutive weeks.""")

# ================================================================== 4. experimental setup
s = S[3]
body = [ph for ph in s.placeholders if ph.placeholder_format.idx == 1][0]
place(body, 0.92, 1.85, 6.35, 5.5)
fill(body.text_frame, [
    "**Tools:** Python 3.13, PyTorch 2.10 (CUDA, mixed precision), segmentation-models-pytorch 0.6, OpenCV, pycocotools, scikit-learn, pandas, Jupyter; one NVIDIA RTX 4060 laptop GPU (8 GB)",
    "**Formulation:** semantic segmentation (filament / not) → each 8-connected blob = one filament. Touching filaments are rare (0.2%): a perfect mask split this way scores PQ 0.997",
    "**Models and losses:** 9 designs (FCN-32s, FCN-8s, U-Net ×3 encoders, U-Net++, FPN, DeepLabV3+, PSPNet) and 4 losses, all trained the same way",
    "**Training:** 512×512 crops, 800 per epoch, batch 8; AdamW (lr 3×10⁻⁴, weight decay 10⁻⁴); LR halved after 2 epochs without gain; early stopping: patience 5, min gain 0.002, max 30 epochs",
    "**Tuning:** one factor at a time on fold 0: loss → architecture → 2nd seed for the top 2 → learning rate → post-processing grid (threshold × minimum blob size)",
    "**Evaluation:** PQ = SQ × RQ (match if IoU > 0.5), against every annotator, in the original 2048² frame; equals the organisers' code (0.322735 = 0.322735). Choose on 40 monitor images, quote 99 report images, confirm with 5-fold CV",
], size=12, space=4)
pic(s, "early_stopping_example.png", 7.4, 2.4, w=5.45)
box(s, 7.4, 4.9, 5.0, 0.8, ["Early stopping, chosen model: best at epoch 12, stopped at 17. Best epochs over the 5 folds: 12, 6, 15, 9, 13, so no fixed epoch count fits all."],
    size=10, color=GREY, bullets=False)
notes(s, """Tools: PyTorch with segmentation-models-pytorch for the models, OpenCV and pycocotools for masks, scikit-learn for the split, all on one RTX 4060 laptop GPU.
We framed it as semantic segmentation: the model marks each pixel as filament or not, and each connected blob becomes one filament. That is enough because touching filaments are rare; a perfect mask split into blobs already scores 0.997.
We compared nine designs and four losses under identical training. Training uses 512 crops, 800 per epoch, AdamW at 3e-4, and the learning rate is halved when the score stalls.
We don't fix the number of epochs: after every epoch we score 40 monitor images, keep the best model, and stop after 5 epochs without improvement. The graph shows the chosen model peaking at epoch 12. The best epoch differed in every fold, from 6 to 15.
We tuned one thing at a time: loss, then architecture, then a second seed, then learning rate, then post-processing.
Evaluation is Panoptic Quality, scored exactly like Kaggle; our code gives the organisers' number to six decimals. Choices are made on 40 monitor images and the score we quote comes from 99 report images that were never used for a choice.""")

# ================================================================== 5. (new) choosing the model
s = new_slide_after(4)
title_of(s, "Experimental Setup: Choosing the Model")
pic(s, "architecture_sweep.png", 0.92, 2.25, w=10.3)
body = [ph for ph in s.placeholders if ph.placeholder_format.idx == 1][0]
place(body, 0.92, 5.45, 8.45, 1.95)
fill(body.text_frame, [
    "**Loss** (U-Net ResNet34, monitor PQ): BCE+Dice 0.307 · Focal+Dice 0.286 · Dice 0.265 · Tversky 0.258",
    "**Second seed:** U-Net EfficientNet-B3 0.312 → 0.310 (stable) · DeepLabV3+ 0.294 → 0.233 (unstable)",
    "**Learning rate** (EfficientNet-B3): 3×10⁻⁴ 0.312 · 1×10⁻⁴ 0.295 · 1×10⁻³ 0.251",
    "**Chosen: U-Net + EfficientNet-B3, BCE+Dice, lr 3×10⁻⁴.** Tied with U-Net ResNet34 (gap < 0.01), but 13.2M vs 24.4M parameters",
], size=12, space=3)
notes(s, """Every model was trained on folds 1 to 4 and scored on fold 0, with default post-processing. Blue bars are the monitor images we chose with, orange the report images we quote.
Most designs use the same ResNet34 encoder, so differences come from the decoder.
The bare CNN, FCN-32s, is worst at 0.11: it has no decoder, and its 32-pixel grid is too coarse for 11-pixel filaments. U-Nets are best because skip connections pass fine detail to the decoder. More complex decoders, U-Net++ and FPN, didn't help, and pretraining helps: the same U-Net from scratch is 0.07 lower.
For the loss, BCE plus Dice won: BCE gives a steady signal on every pixel, Dice keeps focus on the tiny filament area.
DeepLabV3+ looked good but collapsed with a second seed, so its first score was luck.
The two pretrained U-Nets are tied within noise. We chose EfficientNet-B3 because it was highest on the monitor images and has 46% fewer parameters, which matters for efficiency. Its second seed gave the same score.""")

# ================================================================== 6. results
s = S[4]
body = [ph for ph in s.placeholders if ph.placeholder_format.idx == 1][0]
body._element.getparent().remove(body._element)
table(s, 0.92, 1.95, 6.5, [
    ["Estimate", "PQ", "Images"],
    ["**Kaggle public leaderboard** (rank 789)", "**0.33**", "~90 test"],
    ["Local, 5 folds, tuned + flip TTA", "**0.368 ± 0.019**", "507"],
    ["Local, fold 0 report (SQ 0.673 × RQ 0.588)", "0.396", "99"],
    ["Local, 5 folds, default settings", "0.298 ± 0.011", "507"],
    ["Two experts scored against each other", "0.343", "93 sets"],
], col_w=[4.0, 1.5, 1.0], size=11, row_h=0.36)
placeholder_box(s, 7.75, 2.45, 4.6, 1.95, "Paste the Kaggle leaderboard screenshot here\n(team Research021: public PQ 0.33, rank 789)")
pic(s, "approach_comparison.png", 0.92, 4.25, w=8.35)
box(s, 9.45, 4.5, 2.9, 1.0, ["Fold 0 was the easiest fold, so 0.396 was optimistic. Kaggle's 0.33 is about 2 std below the fair 5-fold mean."],
    size=10, color=GREY, bullets=False)
notes(s, """Our Kaggle public score is 0.33, rank 789.
Locally, the fair estimate is 0.368 plus or minus 0.019: each of the five fold models scored on its own held-out images, with the tuned post-processing and flip test-time augmentation. Fold 0 alone gives 0.396, but fold 0 was where every choice was made and it turned out to be the easiest fold.
PQ is SQ times RQ. Our outlines are decent, SQ 0.67; the weaker part is recognition, RQ 0.59: finding the right filaments without inventing extras.
For reference, two human experts agree with each other at only 0.34, so the labels themselves are noisy.
The chart on the left compares every approach on the same images, and on the right shows what each step added for the final model: the tuned threshold and size filter add 0.07, flip TTA another 0.015.""")

# ================================================================== 7. (new) kinds of errors
s = new_slide_after(6)
title_of(s, "Results: Kinds of Errors")
pic(s, "error_kinds.png", 0.92, 2.3, w=10.6)
body = [ph for ph in s.placeholders if ph.placeholder_format.idx == 1][0]
place(body, 0.92, 4.75, 8.45, 2.65)
fill(body.text_frame, [
    "On the 99 report images: **732 of 1,136 real filaments found (64%)**; 620 false filaments, 404 missed",
    "**Missed filaments:** mostly ones the experts disagree on: found 83% of the time when 2+ experts drew it, 34% when only one did",
    "**Noise (false filaments):** 30% of our \"false\" filaments were drawn by another annotator; small predictions are least reliable (43% real at 500–1,000 px)",
    "**Fine-scale structure:** filaments under 500 px found only 25% of the time; thin barbs often cut off",
    "**Fragmentation:** 35% of filaments over 5,000 px broken into pieces (69 filaments, 6% overall). **Merging:** rare, 15 cases",
], size=12, space=3)
notes(s, """Where does the score go? On the 99 report images we find 64% of the real filaments, and about half of our predictions are real.
Most misses are filaments the experts themselves disagree on: if two or more experts drew a filament we find it 83% of the time, if only one did, 34%.
Likewise 30% of our so-called false filaments were drawn by a different annotator, so part of the error is label noise, not the model.
The real weaknesses: small filaments, found only a quarter of the time, and the thin barbs. Long filaments break into pieces: 35% of the big ones are fragmented, which counts as a miss plus several false positives.
Merging neighbouring filaments is rare, so instance separation is not the main problem.
Position on the disk barely matters, and every telescope scores between 0.35 and 0.44.""")

# ================================================================== 8. example predictions
s = S[5]
body = [ph for ph in s.placeholders if ph.placeholder_format.idx == 1][0]
pic(s, "test_predictions.png", 0.92, 2.1, w=8.45)
pic(s, "val_best_worst.png", 0.92, 4.95, w=8.45)
place(body, 9.5, 2.4, 2.95, 3.05)
fill(body.text_frame, [
    "**Viable:** every prediction sits on a dark, elongated feature; 8.3 per image vs 7.5 drawn by experts",
    "**Best (PQ 0.77):** few, clear filaments",
    "**Worst (PQ 0.04):** a long winding filament found only in pieces; faint ones missed",
    "**Why:** low contrast, gaps along long thin filaments, expert disagreement",
], size=11, space=4)
notes(s, """Top: test images, one per telescope, with the model's predictions in red, exactly as written to the submission file, plus a zoom on the largest filament. Every prediction sits on a visible dark, elongated feature, and the outlines follow the edges.
Bottom: validation images, where we have labels, so best and worst are measured. Green is the expert, red is the model.
The best image scores 0.77: a few clear filaments. The median is 0.41. The worst, Mauna Loa, scores 0.04: the long winding filament is found only as short pieces, and no piece covers half of it, so it counts as missed, and every piece counts as false.
So the predictions are viable; they fail on faint filaments, on long thin ones that break into pieces, and where the experts themselves disagree.""")

# ================================================================== 9. enhancement
s = S[6]
body = [ph for ph in s.placeholders if ph.placeholder_format.idx == 1][0]
body._element.getparent().remove(body._element)
table(s, 0.92, 1.95, 8.45, [
    ["Idea", "Effect on report PQ", "Cost", "Kept?"],
    ["Size filter + threshold (500 px, 0.6)", "+0.072 (0.309 → 0.381); 5-fold +0.055", "none", "**Yes**"],
    ["Flip test-time augmentation", "+0.015; 5-fold +0.016", "3.4× inference", "**Yes**"],
    ["Hysteresis / closing (join pieces)", "no gain (joined neighbours)", "none", "No"],
    ["Keep confident small blobs", "+0.0007 on monitor", "none", "No"],
    ["clDice loss (keep filaments intact)", "fragments 161 → 109 untuned; tuned 0.394, within seed spread", "~2× epochs", "All-data model"],
    ["Ensemble of 3 EfficientNet-B3 runs", "0.399 vs 0.386 average single model", "3× inference", "**Yes**"],
    ["Retrain on all 707 images", "fixed-epoch recipe −0.013 to −0.020", "none", "Kaggle decides"],
], col_w=[2.75, 3.25, 1.2, 1.25], size=10, row_h=0.42)
box(s, 9.5, 2.4, 2.95, 3.05, [
    "**Model:** 13.2M parameters (53 MB)",
    "**Training:** 21 min",
    "**Inference:** 0.55 s per image; 1.8 s with TTA; 9.2 s for the 5-model ensemble",
    "**Compression:** chose an encoder 46% smaller at equal PQ; quantisation and pruning not tested yet",
], size=11, space=4)
box(s, 0.92, 5.45, 8.45, 1.0, [
    "Biggest single gain: removing small false blobs, not a new model. Ensembling removes the ±0.01 luck of a single training run.",
], size=12, bullets=False)
notes(s, """Beyond the standard pipeline we tried seven ideas and measured each on the report images.
The biggest gain came from post-processing: dropping blobs under 500 pixels and raising the threshold to 0.6 added 0.07, because most tiny predictions were false. Flip test-time augmentation added 0.015 but makes inference 3.4 times slower.
To keep filaments intact we tried hysteresis and closing, which joined neighbours instead, and the clDice loss, which compares centrelines. clDice cut fragmented filaments from 161 to 109 before tuning, but after tuning it was no better than changing the seed.
Keeping only confident small blobs barely helped, so it was dropped.
Averaging three models gave 0.399 against 0.386 for the average single model, at three times the inference cost.
Efficiency: 13.2 million parameters, 21 minutes to train, about half a second per image. Choosing EfficientNet-B3 made the model 46% smaller at the same accuracy; we have not yet tried quantisation or pruning.""")

# ================================================================== 10. conclusions
s = S[7]
body = [ph for ph in s.placeholders if ph.placeholder_format.idx == 1][0]
place(body, 0.44, 1.75, 7.0, 5.6)
fill(body.text_frame, [
    "# Main take-aways",
    "Keep full resolution and use a pretrained U-Net: skip connections keep thin filaments (bare CNN 0.12 vs U-Net 0.31)",
    "Post-processing gave the largest gain (+0.07): removing small false blobs",
    "Fair local estimate 0.368 ± 0.019, Kaggle 0.33; experts agree at 0.34, so label noise limits everyone",
    "# Biggest roadblocks",
    "Noisy labels: experts disagree on faint filaments",
    "Small, faint and long broken filaments",
    "Hidden pipeline bugs (e.g. the sky predicted as one giant filament), found only by looking at masks",
    "One laptop GPU: ~20 min per run limited the number of runs and seeds",
    "# What we would do differently",
    "Spread model choices over all 5 folds, not fold 0 only",
    "Look at predicted masks from day one",
    "Test crop size, more seeds and compression earlier",
], size=14, space=3)
notes(s, """Three take-aways. Full resolution and a pretrained U-Net matter most for thin filaments. The largest single gain came from simple post-processing. And our fair local estimate, 0.368, is close to Kaggle's 0.33 and to the 0.34 at which two experts agree, so label noise caps everyone.
Roadblocks: noisy labels, small and faint filaments, long filaments that break, and bugs that no metric showed us, such as the sky being predicted as one giant filament, which we found only by looking at the masks. One laptop GPU also limited how many runs we could do.
If we did it again, we would make choices across all five folds instead of fold 0 only, inspect predictions from the start, and test crop size, more seeds and compression earlier.""")

# ================================================================== 11. bibliography
s = S[8]
body = [ph for ph in s.placeholders if ph.placeholder_format.idx == 1][0]
place(body, 0.44, 1.65, 7.05, 5.75)
fill(body.text_frame, [
    "Ahmadzadeh, A. et al. (2024). A dataset of manually annotated filaments from H-alpha observations (MAGFiLO). Scientific Data, 11, 1031.",
    "Kirillov, A. et al. (2019). Panoptic Segmentation. CVPR.",
    "Ronneberger, O., Fischer, P. & Brox, T. (2015). U-Net. MICCAI.",
    "Long, J., Shelhamer, E. & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.",
    "He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.",
    "Tan, M. & Le, Q. (2019). EfficientNet. ICML.",
    "Zhou, Z. et al. (2018). UNet++. DLMIA.",
    "Lin, T.-Y. et al. (2017). Feature Pyramid Networks. CVPR.",
    "Chen, L.-C. et al. (2018). DeepLabV3+ (Encoder-Decoder with Atrous Separable Convolution). ECCV.",
    "Zhao, H. et al. (2017). Pyramid Scene Parsing Network. CVPR.",
    "Shit, S. et al. (2021). clDice: a topology-preserving loss for tubular structures. CVPR.",
    "Iakubovskii, P. segmentation_models_pytorch. github.com/qubvel-org/segmentation_models.pytorch",
    "Kaggle (2026). Solar Filament Segmentation Challenge. kaggle.com/competitions/filament-segmentation-2026",
], size=10, bullets=False, space=5)

prs.save(OUT)
print("saved", OUT, "|", len(prs.slides), "slides")
