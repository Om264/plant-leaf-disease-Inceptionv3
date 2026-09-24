# Plant Leaf Disease Classification — Inception v3

Fine-tunes ImageNet-pretrained **Inception v3** (via [`timm`](https://github.com/huggingface/pytorch-image-models))
on a custom plant-leaf image dataset, with full training, evaluation, prediction, and
explainability (Grad-CAM) tooling.

## A note on the "99% accuracy" figure

Review papers reporting ~99% accuracy for Inception v3 on leaf-disease classification are
almost always evaluated on **PlantVillage** or similarly clean, lab-photographed datasets:
single leaf per image, plain background, controlled lighting, thousands of images per class.
On those datasets several architectures — not just Inception v3 — reach 97–99%, because the
task is close to saturated, not because Inception v3 is uniquely powerful.

What actually drives accuracy, roughly in order of importance:

1. **Data leakage** — if photos of the same physical leaf, plant, or field plot end up in
   both train and test, accuracy is inflated and won't hold on new leaves. This is the single
   most common reason lab benchmarks look better than field deployments.
2. **Dataset size and class balance** — per-class image counts matter more than backbone choice.
3. **Image quality/consistency** — background clutter, lighting, occlusion, multiple leaves per
   photo all make the task harder than PlantVillage.
4. **Backbone architecture** — matters, but is usually a smaller effect than the above once you're
   comparing modern CNNs (Inception v3, EfficientNet, ResNet, MobileNetV3, ...) at similar scale.

This repository gives you a correctly-implemented, leakage-aware Inception v3 pipeline. Whether
it reaches 99% on **your** data depends on the above, and is worth reporting honestly either way.

## Why Inception v3 specifically

Two things distinguish Inception v3 from the MobileNetV3 pipeline you may have seen elsewhere:

- **299×299 input** (vs. MobileNetV3's 224×224) — more spatial detail for small lesions/spots,
  at higher compute cost.
- **Auxiliary classifier head** (`AuxLogits`), injected partway through the network. During
  training its loss is added to the main loss (`total = main_loss + 0.4 * aux_loss`), which acts
  as a regularizer and improves gradient flow into early layers. It's discarded at inference —
  the [`plantdisease/model.py`](plantdisease/model.py) module handles this automatically (see
  the docstring there for a subtlety: timm returns the aux tuple in eval mode too, unlike
  torchvision's implementation, so all inference code routes through a `forward_logits()` helper
  that always discards it).

## Repository layout

```
├── README.md
├── requirements.txt / requirements-dev.txt
├── train.py              # fine-tune + evaluate on the test set
├── evaluate.py            # evaluate an existing checkpoint on any labeled folder
├── predict.py              # predict on one image or a folder of images
├── gradcam.py               # Grad-CAM heatmap for one prediction
├── plantdisease/
│   ├── data.py            # dataset discovery, stratified split, transforms
│   ├── model.py            # Inception v3 creation, aux-logits handling, checkpoint I/O
│   ├── engine.py            # train/eval loops (aux-loss-aware)
│   ├── metrics.py            # accuracy/precision/recall/F1/kappa/MCC/ROC-AUC + reports
│   ├── plots.py             # training curves, confusion matrices
│   └── utils.py              # seeding, device
├── scripts/
│   └── split_dataset.py    # group-aware stratified split (prevents leakage)
└── tests/
    └── test_smoke.py       # end-to-end test, random-init model, no internet needed
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

GPU strongly recommended: Inception v3 at 299px is considerably heavier than MobileNetV3.
On CPU, expect training to be slow — reduce `--batch_size` and image count for a quick check.

## 1. Prepare your dataset

Two layouts are supported.

**A. Already split:**
```
data/
  train/  Healthy/*.jpg  Blight/*.jpg  Rust/*.jpg ...
  val/    Healthy/*.jpg  ...
  test/   Healthy/*.jpg  ...
```
(`val`/`validation`/`valid` and `test`/`testing` are all recognized.)

**B. Single folder** — `train.py` will create a stratified 70/15/15 split automatically:
```
data/
  Healthy/*.jpg
  Blight/*.jpg
  Rust/*.jpg
```

### Avoiding data leakage

If your images are frames/crops from the same physical leaves, plants, or field plots, a
naive random split can put near-duplicates in both train and test — this is very likely how
some papers get to 99%. Use the group-aware splitter instead of the automatic split:

```bash
python scripts/split_dataset.py --src raw_data --dst data/split \
    --group_regex "^(leaf\d+)_"     # example: files named leaf012_img03.jpg
```
This keeps all images sharing a group id in the same split. If you don't have identifiable
groups, at minimum split by *plant* or *acquisition session* rather than by individual image
whenever that information exists.

## 2. Train

```bash
python train.py --data_dir data/split --output_dir runs/exp1 --epochs 40
```

Key options (`python train.py --help` for the full list):

| Flag | Default | Notes |
|---|---|---|
| `--model` | `inception_v3` | any timm model name works; other Inception variants too |
| `--img_size` | `299` | Inception v3's native size; going much below ~224 will error out |
| `--no_aux_logits` | off | disables the auxiliary classifier loss |
| `--aux_weight` | `0.4` | weight of the aux head's loss (standard recipe value) |
| `--freeze_epochs` | `2` | epochs training only the classifier head before unfreezing |
| `--batch_size` | `32` | lower this first if you hit GPU out-of-memory at 299px |
| `--class_weights` | off | use for imbalanced disease classes |
| `--patience` | `8` | early stopping on validation macro-F1 |

Outputs in `runs/exp1/`:
- `best_model.pth` — checkpoint (weights + class names + preprocessing config)
- `test_metrics.json`, `test_classification_report.txt`, `test_per_class_metrics.csv`
- `test_confusion_matrix.png` (+ normalized version), `test_predictions.csv`
- `training_curves.png`, `training_history.csv`
- `args.json` — the exact arguments used, for reproducibility

Metrics reported: accuracy, precision/recall/F1 (macro & weighted), per-class
precision/recall/F1/specificity, Cohen's kappa, Matthews correlation coefficient, ROC-AUC
(one-vs-rest macro), plus parameter count, checkpoint size, and CPU/GPU inference latency.

## 3. Evaluate on a separate labeled set

```bash
python evaluate.py --checkpoint runs/exp1/best_model.pth --test_dir data/external_test
```
Useful for testing on field-collected images after training on a lab dataset, or vice versa.

## 4. Predict on new images

```bash
python predict.py --checkpoint runs/exp1/best_model.pth --input leaf.jpg
python predict.py --checkpoint runs/exp1/best_model.pth --input new_leaves/ --csv preds.csv
```

## 5. Explain a prediction (Grad-CAM)

```bash
python gradcam.py --checkpoint runs/exp1/best_model.pth --image leaf.jpg --output cam.png
```
Highlights which region of the leaf drove the prediction — useful for sanity-checking that
the model is actually looking at lesions rather than background or pot/label artifacts, which
is a common failure mode behind inflated benchmark numbers.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
Runs a full train → checkpoint → predict → evaluate cycle on tiny synthetic images with a
randomly initialized model (no internet access or pretrained weights needed), plus a check
of the `--no_aux_logits` path. Takes about a minute on CPU.

## Reproducibility

`--seed` (default 42) seeds Python, NumPy, and PyTorch. `args.json` in every run's output
folder records the exact configuration used.

## License

MIT — see [LICENSE](LICENSE).
