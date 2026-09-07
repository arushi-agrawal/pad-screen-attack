# Screen-attack detection for biometric PAD

A small prototype that reads face images and outputs, for each one, a score for whether it was photographed off a
screen (a replay attack) rather than being an original capture. Score is 0 to 1, higher means attack.

The full reasoning, findings, experiments, and limitations are in the technical report:
[Google Doc](https://docs.google.com/document/d/1VHegu-n7RD2m79tWr012_-2Zo6AhxwYwC3eFpK4w5k4/edit), also in this
repository as `docs/PAD_technical_report.docx`. This file covers setup, how to run, assumptions, and dependencies.

## Quick start

```bash
git clone <this repo> && cd pad-screen-attack
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-infer.txt --extra-index-url https://download.pytorch.org/whl/cpu
python infer.py --input /path/to/images
```

`outputs.csv` appears in the current folder with two columns, `file` and `score`. Or with Docker:

```bash
docker build -t pad .
docker run --rm -v /path/to/images:/data:ro -v "$PWD":/out pad --input /data --output /out/outputs.csv
```

The image is CPU only, about 1.6 GB, most of it PyTorch. On macOS with colima, mounted paths must be under your home
folder; anything under `/tmp` mounts as empty.

## What `infer.py` does

- Reads a folder recursively (or a single file) and scores every file Pillow can open, whatever the extension:
  PNG, JPEG, HEIC/HEIF (what iPhones save by default), BMP, TIFF, WebP, GIF.
- Grayscale, palette, and RGBA images are converted to RGB. EXIF rotation is applied. Metadata is never read.
- Writes a CSV with `file` (path relative to the input folder) and `score`. Non-image files are skipped and listed
  on the console; an image that cannot be scored keeps its row with a blank score and the reason is printed.
- Prints how many files were found, a progress bar, the median time per image, any image scored without a
  detected face (the image centre is used and the score is less reliable), and the first rows of the CSV.
- Does **not** apply a threshold. The operating point is the deployer's decision. Three reference thresholds,
  read off held-out development data, are in `artefacts/config.json` and printed at the end:

| Threshold | Real users flagged | Meaning |
|---|---|---|
| 0.59 | about 1 in 100 | strict on friction; starts missing far-range attacks |
| **0.43** | about 1 in 20 | **recommended**: passed no attack on any held-out set |
| 0.37 | about 1 in 10 | looser, no accuracy gain on this data |

Options: `--output` (default `outputs.csv`), `--artefacts` (default `artefacts/`), `--no-recursive`, `--threads`.

Speed: 0.35 s per image at 1 megapixel to 1.1 s at 24 megapixels on a laptop CPU with four threads. The network
is a fixed 0.25 s; the rest is file decoding.

## How it works, in one paragraph

Find the face (OpenCV Haar cascade, image centre as fallback), cut a square around it at the image's own resolution,
take 36 patches of 128 x 128 pixels on a fixed grid, score each patch with a small network (MobileNetV3-small,
ImageNet weights, first seven blocks frozen, single output), and average the 36 scores. Native resolution matters:
the screen's pixel grid, moire, and banding live at the camera's pixel scale and are erased by resizing the face to
224 or 512 pixels. Training augments every patch with capture variation (colour, focus, noise, compression) and a
random shrink of up to two thirds, so the model cannot rely on attacks simply having more pixels than real images.
Section 3 of the report explains each choice.

## Assumptions

- The attack type is screen replay only. Prints, masks, and other instruments are out of scope; the score on them
  is undefined, not low.
- "Bona fide" in the provided data means an original digital file from a public face set, not a live capture. A
  real user's phone selfie is the case that matters most and the one this data cannot test. This is the largest
  stated risk (report, section 6).
- A missed attack is worse than a wrongly flagged real user. The recommended threshold caps friction and then
  minimises attacks passed, never the reverse.
- Each image contains one frontal face large enough to detect. Otherwise the centre is scored and the image is
  named on the console as scored without a face.
- Latency is not a constraint for the prototype; it is measured and reported, not optimised.
- Images arrive at whatever resolution the camera produced. Do not downscale them before scoring: it saves no
  time and removes the screen texture the model detects.

## Dependencies

Python 3.9 or later, CPU only.

- `requirements-infer.txt`: what `infer.py` needs. numpy, pandas, pillow, pillow-heif, opencv-python-headless,
  tqdm, torch, torchvision. Install with the PyTorch CPU index as shown above, or pip pulls a much larger build.
- `requirements.txt`: the full development environment, adding scikit-learn, matplotlib, and Jupyter for the
  scripts and notebooks.

Every version is pinned to the one the results were produced with.

## Repository layout

```
infer.py                  folder of images in, CSV of scores out
artefacts/                model.pt (shipped weights), model_dev.pt (validated weights), config.json
pad/                      library: data.py (pipeline), metrics.py (APCER, BPCER, EER, AUC, bootstrap), model.py
scripts/                  one file per step: make_splits, baselines, train, anomaly, error_analysis, tradeoffs, plot
notebooks/                01_data_analysis (what the data taught us), 02_experiments (every model, galleries, misses)
derived/                  splits, face boxes, classical features (regenerable)
results/                  score files, result JSONs, figures, the four final-recipe fold models
docs/                     technical report, working record, interview Q&A
run_experiments.sh        reproduces everything from splits to the shipped model, by stage
```

Two model files ship. `model_dev.pt` was trained on the 820 development images and is the model every number in
the report belongs to, including the locked test set. `model.pt` is the same recipe trained on all 1023 images and
is what `infer.py` loads; it has not been separately validated. `config.json` records which is which.

## Reproducing the experiments

Put the dataset at `data/Bonafide/` and `data/Screens/` (or set `PAD_DATA` to the folder that contains them),
install `requirements.txt`, then:

```bash
./run_experiments.sh          # everything, about four hours on a laptop CPU
./run_experiments.sh cnn      # one stage: splits | baseline | cnn | density | facerel | anomaly | analysis | freeze | ship
```

The first run builds a face cache under `cache/` (about 1 GB) so later epochs read small files. Every run writes a
result JSON and per-image scores to `results/`; the notebooks read those and never retrain. Splits are fixed by a
seed in `scripts/make_splits.py` and never re-rolled; fold 4 is the locked test set and is read only by the
`freeze` stage.

## Where it breaks

Far-range attacks, where the camera never resolved the screen texture; real users under harsh lighting or with
fabric textures near the face; live captures, which the training data does not contain; displays, rooms, and
attackers other than the one session in the data; and the face detector, which misses profiles and occlusion.
Section 6 of the report gives the evidence for each and what would catch it in deployment.
