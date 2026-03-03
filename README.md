# Conditional Activation Transport
This is the official repository for the paper "Conditioned Activation Transport for T2I Safety Steering".

![Comparison of steering methods in safety](images/front_comparison.jpg)

Existing methods fail to remove harmful content or alter the semantic content of images. CAT suppresses unsafe content without compromising an image's quality or semantics.

---

## Abstract
Despite their impressive capabilities, current Text-to-Image (T2I) models remain prone to generating unsafe and toxic content. While activation steering offers a promising inference-time intervention, we observe that linear activation steering frequently degrades image quality when applied to benign prompts. To address this trade-off, we first construct **SafeSteerDataset**, a contrastive dataset containing **2300 safe and unsafe prompt pairs with high cosine similarity**.

Leveraging this data, we propose **_Conditioned Activation Transport (CAT)_**, a framework that employs a geometry-based conditioning mechanism and nonlinear transport maps. By conditioning transport maps to activate only within unsafe activation regions, we minimize interference with benign queries. We validate our approach on two state-of-the-art architectures: **Z-Image** and **Infinity**. Experiments demonstrate that CAT generalizes effectively across these backbones, significantly reducing Attack Success Rate while maintaining image fidelity compared to unsteered generations.

---

## Repository structure

High-level layout:
- `experiments/`
  - `generate_images.py` – baseline image generation
  - `steering.py` – main steering script
  - `images_eval.py` – evaluate folders of generated images
  - `recompute_asr.py` – recompute ASR thresholds
  - `representation_differences.py` – representation probing (Infinity)
  - `config/` – Hydra configs (model / experiment / steering / conditioning / eval)
- `steering/` – activation collection, steering methods and CAT transport training
- `eval/` – safety classifier utilities (ShieldGemma2, CLIPScore, FID wrappers)
- `Infinity/` – modified Infinity model code and tools
- `Z-Image/` – modified Z-Image model code and tools
- `utils/` – dataset + experiment helpers

---

## Installation

### Environment

We recommend **Python 3.11** and a CUDA-enabled GPU setup.

**Option A - Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate cat
```

**Option B - pip:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Data
SafeSteerDataset is expected as Parquet and consists of contrastive safe/unsafe prompt pairs with metadata (e.g., category).

It is located in the `data` subdirectory, separated into train and test splits.

## Reproducing results
### Configuration (Hydra)

Our experiment scripts use Hydra with a top-level config at:

- `experiments/config/config.yaml`

It composes:
- `model`: which backbone + wrapper to run
- `experiment`: dataset paths + categories + batch sizes
- `steering`: method + layers + strength sweep
- `conditioning`: conditioning mechanism and kwargs

You can override any value from the CLI, e.g.:

```bash
python experiments/steering.py steering.STRENGTHS="[0.1,0.2,0.3]" conditioning.CONDITIONER_KWARGS.threshold=0.3
```

---

### Unsteered generations (baseline)
Generates images with the selected model, no steering applied:

```bash
python experiments/generate_images.py \
  model=zimage/6b_model \
  experiment=sexual \
```

This script loads the test prompts and saves `prompts_used.txt` next to the images.

### Steered generations
The main entry point is:

- `experiments/steering.py`

At a high level it:
1. loads a contrastive train set + unsafe test prompts,
2. caches activations (text and/or vision),
3. trains transport + conditioner (depending on config),
4. generates steered images for a sweep over `steering.STRENGTHS` (and optionally diffusion “scales/steps”)

A “CAT-like” configuration usually corresponds to:
- **Nonlinear transport**: `steering.METHOD=MLP_TRANSPORT`
- **Geometry-based conditioning**: `conditioning=mahalanobis` (GDA-based mask with a probability threshold)
- **Conditional application**: `steering.MODE=CONDITIONAL`

This script also enables generating images using other types of steering (ActAdd / LinearACT). Selected steering method can be specified in the Hydra config.

Example command:

```bash
python experiments/steering.py \
  model=zimage/6b_model \
  experiment=sexual \
  steering=zimage/default_train \
  conditioning=mahalanobis
```

The steering config controls the sweep and where steering is applied (layers / steps), e.g.:
`STRENGTHS`, `LAYERS_TEXT`, `LAYERS_VISION`, `STEPS_VISION`, plus transport training kwargs.


## Evaluation
Use:

- `experiments/images_eval.py <RUN_DIR>`

It can compute (depending on `experiments/config/img_eval.yaml`):
- safety classifier scores (ShieldGemma2 policies)
- **ASR** derived from policy triggers
- CLIP score on benign prompts
- FID against a reference dataset

Example:

```bash
python experiments/images_eval.py outputs/cat/2026-02-13_12-00-00 \
  --eval_name evaluation.csv \
  --per_image_name images.csv
```

## Acknowledgements

This repository includes modified code for [**Z-Image**](https://github.com/Tongyi-MAI/Z-Image) and [**Infinity**](https://github.com/FoundationVision/Infinity), and uses [**ShieldGemma2**](https://huggingface.co/google/shieldgemma-2-4b-it) for safety evaluation and [**OpenCLIP**](https://github.com/mlfoundations/open_clip) for alignment scoring.

This research was supported by the Polish National Science Centre (NCN) within grant no. 2023/51/I/ST6/02854. We gratefully acknowledge Poland's high-performance Infrastructure PLGrid for providing computer facilities and support within computational grants no. PLG/2025/018230 and PLG/2025/018391.

This work was also supported by the German Research Foundation (DFG) within the framework of the Weave Programme under the project titled "Protecting Creativity: On the Way to Safe Generative Models" with number 545047250.

# How to Cite
If you find our dataset and research useful for your work, please cite it using the following BibTeX entry:

```bibtex
@misc{chrabaszcz2026conditionedactivationtransportt2i,
      title={Conditioned Activation Transport for T2I Safety Steering},
      author={Maciej Chrabąszcz and Aleksander Szymczyk and Jan Dubiński and Tomasz Trzciński and Franziska Boenisch and Adam Dziedzic},
      year={2026},
      eprint={2603.03163},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.03163},
}
```