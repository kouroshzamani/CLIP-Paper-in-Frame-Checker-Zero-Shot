# CLIP-Paper-in-Frame-Checker-Zero-Shot-


Detect whether a paper/document is present in an image and whether the **entire page is fully visible** (vs **cropped/close-up**) using **CLIP zero-shot** with **prompt engineering + prompt ensembling**.  
✅ **No OpenCV required.**

---

## Why this project?
When capturing documents with a phone camera, images often become unusable because:
- there is **no paper** in the frame,
- paper exists but is **zoomed-in / cropped**,
- the paper is **fully visible** (good capture).

This repo provides a simple, interpretable baseline using CLIP text-image similarity.

---

## Features
- **Exercise 1 (Single-stage):** one-shot label selection from a single label list
- **Exercise 2 (Two-stage gating):**
  - Stage 1: `paper` vs `no_paper`
  - Stage 2 (if paper): `full_view` vs `partial_view`
- **Prompt engineering:** carefully written prompts for each label (including negative cases)
- **Prompt ensembling:** multiple prompts per label + `top-k mean` aggregation
- Outputs:
  - `ex_predictions.json`
  - `ex_predictions.csv`

---

## Method (high-level)
CLIP computes similarity between:
- the input image
- a set of text prompts describing each class

### Prompt Engineering
We improve accuracy by rewriting prompts to be:
- more specific (e.g., “entire sheet fits in the frame”)
- less ambiguous (avoid vague wording)
- robust to domain variations (e.g., notebook pages, torn/perforated edges)

### Prompt Ensembling
Instead of 1 prompt per class, we use multiple prompts and aggregate scores:
- `score(label) = mean(top-k prompt scores)`

This reduces sensitivity to wording.

---

## Installation
```bash
pip install torch pillow transformers
Note: If you set use_fast=True for CLIPProcessor, you may need torchvision.
This repo works fine with the default (slow) processor (no torchvision required).

Usage (Exercise 2: two-stage)
Edit the folder path in __main__ and run:

bash
Copy code
python kouroshzamani-task2.py
Outputs are saved as:

ex_predictions.json

ex_predictions.csv

Configuration
You can tune:

PAPER_MARGIN: how much paper must beat no_paper in Stage 1

FULL_VIEW_MARGIN: how much full_view must beat partial_view in Stage 2

AGG_TOPK: top-k prompts used for aggregation

Example defaults:

PAPER_MARGIN = 0.20

FULL_VIEW_MARGIN = 0.35

AGG_TOPK = 2

Output format
JSON
Each image produces:

final prediction: no_paper / full / partial

stage 1 scores and confidence

stage 2 scores and delta

CSV
Includes:

stage 1 scores (paper, no_paper)

stage 2 scores (full_view, partial_view)

delta (full_view - partial_view)
```

```bash

Example Output

IMAGE CANDIDATES: 5
FILES: ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
PAPER_MARGIN=0.2 | FULL_VIEW_MARGIN=0.35 | AGG_TOPK=2

[1/5] 1.jpg -> partial
  stage2 scores={'full_view': 23.893352508544922, 'partial_view': 23.7387638092041} | delta(full-partial)=0.155 | threshold=0.35

[2/5] 2.jpg -> partial
  stage2 scores={'full_view': 26.721874237060547, 'partial_view': 27.87808609008789} | delta(full-partial)=-1.156 | threshold=0.35

[3/5] 3.jpg -> full
  stage2 scores={'full_view': 25.496267318725586, 'partial_view': 24.737213134765625} | delta(full-partial)=0.759 | threshold=0.35

[4/5] 4.jpg -> partial
  stage2 scores={'full_view': 22.671401977539062, 'partial_view': 23.44445037841797} | delta(full-partial)=-0.773 | threshold=0.35

[5/5] 5.jpg -> no_paper
  stage1 scores={'paper': 25.49460220336914, 'no_paper': 29.81293487548828} | paper_conf=False

Saved JSON: C:\Users\EDKO_KOUROSH\Desktop\TJMASTER\.venv\ex_predictions.json
Saved CSV : C:\Users\EDKO_KOUROSH\Desktop\TJMASTER\.venv\ex_predictions.csv
TOTAL RESULTS: 5

Process finished with exit code 0
Interpretation

1.jpg, 2.jpg, 4.jpg → partial (paper present but not a full-page view)

3.jpg → full (full-view score clearly beats partial-view by the threshold)

5.jpg → no_paper (paper did not confidently beat no_paper in stage 1)

```

Notes / Limitations
This is a CLIP-only approach (global understanding). It may confuse:

white paper on white background (low contrast)

extreme lighting / reflections

textures that resemble paper

For production-grade accuracy, consider:

more dataset-specific prompts

light fine-tuning (linear probe / LoRA)

explicit geometry checks (OpenCV) if allowed

