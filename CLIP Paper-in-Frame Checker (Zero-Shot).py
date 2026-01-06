# tamrin2_clip_only_v5_fullpage_vs_partial.py
# CLIP-only (NO OpenCV), step-by-step (Exercise 2 style):
#
# Stage 1: paper vs no_paper
# Stage 2 (if paper): full_view (entire sheet visible) vs partial_view (close-up / cropped)
#
# Final:
#   no_paper
#   full     (only if full_view clearly beats partial_view)
#   partial  (otherwise)

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Stage 1: accept "paper" only if it beats "no_paper" by this margin
PAPER_MARGIN = 0.20

# Stage 2: output FULL only if full_view beats partial_view by this margin
# (conservative: avoids false FULL for zoom/close-up shots)
FULL_VIEW_MARGIN = 0.35  # good default for your examples

# Aggregation: mean of top-k prompts per label
AGG_TOPK = 2


# ----------------------------
# Prompts
# ----------------------------
LABELS_STAGE1 = {
    "paper": [
        "a photo containing a sheet of paper",
        "a paper page is visible in the image",
        "a document page made of paper is present",
        "a sheet of paper lying on a desk or table",
        "a paper page with visible edges",
    ],
    "no_paper": [
        "a photo of a desk or tabletop with no paper",
        "a wooden table surface without any sheet of paper",
        "a table or desk surface, no document, no page",
        "background only, no paper present",
        "a workspace surface with objects but no paper",
    ],
}

# Stage 2: explicit "entire sheet visible" vs "only part of sheet visible"
# IMPORTANT: we avoid vague words like "zoomed" and instead say "only part of the page is shown"
LABELS_STAGE2 = {
    "full_view": [
        "the entire sheet of paper is visible inside the photo, with background visible around all sides",
        "a full page is shown completely in the image (not a close-up), the whole sheet fits in the frame",
        "the full outline of the paper is visible, and you can see table/desk around the paper",
        "a notebook page fully visible even if the left edge is torn or perforated, and the whole page is inside the frame",
        "a complete sheet of paper photographed on a table, fully visible with space around it",
    ],
    "partial_view": [
        "only part of the paper is shown in the photo (close-up), the whole sheet does not fit in the frame",
        "the paper extends beyond the image border, so it is not fully visible",
        "a close-up of notebook paper where the full page is not shown",
        "only a corner or portion of the sheet is visible, not the entire page",
        "the paper is cropped by the photo border, not a full-page view",
    ],
}


# ----------------------------
# CLIP loading
# ----------------------------
def load_clip() -> Tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # Use the default (slow) processor to avoid requiring torchvision
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model.eval()
    return model, processor


# ----------------------------
# Scoring helpers
# ----------------------------
def build_bank(labels: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[int]]]:
    all_prompts: List[str] = []
    label_to_indices: Dict[str, List[int]] = {}
    for label, prompts in labels.items():
        idxs = []
        for p in prompts:
            idxs.append(len(all_prompts))
            all_prompts.append(p)
        label_to_indices[label] = idxs
    return all_prompts, label_to_indices


@torch.inference_mode()
def score_all_prompts(model: CLIPModel, processor: CLIPProcessor, image: Image.Image, prompts: List[str]) -> torch.Tensor:
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(DEVICE)
    return model(**inputs).logits_per_image[0].float()


def topk_mean(values: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, values.numel())
    return torch.topk(values, k=k).values.mean()


@torch.inference_mode()
def run_stage(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    labels: Dict[str, List[str]],
    agg_topk: int = 2,
) -> Tuple[str, Dict[str, float], float]:
    prompts, label_to_indices = build_bank(labels)
    logits = score_all_prompts(model, processor, image, prompts)

    scores: Dict[str, float] = {}
    for label, idxs in label_to_indices.items():
        scores[label] = float(topk_mean(logits[idxs], agg_topk).cpu().item())

    best = max(scores, key=scores.get)
    vals = sorted(scores.values(), reverse=True)
    gap = (vals[0] - vals[1]) if len(vals) >= 2 else 0.0
    return best, scores, float(gap)


def margin_ok(pos_label: str, neg_label: str, scores: Dict[str, float], margin: float) -> bool:
    return scores.get(pos_label, -1e9) >= scores.get(neg_label, -1e9) + margin


# ----------------------------
# Prediction
# ----------------------------
@torch.inference_mode()
def predict(model: CLIPModel, processor: CLIPProcessor, image_path: str) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")

    # Stage 1
    s1_pred, s1_scores, s1_gap = run_stage(model, processor, image, LABELS_STAGE1, agg_topk=AGG_TOPK)
    paper_conf = margin_ok("paper", "no_paper", s1_scores, PAPER_MARGIN)

    if (s1_pred != "paper") or (not paper_conf):
        return {
            "file": os.path.basename(image_path),
            "pred": "no_paper",
            "stage1": {"pred": s1_pred, "scores": s1_scores, "gap": s1_gap, "paper_confident": paper_conf},
        }

    # Stage 2
    s2_pred, s2_scores, s2_gap = run_stage(model, processor, image, LABELS_STAGE2, agg_topk=AGG_TOPK)

    full_score = s2_scores["full_view"]
    partial_score = s2_scores["partial_view"]
    delta_full_minus_partial = full_score - partial_score

    # Conservative rule: FULL only if clearly supported
    final = "full" if delta_full_minus_partial >= FULL_VIEW_MARGIN else "partial"

    return {
        "file": os.path.basename(image_path),
        "pred": final,
        "stage1": {"pred": s1_pred, "scores": s1_scores, "gap": s1_gap, "paper_confident": paper_conf},
        "stage2": {
            "pred": s2_pred,
            "scores": s2_scores,
            "gap": s2_gap,
            "delta_full_minus_partial": float(delta_full_minus_partial),
        },
        "params": {"PAPER_MARGIN": PAPER_MARGIN, "FULL_VIEW_MARGIN": FULL_VIEW_MARGIN, "AGG_TOPK": AGG_TOPK},
    }


# ----------------------------
# Outputs
# ----------------------------
def save_csv(results: List[Dict[str, Any]], out_csv: str):
    header = [
        "file", "pred",
        "s1_paper", "s1_no_paper", "paper_confident",
        "s2_full_view", "s2_partial_view", "delta_full_minus_partial",
        "PAPER_MARGIN", "FULL_VIEW_MARGIN", "AGG_TOPK"
    ]
    lines = [",".join(header)]

    for r in results:
        s1 = r.get("stage1", {})
        s2 = r.get("stage2", {})
        s1_sc = s1.get("scores", {})
        s2_sc = s2.get("scores", {})
        params = r.get("params", {})

        row = [
            r.get("file", ""),
            r.get("pred", ""),
            str(s1_sc.get("paper", "")),
            str(s1_sc.get("no_paper", "")),
            str(s1.get("paper_confident", "")),
            str(s2_sc.get("full_view", "")),
            str(s2_sc.get("partial_view", "")),
            str(s2.get("delta_full_minus_partial", "")),
            str(params.get("PAPER_MARGIN", "")),
            str(params.get("FULL_VIEW_MARGIN", "")),
            str(params.get("AGG_TOPK", "")),
        ]
        lines.append(",".join(row))

    Path(out_csv).write_text("\n".join(lines), encoding="utf-8")


def run_folder(images_dir: str, out_json: str = "ex_predictions.json", out_csv: str = "ex_predictions.csv"):
    model, processor = load_clip()

    images_dir = Path(images_dir)
    candidates = [p for p in sorted(images_dir.glob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    print("DIR:", str(images_dir))
    print("IMAGE CANDIDATES:", len(candidates))
    print("FILES:", [p.name for p in candidates])
    print(f"PAPER_MARGIN={PAPER_MARGIN} | FULL_VIEW_MARGIN={FULL_VIEW_MARGIN} | AGG_TOPK={AGG_TOPK}")

    results: List[Dict[str, Any]] = []
    for i, p in enumerate(candidates, start=1):
        r = predict(model, processor, str(p))
        results.append(r)

        print(f"\n[{i}/{len(candidates)}] {p.name} -> {r['pred']}")
        if r["pred"] == "no_paper":
            print(f"  stage1 scores={r['stage1']['scores']} | paper_conf={r['stage1']['paper_confident']}")
        else:
            s2 = r["stage2"]
            sc = s2["scores"]
            print(
                f"  stage2 scores={sc} | delta(full-partial)={s2['delta_full_minus_partial']:.3f} | threshold={FULL_VIEW_MARGIN}"
            )

    Path(out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    save_csv(results, out_csv)

    print(f"\nSaved JSON: {Path(out_json).resolve()}")
    print(f"Saved CSV : {Path(out_csv).resolve()}")
    print("TOTAL RESULTS:", len(results))


if __name__ == "__main__":
    run_folder(
        images_dir=r"C:\Users\EDKO_KOUROSH\Desktop\TJMASTER\.venv\kouroshzamani",
        out_json="ex_predictions.json",
        out_csv="ex_predictions.csv",
    )
