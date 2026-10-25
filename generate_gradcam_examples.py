"""Create class-specific Grad-CAM examples for validation TP/FP/FN/TN cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from gradcam_engine import GradCAM, default_target_layer
from train_engine import load_isic_binary_records, load_xray_binary_records
from vision_engine import Modality, build_model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "experiments" / "binary_baselines" / "gradcam_examples"


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("The requested XPU is unavailable")
    return device


def run_one(modality: str, device: torch.device, limit: int) -> list[dict]:
    checkpoint_path = ROOT / "artifacts" / "weights" / f"{modality}_binary_10k_v2.pth"
    eval_path = ROOT / "artifacts" / "logs" / f"{modality}_binary_10k_v2_evaluation.json"
    predictions_path = ROOT / "artifacts" / "logs" / f"{modality}_binary_10k_v2_validation_predictions.csv"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = tuple(checkpoint["class_names"])
    dataset_root = ROOT / "data" / ("nih_chest_xray" if modality == "xray" else "isic_2019")
    records, record_names, _ = (load_xray_binary_records(dataset_root) if modality == "xray"
                                else load_isic_binary_records(dataset_root))
    if tuple(record_names) != class_names:
        raise ValueError(f"Checkpoint and record labels differ for {modality}")
    id_key = 3
    by_id = {record[id_key]: (record[0], class_names[record[1]]) for record in records}
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    positive = evaluation["metrics"]["positive_class"]
    threshold = evaluation["metrics"]["decision_threshold"]
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    categorized = {key: [] for key in ("true_positive", "false_positive", "false_negative", "true_negative")}
    for row in predictions:
        actual, predicted = row["actual_label"], row["predicted_label"]
        score = float(row["positive_probability"])
        if actual == positive and predicted == positive: key = "true_positive"
        elif actual != positive and predicted == positive: key = "false_positive"
        elif actual == positive and predicted != positive: key = "false_negative"
        else: key = "true_negative"
        categorized[key].append((row, score))
    # Choose representative hard cases near the decision boundary, while making
    # the most confident positive mistakes easy to inspect.
    chosen = []
    for case, candidates in categorized.items():
        if not candidates: continue
        if case == "false_negative": selected = max(candidates, key=lambda x: x[1])
        elif case == "false_positive": selected = max(candidates, key=lambda x: x[1])
        elif case == "true_positive": selected = max(candidates, key=lambda x: x[1])
        else: selected = min(candidates, key=lambda x: x[1])
        chosen.append((case, *selected))
    if limit > 0:
        chosen = chosen[:limit]

    modality_enum = Modality.CHEST_XRAY if modality == "xray" else Modality.SKIN_LESION
    model = build_model(modality_enum, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    output_records = []
    out_dir = OUT / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    for case, row, score in chosen:
        sample_id = row["sample_id"]
        if sample_id not in by_id: raise KeyError(f"Validation sample {sample_id} missing from dataset")
        image_path, actual_from_metadata = by_id[sample_id]
        if actual_from_metadata != row["actual_label"]:
            raise ValueError(f"Saved label mismatch for sample {sample_id}")
        predicted = row["predicted_label"]
        target_index = class_names.index(predicted)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        filename = f"{case}_{sample_id}"
        overlay_path = out_dir / f"{filename}_overlay.jpg"
        with GradCAM(model, default_target_layer(modality_enum), device=device) as cam:
            result = cam.generate(image, target_index, output_path=overlay_path)
        output_records.append({
            "modality": modality, "sample_id": sample_id, "case": case,
            "actual_label": row["actual_label"], "predicted_label": predicted,
            "positive_class": positive, "positive_probability": score,
            "decision_threshold": threshold,
            "gradcam_target_class": predicted,
            "target_layer": default_target_layer(modality_enum),
            "original_path": str(result.original_path.relative_to(ROOT)),
            "heatmap_path": str(result.heatmap_path.relative_to(ROOT)),
            "overlay_path": str(result.output_path.relative_to(ROOT)),
        })
    del model
    return output_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("xray", "skin", "both"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--limit-per-modality", type=int, default=4)
    args = parser.parse_args()
    if args.limit_per_modality < 1: parser.error("--limit-per-modality must be positive")
    device = choose_device(args.device)
    modalities = ("xray", "skin") if args.modality == "both" else (args.modality,)
    result = []
    for modality in modalities:
        result.extend(run_one(modality, device, args.limit_per_modality))
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / "gradcam_examples.json"
    out_file.write_text(json.dumps({"device": str(device), "examples": result}, indent=2), encoding="utf-8")
    print(f"device={device}; generated {len(result)} examples; manifest={out_file}")


if __name__ == "__main__":
    main()
