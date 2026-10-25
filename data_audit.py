"""Create a reproducible, local audit of the project's NIH and ISIC files.

Run from any working directory with ``python data_audit.py --data-dir data``.
The audit reads metadata and image headers; by default it fully decodes a
deterministic sample rather than decoding every multi-gigabyte image file.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ISIC_CLASSES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK")
PROJECT_ROOT = Path(__file__).resolve().parent


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _sample_quality(paths: list[Path], size: int, seed: int) -> dict:
    sample = random.Random(seed).sample(paths, min(size, len(paths)))
    extensions, dimensions, modes = Counter(), Counter(), Counter()
    failures = []
    for path in sample:
        extensions[path.suffix.lower()] += 1
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                image.load()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            failures.append({"path": str(path), "error": str(exc)[:300]})
    return {
        "sample_count": len(sample), "seed": seed,
        "sample_extensions": dict(extensions), "sample_dimensions": dict(dimensions),
        "sample_pil_modes": dict(modes), "sample_decode_failures": failures,
        "full_decode_performed": False,
    }


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _safe_checkpoint_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _nih_audit(data_root: Path, sample_size: int) -> tuple[dict, list[tuple[str, int]]]:
    root = data_root / "nih_chest_xray"
    image_files = _images(root)
    by_name = {p.name: p for p in image_files}
    rows = _load_csv(root / "Data_Entry_2017.csv")
    per_label, exact_labels = Counter(), Counter()
    missing = 0
    label_to_paths: dict[str, list[Path]] = {"No Finding": [], "Pneumonia": []}
    image_to_patient = {}
    for row in rows:
        filename = row.get("Image Index", "").strip()
        labels = [part.strip() for part in row.get("Finding Labels", "").split("|") if part.strip()]
        exact_labels["|".join(labels)] += 1
        per_label.update(labels)
        image_to_patient[filename] = row.get("Patient ID", "").strip()
        path = by_name.get(filename)
        if path is None:
            missing += 1
        elif len(labels) == 1 and labels[0] in label_to_paths:
            label_to_paths[labels[0]].append(path)
    train_ids = set(x.strip() for x in (root / "train_val_list.txt").read_text(encoding="utf-8-sig").splitlines() if x.strip())
    test_ids = set(x.strip() for x in (root / "test_list.txt").read_text(encoding="utf-8-sig").splitlines() if x.strip())
    model_meta = _safe_checkpoint_metadata(PROJECT_ROOT / "artifacts/weights/xray_binary_10k_v2.json")
    manifest = model_meta.get("split_manifest", {})
    train_manifest, val_manifest = set(manifest.get("train_ids", [])), set(manifest.get("val_ids", []))
    selected = train_manifest | val_manifest
    selected_patients = {image_to_patient.get(name) for name in selected}
    official_test_patients = {image_to_patient.get(name) for name in test_ids}
    patient_rows = Counter(value for value in image_to_patient.values() if value)
    split_quality = {
        "official_train_val_image_ids": len(train_ids), "official_test_image_ids": len(test_ids),
        "official_image_id_overlap": len(train_ids & test_ids),
        "baseline_train_image_ids_in_official_test": len(train_manifest & test_ids),
        "baseline_validation_image_ids_in_official_test": len(val_manifest & test_ids),
        "baseline_selected_patient_ids_overlapping_official_test": len(selected_patients & official_test_patients),
        "baseline_train_validation_image_id_overlap": len(train_manifest & val_manifest),
    }
    result = {
        "source": {"dataset": "NIH ChestX-ray14", "download_id": "nih-chest-xrays/data",
                   "local_label_file": str(root / "Data_Entry_2017.csv"),
                   "official_source": "https://nihcc.app.box.com/v/ChestXray-NIHCC"},
        "inventory": {"image_files": len(image_files), "metadata_rows": len(rows),
                      "unique_patients": len({v for v in image_to_patient.values() if v}),
                      "repeated_patient_groups": sum(value > 1 for value in patient_rows.values()),
                      "missing_image_paths": missing, "extension_counts": dict(Counter(p.suffix.lower() for p in image_files))},
        "labels": {"per_finding_positive_rows_with_overlap": dict(per_label),
                   "top_exact_label_sets": exact_labels.most_common(15),
                   "multi_finding_rows": sum(count for label, count in exact_labels.items() if "|" in label),
                   "exact_no_finding": exact_labels.get("No Finding", 0),
                   "exact_pneumonia": exact_labels.get("Pneumonia", 0)},
        "quality_sample": _sample_quality(image_files, sample_size, 42),
        "split_audit": split_quality,
        "duplicate_analysis": {"method": "not performed for full corpus; repeat patient IDs tracked separately",
                               "exact_pixel_duplicates": None, "near_duplicates": None},
    }
    return result, [(label, len(paths)) for label, paths in label_to_paths.items()]


def _isic_audit(data_root: Path, sample_size: int) -> tuple[dict, list[tuple[str, int]]]:
    root = data_root / "isic_2019"
    image_files = _images(root)
    by_stem = {p.stem: p for p in image_files}
    labels_path = root / "ISIC_2019_Training_GroundTruth.csv"
    rows = _load_csv(labels_path)
    counts, missing, ambiguous = Counter(), 0, 0
    label_paths: dict[str, list[Path]] = {label: [] for label in ISIC_CLASSES}
    ids = set()
    for row in rows:
        image_id = row["image"]
        ids.add(image_id)
        positives = [name for name in ISIC_CLASSES if row.get(name) in {"1", "1.0"}]
        if len(positives) != 1:
            ambiguous += 1
            continue
        label = positives[0]
        counts[label] += 1
        path = by_stem.get(image_id)
        if path is None:
            missing += 1
        else:
            label_paths[label].append(path)
    metadata_path = root / "ISIC_2019_Training_Metadata.csv"
    metadata = _load_csv(metadata_path) if metadata_path.is_file() else []
    lesion_ids = [row.get("lesion_id", "").strip() for row in metadata]
    repeated = Counter(x for x in lesion_ids if x)
    checkpoint = _safe_checkpoint_metadata(PROJECT_ROOT / "artifacts/weights/skin_binary_10k_v2.json")
    manifest = checkpoint.get("split_manifest", {})
    train_ids, val_ids = set(manifest.get("train_ids", [])), set(manifest.get("val_ids", []))
    lesion_by_image = {row.get("image"): row.get("lesion_id", "").strip() for row in metadata}
    train_groups = {lesion_by_image.get(x) for x in train_ids if lesion_by_image.get(x)}
    val_groups = {lesion_by_image.get(x) for x in val_ids if lesion_by_image.get(x)}
    result = {
        "source": {"dataset": "ISIC 2019 training split", "download_id": "salviohexia/isic-2019-skin-lesion-images-for-classification",
                   "official_challenge": "https://challenge.isic-archive.com/landing/2019/",
                   "official_data_and_license": "https://challenge.isic-archive.com/data/",
                   "license_note": "Official aggregate training data is listed as CC BY-NC; verify contributor attribution and terms."},
        "inventory": {"image_files": len(image_files), "ground_truth_rows": len(rows),
                      "metadata_rows": len(metadata), "unique_ground_truth_image_ids": len(ids),
                      "missing_image_paths": missing, "ambiguous_rows": ambiguous,
                      "extension_counts": dict(Counter(p.suffix.lower() for p in image_files)),
                      "lesion_id_rows_nonempty": sum(bool(x) for x in lesion_ids),
                      "unique_lesion_ids": len(repeated), "repeated_lesion_groups": sum(v > 1 for v in repeated.values()),
                      "patient_id_column_available": any("patient" in key.lower() for key in metadata[0]) if metadata else False},
        "labels": dict(counts), "quality_sample": _sample_quality(image_files, sample_size, 43),
        "split_audit": {"baseline_train_image_ids": len(train_ids), "baseline_validation_image_ids": len(val_ids),
                        "train_validation_image_id_overlap": len(train_ids & val_ids),
                        "baseline_train_validation_lesion_id_overlap": len(train_groups & val_groups)},
        "duplicate_analysis": {"method": "lesion IDs report repeated lesion images; pixel/near-duplicate scan not performed",
                               "repeated_lesion_groups_all_labels": sum(v > 1 for v in repeated.values()),
                               "exact_pixel_duplicates": None, "near_duplicates": None},
    }
    return result, [(label, counts.get(label, 0)) for label in ISIC_CLASSES if counts.get(label, 0)]


def _representative_grid(items: list[tuple[str, list[Path]]], output: Path, seed: int) -> None:
    rng = random.Random(seed)
    nrows = len(items)
    fig, axes = plt.subplots(nrows, 4, figsize=(10, 2.5 * nrows), squeeze=False)
    for row_index, (label, paths) in enumerate(items):
        chosen = rng.sample(paths, min(4, len(paths))) if paths else []
        for col_index in range(4):
            ax = axes[row_index, col_index]
            ax.axis("off")
            if col_index < len(chosen):
                with Image.open(chosen[col_index]) as image:
                    ax.imshow(ImageOps.fit(image.convert("RGB"), (260, 200)))
                ax.set_title(f"{label} | {chosen[col_index].stem[:24]}", fontsize=8)
            elif col_index == 0:
                ax.text(.5, .5, f"{label}: no local images", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_audit(data_dir: Path, output_dir: Path, sample_size: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    nih, nih_counts = _nih_audit(data_dir, sample_size)
    isic, isic_counts = _isic_audit(data_dir, sample_size)
    results = {"audit_date": "2026-10-25", "sample_size": sample_size, "seed": 42,
               "datasets": {"nih_chest_xray": nih, "isic_2019": isic}}
    (output_dir / "dataset_audit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (output_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dataset", "class", "raw_count"))
        writer.writerows(("NIH ChestX-ray14", label, count) for label, count in nih_counts)
        writer.writerows(("ISIC 2019", label, count) for label, count in isic_counts)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, data in ((axes[0], "NIH ChestX-ray14 finding rows", nih_counts),
                            (axes[1], "ISIC 2019 training diagnoses", isic_counts)):
        labels, values = zip(*data)
        ax.barh(labels, values, color="#2D6CDF" if "NIH" in title else "#0F9D91")
        ax.invert_yaxis(); ax.set_title(title); ax.set_xlabel("Positive records (multi-label counts overlap)")
        ax.grid(axis="x", alpha=.2)
        for y, value in enumerate(values): ax.text(value, y, f" {value:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=170, bbox_inches="tight")
    fig.savefig(output_dir / "class_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    nih_paths = _images(data_dir / "nih_chest_xray")
    nih_rows = _load_csv(data_dir / "nih_chest_xray" / "Data_Entry_2017.csv")
    nih_map = {p.name: p for p in nih_paths}
    nih_rep = {"No Finding": [], "Pneumonia": []}
    for row in nih_rows:
        label = row["Finding Labels"].strip()
        if label in nih_rep and row["Image Index"] in nih_map: nih_rep[label].append(nih_map[row["Image Index"]])
    isic_paths = _images(data_dir / "isic_2019")
    isic_map = {p.stem: p for p in isic_paths}
    isic_rows = _load_csv(data_dir / "isic_2019" / "ISIC_2019_Training_GroundTruth.csv")
    isic_rep = {label: [] for label in ISIC_CLASSES[:-1]}
    for row in isic_rows:
        labels = [label for label in ISIC_CLASSES[:-1] if row.get(label) in {"1", "1.0"}]
        if len(labels) == 1 and row["image"] in isic_map: isic_rep[labels[0]].append(isic_map[row["image"]])
    _representative_grid(list(nih_rep.items()), output_dir / "nih_representative_images.png", 42)
    _representative_grid(list(isic_rep.items()), output_dir / "isic_representative_images.png", 42)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "dataset_audit")
    parser.add_argument("--sample-size", type=int, default=250)
    args = parser.parse_args()
    if not args.data_dir.is_absolute():
        args.data_dir = (Path.cwd() / args.data_dir).resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = (Path.cwd() / args.output_dir).resolve()
    if args.sample_size < 1: parser.error("--sample-size must be positive")
    result = run_audit(args.data_dir, args.output_dir, args.sample_size)
    print(json.dumps({name: value["inventory"] for name, value in result["datasets"].items()}, indent=2))
    print(f"Audit files written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
