"""Training pipeline for NIH ChestX-ray14 and ISIC 2019 multi-label models.

Example: python train_engine.py --modality both --epochs 20 --folds 5 --pretrained
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from vision_engine import Modality, build_model

NIH_LABELS = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
)
ISIC_LABELS = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
ISIC_SOURCE_COLUMNS = {"MEL": "MEL", "NV": "NV", "BCC": "BCC", "AKIEC": "AK", "BKL": "BKL", "DF": "DF", "VASC": "VASC"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_transforms(size: int, training: bool, modality: str) -> transforms.Compose:
    """Use spatial augmentation for both modalities and color augmentation for skin."""
    if not training:
        return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ops: list[Callable] = [
        transforms.RandomResizedCrop(size, scale=(0.78, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(0.5),
    ]
    if modality == "skin":
        ops.append(transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.16, hue=0.025))
    ops.extend([
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.92, 1.08)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    return transforms.Compose(ops)


class NIHChestXrayDataset(Dataset):
    """NIH images and 14 independent disease targets; retains patient IDs for splitting."""
    def __init__(self, root, transform=None):
        self.root, self.transform = Path(root), transform
        labels_csv = self.root / "Data_Entry_2017.csv"
        if not labels_csv.is_file():
            raise FileNotFoundError(f"NIH labels CSV not found: {labels_csv}")
        files = {p.name: p for p in self.root.glob("images_*/images/*") if p.is_file()}
        if not files:
            files = {p.name: p for p in self.root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
        self.records = []
        with labels_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                path = files.get(row["Image Index"])
                if path is None:
                    continue
                found = set(row["Finding Labels"].split("|"))
                self.records.append((path, [float(label in found) for label in NIH_LABELS], row["Patient ID"]))
        if not self.records:
            raise RuntimeError(f"No NIH images matched {labels_csv}")
        self.targets = np.asarray([r[1] for r in self.records], dtype=np.float32)
        self.groups = np.asarray([r[2] for r in self.records])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, target, _ = self.records[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.float32)


class ISIC2019Dataset(Dataset):
    """ISIC ground-truth one-hot rows represented as the API's seven output labels."""
    def __init__(self, root, transform=None):
        self.root, self.transform = Path(root), transform
        labels_csv = self.root / "ISIC_2019_Training_GroundTruth.csv"
        if not labels_csv.is_file():
            raise FileNotFoundError(f"ISIC ground-truth CSV not found: {labels_csv}")
        files = {p.stem: p for p in self.root.glob("*/*.jpg")}
        if not files:
            files = {p.stem: p for p in self.root.rglob("*.jpg")}
        self.records = []
        self.skipped_unmapped = 0
        with labels_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                path = files.get(row["image"])
                if path is not None:
                    target = [float(row[ISIC_SOURCE_COLUMNS[label]]) for label in ISIC_LABELS]
                    # The serving model currently exposes seven labels and has
                    # no SCC output. Do not silently train SCC images as an
                    # all-negative (“healthy”) example for those seven labels.
                    if not any(target):
                        self.skipped_unmapped += 1
                        continue
                    self.records.append((path, target))
        if not self.records:
            raise RuntimeError(f"No ISIC images matched {labels_csv}")
        self.targets = np.asarray([r[1] for r in self.records], dtype=np.float32)
        self.strata = self.targets.argmax(axis=1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, target = self.records[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.float32)


class BinaryImageDataset(Dataset):
    """Filtered two-class view; image tensors are opened lazily, never cached in RAM."""
    def __init__(self, records, class_names, transform=None):
        self.records = records
        self.class_names = tuple(class_names)
        self.transform = transform
        self.targets = np.asarray([row[1] for row in records], dtype=np.int64)
        self.groups = np.asarray([row[2] for row in records], dtype=str)
        self.sample_ids = [row[3] for row in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, _group, _sample_id = self.records[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def load_xray_binary_records(root):
    """Keep only exact No Finding/Pneumonia NIH rows; discard mixed diagnoses."""
    root = Path(root)
    csv_path = root / "Data_Entry_2017.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"NIH labels CSV not found: {csv_path}")
    files = {p.name: p for p in root.glob("images_*/images/*") if p.is_file()}
    if not files:
        files = {p.name: p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    records, excluded_other, missing_image = [], 0, 0
    class_names = ("No Finding", "Pneumonia")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            finding = row["Finding Labels"].strip()
            if finding == "No Finding":
                label = 0
            elif finding == "Pneumonia":
                label = 1
            else:
                excluded_other += 1
                continue
            path = files.get(row["Image Index"])
            if path is None:
                missing_image += 1
                continue
            records.append((path, label, row["Patient ID"], row["Image Index"]))
    if not records:
        raise RuntimeError(f"No usable No Finding/Pneumonia images matched {csv_path}")
    _print_binary_filter_summary("xray", records, excluded_other, missing_image, class_names)
    return records, class_names, {"excluded_other_or_mixed": excluded_other, "missing_image": missing_image}


def load_isic_binary_records(root):
    """Keep unambiguous one-hot MEL/NV records; use duplicate lesion IDs as groups."""
    root = Path(root)
    labels_path = root / "ISIC_2019_Training_GroundTruth.csv"
    metadata_path = root / "ISIC_2019_Training_Metadata.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"ISIC ground-truth CSV not found: {labels_path}")
    files = {p.stem: p for p in root.glob("*/*.jpg")}
    if not files:
        files = {p.stem: p for p in root.rglob("*.jpg")}
    lesion_by_image = {}
    if metadata_path.is_file():
        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                lesion_by_image[row["image"]] = (row.get("lesion_id") or "").strip()
    diagnosis_columns = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK")
    records, excluded_other, excluded_ambiguous, missing_image = [], 0, 0, 0
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(diagnosis_columns).difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"ISIC ground truth is missing diagnosis columns: {sorted(missing_columns)}")
        for row in reader:
            try:
                values = {name: float(row[name]) for name in diagnosis_columns}
            except (TypeError, ValueError):
                excluded_ambiguous += 1
                continue
            positives = [name for name, value in values.items() if value > 0.5]
            if len(positives) != 1 or any(value not in (0.0, 1.0) for value in values.values()):
                excluded_ambiguous += 1
                continue
            diagnosis = positives[0]
            if diagnosis not in {"MEL", "NV"}:
                excluded_other += 1
                continue
            image_id = row["image"]
            path = files.get(image_id)
            if path is None:
                missing_image += 1
                continue
            # ISIC 2019 does not supply patient IDs. Group repeated images of
            # the same lesion when lesion_id is present; otherwise each image
            # has a unique group and the limitation is logged below.
            group = lesion_by_image.get(image_id) or f"image:{image_id}"
            records.append((path, 0 if diagnosis == "MEL" else 1, group, image_id))
    if not records:
        raise RuntimeError(f"No usable MEL/NV images matched {labels_path}")
    class_names = ("MEL", "NV")
    lesion_group_counts = defaultdict(int)
    for record in records:
        if not record[2].startswith("image:"):
            lesion_group_counts[record[2]] += 1
    repeated_lesion_groups = sum(count > 1 for count in lesion_group_counts.values())
    _print_binary_filter_summary("skin", records, excluded_other, missing_image, class_names,
                                 excluded_ambiguous=excluded_ambiguous)
    if repeated_lesion_groups:
        print(f"[skin] grouping {repeated_lesion_groups:,} repeated lesion_id groups where available; no patient ID exists in metadata", flush=True)
    else:
        print("[skin] no repeated lesion IDs and no patient ID column; using image-level stratification (patient leakage cannot be assessed)", flush=True)
    return records, class_names, {"excluded_other_diagnoses": excluded_other,
                                  "excluded_ambiguous_labels": excluded_ambiguous,
                                  "missing_image": missing_image,
                                  "patient_id_available": False,
                                  "lesion_id_grouping_available": bool(repeated_lesion_groups)}


def _print_binary_filter_summary(modality, records, excluded_other, missing_image, class_names,
                                 excluded_ambiguous=0):
    counts = np.bincount(np.asarray([r[1] for r in records], dtype=np.int64), minlength=len(class_names))
    print(f"[{modality} binary] retained={len(records):,}; excluded_other={excluded_other:,}; "
          f"excluded_ambiguous={excluded_ambiguous:,}; missing_image={missing_image:,}", flush=True)
    print("[{} binary] class counts: {}".format(
        modality, ", ".join(f"{name}={counts[i]:,}" for i, name in enumerate(class_names))), flush=True)


def stratified_group_split(dataset, seed, val_fraction=0.2):
    """Split once, before independent class caps, preserving groups when possible."""
    if not 0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be greater than 0 and less than 0.5")
    counts = np.bincount(dataset.targets, minlength=len(dataset.class_names))
    if np.any(counts < 2):
        raise ValueError(f"Need at least two images per class for a split; counts={counts.tolist()}")
    group_counts = np.asarray([
        len(np.unique(dataset.groups[dataset.targets == label])) for label in range(len(counts))
    ])
    if np.all(group_counts >= 2):
        folds = max(2, round(1.0 / val_fraction))
        folds = min(folds, min(group_counts))
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        train_indices, val_indices = next(splitter.split(np.zeros(len(dataset)), dataset.targets, dataset.groups))
    else:
        # This fallback is limited to datasets without usable group identifiers.
        train_indices, val_indices = train_test_split(
            np.arange(len(dataset)), test_size=val_fraction, random_state=seed,
            stratify=dataset.targets,
        )
    overlap = set(dataset.groups[train_indices]).intersection(dataset.groups[val_indices])
    if overlap:
        raise RuntimeError(f"Leakage detected: {len(overlap)} groups occur in train and validation")
    return np.asarray(train_indices), np.asarray(val_indices)


def select_binary_indices(dataset, modality, target_total, seed=42):
    """Select exactly target_total unique records, preserving the requested class policy."""
    if target_total < 2:
        raise ValueError("--target-total-images must be at least 2")
    if target_total > len(dataset):
        raise ValueError(
            f"Requested {target_total:,} images for {modality}, but only {len(dataset):,} eligible images exist"
        )
    rng = np.random.default_rng(seed)
    class_counts = np.bincount(dataset.targets, minlength=len(dataset.class_names))
    if np.any(class_counts == 0):
        raise ValueError(f"Both binary classes need eligible examples; counts={class_counts.tolist()}")

    if modality == "xray":
        # The project requests every available exact-label Pneumonia image,
        # then enough No Finding examples to reach the total, without replacement.
        pneumonia = np.flatnonzero(dataset.targets == 1)
        no_finding = np.flatnonzero(dataset.targets == 0)
        if len(pneumonia) >= target_total:
            raise ValueError(
                f"There are {len(pneumonia):,} eligible Pneumonia images, at least the requested total; "
                "a two-class target cannot include all of them and No Finding examples"
            )
        normal_needed = target_total - len(pneumonia)
        if len(no_finding) < normal_needed:
            raise ValueError(
                f"Cannot reach {target_total:,} images: need {normal_needed:,} No Finding examples, "
                f"but only {len(no_finding):,} are eligible"
            )
        selected = np.concatenate((pneumonia, rng.choice(no_finding, size=normal_needed, replace=False)))
    else:
        # Largest-remainder allocation preserves the observed MEL/NV ratio
        # as closely as possible while producing exactly the requested total.
        proportions = class_counts / class_counts.sum()
        raw_allocations = proportions * target_total
        allocations = np.floor(raw_allocations).astype(int)
        remaining = target_total - int(allocations.sum())
        remainder_order = np.argsort(-(raw_allocations - allocations))
        for label in remainder_order[:remaining]:
            allocations[label] += 1
        selected_parts = []
        for label, amount in enumerate(allocations):
            if amount > class_counts[label]:
                raise ValueError(f"Requested {amount} {dataset.class_names[label]} images; only {class_counts[label]} exist")
            members = np.flatnonzero(dataset.targets == label)
            selected_parts.append(rng.choice(members, size=int(amount), replace=False))
        selected = np.concatenate(selected_parts)

    rng.shuffle(selected)
    if len(np.unique(selected)) != target_total:
        raise RuntimeError("Dataset selection unexpectedly duplicated or lost image indices")
    return selected


def cap_binary_split(dataset, indices, max_per_class, seed):
    """Reproducibly cap each split separately, never moving validation samples."""
    if max_per_class <= 0:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(len(dataset.class_names)):
        members = np.asarray(indices)[dataset.targets[indices] == label]
        if len(members) == 0:
            raise ValueError(f"Split has no examples for {dataset.class_names[label]}")
        rng.shuffle(members)
        selected.extend(members[:max_per_class].tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def inverse_frequency_class_weights(targets, num_classes):
    """Set CE weights to N/(K*n_k), countering imbalance without resampling images."""
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"Cannot weight a missing class; counts={counts.tolist()}")
    return len(targets) / (num_classes * counts.astype(np.float64))


def positive_class_index(class_names):
    return 1 if tuple(class_names)[1] == "Pneumonia" else 0


def select_binary_threshold(y_true, probabilities, class_names):
    """Choose an F1-maximizing operating threshold from validation probabilities only."""
    positive_class = positive_class_index(class_names)
    positive_scores = probabilities[:, positive_class]
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], positive_scores)))
    best_key, best_threshold = (-1.0, -1.0, -1.0), 0.5
    for threshold in candidates:
        predicted_positive = positive_scores >= threshold
        actual_positive = y_true == positive_class
        true_positive = int(np.logical_and(predicted_positive, actual_positive).sum())
        false_positive = int(np.logical_and(predicted_positive, ~actual_positive).sum())
        false_negative = int(np.logical_and(~predicted_positive, actual_positive).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        # Prefer sensitivity when multiple thresholds tie on F1, then a higher
        # threshold to avoid needlessly lowering specificity.
        key = (f1, recall, float(threshold))
        if key > best_key:
            best_key, best_threshold = key, float(threshold)
    return best_threshold


def binary_metrics(y_true, probabilities, class_names, threshold=0.5):
    positive_class = positive_class_index(class_names)
    negative_class = 1 - positive_class
    predictions = np.where(probabilities[:, positive_class] >= threshold,
                           positive_class, negative_class)
    per_class = classification_report(
        y_true, predictions, labels=list(range(len(class_names))), target_names=list(class_names),
        output_dict=True, zero_division=0,
    )
    try:
        auc = float(roc_auc_score((y_true == positive_class).astype(int), probabilities[:, positive_class]))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[negative_class, positive_class]).ravel()
    specificity = float(tn / max(tn + fp, 1))
    sensitivity = float(tp / max(tp + fn, 1))
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc_positive_class": auc,
        "positive_class": class_names[positive_class],
        "decision_threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_precision": float(tp / max(tp + fp, 1)),
        "positive_f1": float(per_class[class_names[positive_class]]["f1-score"]),
        "per_class": {name: {metric: (int(per_class[name][metric]) if metric == "support"
                                        else float(per_class[name][metric]))
                              for metric in ("precision", "recall", "f1-score", "support")}
                      for name in class_names},
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(range(len(class_names)))).tolist(),
    }


def set_binary_trainable_layers(model, modality, fine_tune=False):
    """Train the head first, then only the final backbone stage if requested."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    if not fine_tune:
        return
    if modality == "xray":
        model.features.denseblock4.requires_grad_(True)
        model.features.norm5.requires_grad_(True)
    else:
        # EfficientNet's last feature stage is the highest-level visual block.
        model.features[-2:].requires_grad_(True)


def run_binary_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None,
                     scheduler=None, amp=False, deadline=None, decision_threshold=0.5,
                     select_threshold=False):
    training = optimizer is not None
    model.train(training)
    if training:
        # Frozen backbone BatchNorm layers should not keep changing running stats.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                parameter.requires_grad for parameter in module.parameters(recurse=False)
            ):
                module.eval()
    total_loss, targets_all, probabilities_all = 0.0, [], []
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for images, targets in loader:
        if deadline is not None and time.monotonic() >= deadline:
            raise TrainingBudgetExpired
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        total_loss += loss.detach().item() * len(targets)
        targets_all.append(targets.detach().cpu().numpy())
        probabilities_all.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
    y_true = np.concatenate(targets_all)
    probabilities = np.concatenate(probabilities_all)
    data_source = loader.dataset
    while not hasattr(data_source, "class_names") and hasattr(data_source, "dataset"):
        data_source = data_source.dataset
    if not hasattr(data_source, "class_names"):
        raise AttributeError("Binary DataLoader dataset must expose class_names")
    if select_threshold:
        decision_threshold = select_binary_threshold(y_true, probabilities, data_source.class_names)
    return (total_loss / len(loader.dataset),
            binary_metrics(y_true, probabilities, data_source.class_names, decision_threshold))


def _binary_dataset_and_split(args, modality):
    root = Path(args.data_dir) / ("nih_chest_xray" if modality == "xray" else "isic_2019")
    records, class_names, exclusions = (
        load_xray_binary_records(root) if modality == "xray" else load_isic_binary_records(root)
    )
    eligible = BinaryImageDataset(records, class_names)
    selected_indices = select_binary_indices(eligible, modality, args.target_total_images, args.seed)
    selected_records = [records[i] for i in selected_indices]
    selected = BinaryImageDataset(selected_records, class_names)
    selected_distribution = np.bincount(selected.targets, minlength=len(class_names))
    print(f"[{modality} binary] selected total={len(selected):,}/{len(eligible):,}; "
          + ", ".join(f"{name}={selected_distribution[i]:,}" for i, name in enumerate(class_names)), flush=True)
    train_indices, val_indices = stratified_group_split(selected, args.seed, args.val_fraction)
    train_indices = cap_binary_split(selected, train_indices, args.max_train_per_class, args.seed)
    val_indices = cap_binary_split(selected, val_indices, args.max_val_per_class, args.seed + 1)
    for split_name, split_indices in (("training", train_indices), ("validation", val_indices)):
        split_counts = np.bincount(selected.targets[split_indices], minlength=len(class_names))
        if np.any(split_counts == 0):
            raise RuntimeError(f"{split_name} split must contain both classes; counts={split_counts.tolist()}")
    train_ids = [selected.sample_ids[i] for i in train_indices]
    val_ids = [selected.sample_ids[i] for i in val_indices]
    actual_total = len(train_ids) + len(val_ids)
    if actual_total != args.target_total_images:
        print(f"[LIMITATION] requested {args.target_total_images:,} selected images, but split/caps yield "
              f"{actual_total:,}; adjust --max-train-per-class/--max-val-per-class or target.", flush=True)
    if set(train_ids).intersection(val_ids):
        raise RuntimeError("Train and validation image IDs overlap")
    return root, selected_records, class_names, exclusions, train_ids, val_ids


def _resolve_manifest_indices(dataset, wanted_ids, field):
    lookup = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    missing = [sample_id for sample_id in wanted_ids if sample_id not in lookup]
    if missing:
        raise ValueError(f"Saved {field} split references {len(missing)} images missing from the current dataset")
    return np.asarray([lookup[sample_id] for sample_id in wanted_ids], dtype=np.int64)


def _binary_checkpoint_payload(model, modality, class_names, args, metrics, epoch, manifest,
                               exclusions, optimizer=None, scheduler=None, scaler=None,
                               best_auc=float("-inf")):
    payload = {
        "task": "binary_classification",
        "experiment_mode": "binary",
        "model_state_dict": model.state_dict(),
        "architecture": "densenet121" if modality == "xray" else "efficientnet_b4",
        "modality": modality,
        "labels": list(class_names),
        "class_names": list(class_names),
        "class_to_idx": {name: index for index, name in enumerate(class_names)},
        "preprocessing": {"image_size": args.image_size, "color_mode": "RGB",
                         "resize": "square bilinear", "mean": list(MEAN), "std": list(STD)},
        "seed": args.seed,
        "settings": {"head_epochs": args.head_epochs, "fine_tune_epochs": args.fine_tune_epochs,
                     "batch_size": args.batch_size, "learning_rate": args.lr,
                     "weight_decay": args.weight_decay, "max_train_per_class": args.max_train_per_class,
                     "max_val_per_class": args.max_val_per_class, "val_fraction": args.val_fraction,
                     "image_size": args.image_size, "pretrained": args.pretrained, "workers": args.workers,
                     "patience": args.patience, "amp": not args.no_amp,
                     "target_total_images": args.target_total_images, "run_tag": args.run_tag},
        "metrics": metrics,
        "decision_threshold": metrics.get("decision_threshold", 0.5) if metrics else 0.5,
        "positive_label": metrics.get("positive_class") if metrics else None,
        "epoch": epoch,
        "best_auc": best_auc,
        "split_manifest": manifest,
        "filter_counts": exclusions,
        "validation_note": "Validation metrics and threshold are not independent test results; no external test set was evaluated.",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_binary_metadata(path, payload, selected_counts):
    json_payload = {key: value for key, value in payload.items()
                    if key not in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"}}
    json_payload["selected_counts"] = selected_counts
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2, allow_nan=True)


def train_binary_experiment(args, modality):
    root, records, class_names, exclusions, train_ids, val_ids = _binary_dataset_and_split(args, modality)
    index_data = BinaryImageDataset(records, class_names)
    resume_payload = None
    if args.resume:
        try:
            resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        except TypeError:
            resume_payload = torch.load(args.resume, map_location="cpu")
        if resume_payload.get("task") != "binary_classification" or resume_payload.get("modality") != modality:
            raise ValueError("--resume must point to a binary last checkpoint for this modality")
        saved_manifest = resume_payload.get("split_manifest")
        if saved_manifest:
            train_ids, val_ids = saved_manifest["train_ids"], saved_manifest["val_ids"]

    train_idx = _resolve_manifest_indices(index_data, train_ids, "train")
    val_idx = _resolve_manifest_indices(index_data, val_ids, "validation")
    if set(train_idx).intersection(val_idx):
        raise RuntimeError("Train and validation sample indices overlap")
    transform_modality = "xray" if modality == "xray" else "skin"
    train_data = BinaryImageDataset(records, class_names, make_transforms(args.image_size, True, transform_modality))
    val_data = BinaryImageDataset(records, class_names, make_transforms(args.image_size, False, transform_modality))
    device = resolve_device(args.device, args.cpu)
    amp = device.type in {"cuda", "xpu"} and not args.no_amp
    pin_memory = device.type in {"cuda", "xpu"}
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers,
                   "pin_memory": pin_memory, "persistent_workers": args.workers > 0,
                   "worker_init_fn": _limit_worker_threads if args.workers > 0 else None}
    train_loader = DataLoader(Subset(train_data, train_idx), shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(val_data, val_idx), shuffle=False, **loader_args)
    print(f"[{modality} binary] selected train={len(train_idx):,}, validation={len(val_idx):,}; device={device}", flush=True)
    positive_index = positive_class_index(class_names)
    print(f"[{modality} binary] output mapping: "
          + ", ".join(f"index {i}={name}" for i, name in enumerate(class_names))
          + f"; monitored positive={class_names[positive_index]} at index {positive_index}", flush=True)
    for split_name, indices in (("train", train_idx), ("validation", val_idx)):
        counts = np.bincount(index_data.targets[indices], minlength=len(class_names))
        print(f"[{modality} binary] {split_name} distribution: " +
              ", ".join(f"{name}={counts[i]:,}" for i, name in enumerate(class_names)), flush=True)

    model_modality = Modality.CHEST_XRAY if modality == "xray" else Modality.SKIN_LESION
    model = build_model(model_modality, len(class_names), pretrained=args.pretrained)
    try:
        model.to(device)
        if device.type == "xpu":
            model.eval()
            with torch.no_grad():
                model(torch.zeros((1, 3, args.image_size, args.image_size), device=device))
            sync_device(device)
    except Exception as exc:
        if device.type != "xpu":
            raise
        print(f"[runtime] Intel XPU model warm-up failed ({exc}); falling back to CPU", flush=True)
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
        device = torch.device("cpu")
        amp, pin_memory = False, False
        model.to(device)
        loader_args["pin_memory"] = False
        del train_loader, val_loader
        train_loader = DataLoader(Subset(train_data, train_idx), shuffle=True, **loader_args)
        val_loader = DataLoader(Subset(val_data, val_idx), shuffle=False, **loader_args)
        print(f"[runtime] continuing on fallback device={device}", flush=True)
    head_epochs = args.head_epochs
    total_epochs = head_epochs + args.fine_tune_epochs
    set_binary_trainable_layers(model, modality, fine_tune=False)
    backbone_params, head_params = [], list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr * 0.1}, {"params": head_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs * len(train_loader)))
    scaler = make_scaler(device, amp)
    class_weights = inverse_frequency_class_weights(index_data.targets[train_idx], len(class_names))
    class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    print(f"[{modality} binary] weighted CrossEntropy class weights: "
          + ", ".join(f"{name}={class_weights[i]:.4f}" for i, name in enumerate(class_names)), flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    output_dir = Path(args.output_dir)
    stem = f"{modality}_binary_{args.run_tag}"
    best_path, last_path = output_dir / f"{stem}.pth", output_dir / f"{stem}_last.pth"
    metadata_path = output_dir / f"{stem}.json"
    if resume_payload and Path(args.resume).resolve() != last_path.resolve():
        raise ValueError(f"Resume from the matching output directory's latest checkpoint: {last_path}")
    if not resume_payload and (best_path.exists() or last_path.exists()):
        raise FileExistsError(
            f"Binary outputs already exist for {modality}; use --resume {last_path} "
            "to continue, or choose a new --output-dir for a fresh experiment"
        )
    manifest = {"train_ids": train_ids, "val_ids": val_ids}
    best_auc, best_metrics, start_epoch, patience_counter = -math.inf, None, 0, 0
    if resume_payload:
        model.load_state_dict(resume_payload["model_state_dict"])
        if "optimizer_state_dict" not in resume_payload:
            raise ValueError("Resume checkpoint has no optimizer state; use the *_binary_last.pth checkpoint")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if resume_payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict"):
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
        saved_settings = resume_payload.get("settings", {})
        for key, current in (("image_size", args.image_size), ("batch_size", args.batch_size),
                             ("head_epochs", args.head_epochs), ("fine_tune_epochs", args.fine_tune_epochs),
                             ("max_train_per_class", args.max_train_per_class),
                             ("max_val_per_class", args.max_val_per_class), ("val_fraction", args.val_fraction),
                             ("target_total_images", args.target_total_images), ("run_tag", args.run_tag)):
            if key in saved_settings and saved_settings[key] != current:
                raise ValueError(f"Resume requires the original --{key.replace('_', '-')}={saved_settings[key]}")
        start_epoch = int(resume_payload.get("epoch", 0))
        best_auc = float(resume_payload.get("best_auc", -math.inf))
        best_metrics = resume_payload.get("best_metrics")
        patience_counter = int(resume_payload.get("patience_counter", 0))
        print(f"[{modality} binary] resuming after completed epoch {start_epoch}", flush=True)

    log_dir = output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stem}_training.jsonl"
    log_mode = "a" if resume_payload else "w"
    selected_counts = {
        split: {name: int(np.sum(index_data.targets[indices] == i)) for i, name in enumerate(class_names)}
        for split, indices in (("train", train_idx), ("validation", val_idx))
    }
    with log_path.open(log_mode, encoding="utf-8") as log:
        if not resume_payload:
            log.write(json.dumps({"event": "run_start", "modality": modality,
                                  "selected_counts": selected_counts, "filter_counts": exclusions,
                                  "seed": args.seed, "settings": {
                                      "image_size": args.image_size, "batch_size": args.batch_size,
                                      "head_epochs": args.head_epochs,
                                      "fine_tune_epochs": args.fine_tune_epochs,
                                      "max_train_per_class": args.max_train_per_class,
                                      "max_val_per_class": args.max_val_per_class,
                                  }}, allow_nan=True) + "\n")
            log.flush()
        for epoch_index in range(start_epoch, total_epochs):
            budget_expired = False
            fine_tune = epoch_index >= head_epochs
            set_binary_trainable_layers(model, modality, fine_tune=fine_tune)
            while True:
                try:
                    train_loss, train_stats = run_binary_epoch(
                        model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler,
                        scheduler=scheduler, amp=amp, deadline=args.deadline,
                    )
                    with torch.no_grad():
                        val_loss, val_stats = run_binary_epoch(
                            model, val_loader, criterion, device, deadline=args.deadline,
                            select_threshold=True,
                        )
                except TrainingBudgetExpired:
                    print(f"[{modality} binary] budget reached; resume from {last_path} to continue", flush=True)
                    budget_expired = True
                    break
                except Exception as exc:
                    if device.type != "xpu":
                        raise
                    print(f"[runtime] XPU failed during {modality} training ({exc}); moving model/optimizer to CPU", flush=True)
                    try:
                        torch.xpu.empty_cache()
                    except Exception:
                        pass
                    device = torch.device("cpu")
                    model.to(device)
                    move_optimizer_state(optimizer, device)
                    criterion.to(device)
                    scaler = make_scaler(device, amp=False)
                    amp, pin_memory = False, False
                    loader_args["pin_memory"] = False
                    del train_loader, val_loader
                    train_loader = DataLoader(Subset(train_data, train_idx), shuffle=True, **loader_args)
                    val_loader = DataLoader(Subset(val_data, val_idx), shuffle=False, **loader_args)
                    print(f"[runtime] resuming current epoch on fallback device={device}", flush=True)
                    continue
                else:
                    break
            if budget_expired:
                break
            epoch_number = epoch_index + 1
            auc = val_stats["roc_auc_positive_class"]
            improved = math.isfinite(auc) and auc > best_auc
            if improved:
                best_auc, best_metrics, patience_counter = auc, val_stats, 0
                best_payload = _binary_checkpoint_payload(
                    model, modality, class_names, args, val_stats, epoch_number, manifest, exclusions,
                    best_auc=best_auc,
                )
                best_payload["class_weights"] = class_weights.tolist()
                _atomic_torch_save(best_payload, best_path)
            else:
                patience_counter += 1
            record = {"epoch": epoch_number, "phase": "fine_tune" if fine_tune else "head",
                      "train_loss": train_loss, "validation_loss": val_loss,
                      "train_metrics": train_stats, "validation_metrics": val_stats,
                      "learning_rates": [group["lr"] for group in optimizer.param_groups]}
            print(f"[{modality} binary] epoch {epoch_number}/{total_epochs} "
                  f"phase={record['phase']} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_acc={val_stats['accuracy']:.4f} val_auc={auc:.4f} "
                  f"{val_stats['positive_class']}_sensitivity="
                  f"{val_stats['sensitivity']:.4f} specificity={val_stats['specificity']:.4f} "
                  f"threshold={val_stats['decision_threshold']:.4f}", flush=True)
            log.write(json.dumps(record, allow_nan=True) + "\n")
            log.flush()
            last_payload = _binary_checkpoint_payload(
                model, modality, class_names, args, val_stats, epoch_number, manifest, exclusions,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler, best_auc=best_auc,
            )
            last_payload["class_weights"] = class_weights.tolist()
            last_payload["best_metrics"] = best_metrics
            last_payload["patience_counter"] = patience_counter
            _atomic_torch_save(last_payload, last_path)
            _write_binary_metadata(metadata_path, last_payload, selected_counts)
            if patience_counter >= args.patience:
                print(f"[{modality} binary] early stopping after {args.patience} non-improving validation epochs", flush=True)
                break

    if best_path.is_file():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        _atomic_torch_save(best_payload, best_path)
        print(f"[{modality} binary] best validation checkpoint: {best_path}", flush=True)
        print(f"[{modality} binary] resumable checkpoint: {last_path}", flush=True)
    else:
        print(f"[{modality} binary] no best checkpoint yet; resume with {last_path} after an epoch completes", flush=True)
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    clear_device_cache(device)


def evaluate_binary_checkpoint(args, modality):
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(args.output_dir) / f"{modality}_binary_{args.run_tag}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Binary checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "binary_classification" or checkpoint.get("modality") != modality:
        raise ValueError("Evaluation checkpoint does not match binary experiment and modality")
    root = Path(args.data_dir) / ("nih_chest_xray" if modality == "xray" else "isic_2019")
    records, class_names, _ = load_xray_binary_records(root) if modality == "xray" else load_isic_binary_records(root)
    if tuple(checkpoint["class_names"]) != tuple(class_names):
        raise ValueError("Checkpoint class order does not match the binary dataset")
    dataset = BinaryImageDataset(records, class_names, make_transforms(
        checkpoint.get("preprocessing", {}).get("image_size", args.image_size), False,
        "xray" if modality == "xray" else "skin"))
    saved_ids = checkpoint.get("split_manifest", {}).get("val_ids")
    if saved_ids:
        val_idx = _resolve_manifest_indices(dataset, saved_ids, "validation")
    else:
        _, val_idx = stratified_group_split(dataset, checkpoint.get("seed", args.seed), args.val_fraction)
        val_idx = cap_binary_split(dataset, val_idx, args.max_val_per_class, checkpoint.get("seed", args.seed) + 1)
    model_modality = Modality.CHEST_XRAY if modality == "xray" else Modality.SKIN_LESION
    model = build_model(model_modality, len(class_names)).to(resolve_device(args.device, args.cpu))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = next(model.parameters()).device
    loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type in {"cuda", "xpu"},
                        persistent_workers=args.workers > 0,
                        worker_init_fn=_limit_worker_threads if args.workers > 0 else None)
    saved_threshold = float(checkpoint.get("decision_threshold", 0.5))
    _, metrics = run_binary_epoch(model, loader, nn.CrossEntropyLoss(), device,
                                  decision_threshold=saved_threshold)
    result = {"checkpoint": str(checkpoint_path), "validation_count": len(val_idx),
              "class_names": list(class_names), "metrics": metrics,
              "validation_note": "Validation metrics are not independent test results and do not establish clinical readiness."}
    result_path = Path(args.output_dir).parent / "logs" / f"{modality}_binary_{args.run_tag}_evaluation.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)
    print(f"[{modality} binary] evaluation saved: {result_path}", flush=True)


class MultiLabelFocalLoss(nn.Module):
    """Focal loss on logits: alpha_t * (1-p_t)^gamma * BCEWithLogits.

    BCEWithLogits computes -log(p_t) stably. The modulating factor (1-p_t)^gamma
    approaches zero for easy examples and stays large for hard/misclassified ones,
    focusing gradient updates on difficult and minority examples. alpha_t applies
    alpha to positives and 1-alpha to negatives for class polarity balancing.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        if not 0 <= alpha <= 1 or gamma < 0 or reduction not in {"mean", "sum", "none"}:
            raise ValueError("Invalid focal-loss alpha, gamma, or reduction")
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        targets = targets.to(dtype=logits.dtype)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability = torch.sigmoid(logits)
        p_t = probability * targets + (1 - probability) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class EarlyStopping:
    def __init__(self, patience, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = -math.inf, 0

    def update(self, score):
        improved = math.isfinite(score) and score > self.best + self.min_delta
        if improved:
            self.best, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def macro_roc_auc(targets, probabilities):
    scores = [roc_auc_score(targets[:, i], probabilities[:, i])
              for i in range(targets.shape[1]) if np.unique(targets[:, i]).size == 2]
    return float(np.mean(scores)) if scores else float("nan")


def make_folds(dataset, modality, n_splits, seed):
    """Stratify ISIC by class; split NIH at patient level with multilabel balancing."""
    if modality == "skin":
        counts = np.bincount(dataset.strata)
        if np.any(counts[counts > 0] < n_splits):
            raise ValueError("Each represented ISIC class needs at least --folds examples")
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    .split(np.zeros(len(dataset)), dataset.strata))
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError as exc:
        raise ImportError("NIH cross-validation requires iterative-stratification (listed in requirements.txt)") from exc
    # Aggregate findings per patient, split patient groups, then expand back to
    # images. This prevents the same patient's studies leaking across folds.
    patient_images = defaultdict(list)
    for idx, patient in enumerate(dataset.groups):
        patient_images[str(patient)].append(idx)
    patients = list(patient_images)
    patient_targets = np.asarray([dataset.targets[patient_images[p]].max(axis=0) for p in patients])
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = []
    for train_p, val_p in splitter.split(np.zeros(len(patients)), patient_targets):
        train = [i for p in train_p for i in patient_images[patients[p]]]
        val = [i for p in val_p for i in patient_images[patients[p]]]
        result.append((np.asarray(train), np.asarray(val)))
    return result


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, scheduler=None, amp=False, epoch=0, deadline=None):
    training = optimizer is not None
    model.train(training)
    total_loss, all_targets, all_probs = 0.0, [], []
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for step, (images, targets) in enumerate(loader):
        if deadline is not None and time.monotonic() >= deadline:
            raise TrainingBudgetExpired
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step(epoch + (step + 1) / len(loader))
        total_loss += loss.detach().item() * images.size(0)
        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
    y, p = np.concatenate(all_targets), np.concatenate(all_probs)
    return total_loss / len(loader.dataset), macro_roc_auc(y, p)


class TrainingBudgetExpired(Exception):
    """Raised between batches when a time-budgeted run reaches its deadline."""


def _limit_worker_threads(_worker_id):
    """Keep each DataLoader subprocess from starting its own thread pool."""
    torch.set_num_threads(1)


def configure_cpu_parallelism(args):
    """Allocate available logical CPUs between model ops and input workers."""
    cpu_count = os.cpu_count() or 1
    # Reserve workers for disk reads / PIL transforms, and use the remaining
    # logical CPUs for PyTorch's intra-op kernels. Explicit --cpu-threads wins.
    args.workers = min(args.workers, max(0, cpu_count - 1))
    torch_threads = args.cpu_threads or max(1, cpu_count - args.workers)
    torch_threads = min(torch_threads, cpu_count)
    torch.set_num_threads(torch_threads)
    print(
        f"[runtime] {platform.processor() or 'CPU'}: {cpu_count} logical CPUs; "
        f"PyTorch threads={torch_threads}, DataLoader workers={args.workers}",
        flush=True,
    )


def resolve_device(requested="auto", force_cpu=False):
    """Choose CUDA, Intel XPU, or CPU, respecting an explicit device request."""
    if force_cpu or requested == "cpu":
        return torch.device("cpu")
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but this PyTorch environment has no CUDA device")
        return torch.device("cuda")
    if requested == "xpu":
        if not xpu_available:
            raise RuntimeError("XPU requested, but torch.xpu is unavailable; install the PyTorch XPU build and Intel graphics driver")
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if xpu_available:
        return torch.device("xpu")
    return torch.device("cpu")


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def clear_device_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()


def move_optimizer_state(optimizer, device):
    """Move tensor-valued optimizer slots when a run falls back from XPU to CPU."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def make_scaler(device, amp):
    # CUDA uses FP16 gradient scaling. XPU runs BF16 autocast without a scaler,
    # which avoids FP64-dependent scaling on some Intel client GPUs.
    return torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")


class DatasetIndexView:
    """Metadata-only selected view for generating folds on a sampled dataset."""
    def __init__(self, dataset, indices):
        self.targets = dataset.targets[indices]
        if hasattr(dataset, "groups"):
            self.groups = dataset.groups[indices]
        if hasattr(dataset, "strata"):
            self.strata = dataset.strata[indices]

    def __len__(self):
        return len(self.targets)


def select_dataset_indices(dataset, modality, fraction, seed):
    """Select a label-stratified subset once, before splitting into CV folds."""
    if fraction >= 0.999:
        return np.arange(len(dataset))
    if not 0 < fraction <= 1:
        raise ValueError("sample fraction must be in (0, 1]")
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    if modality == "xray":
        # Select whole patients so the later train/validation folds remain
        # patient-disjoint and no patient is partially represented by sampling.
        patient_images = defaultdict(list)
        for idx, patient in enumerate(dataset.groups):
            patient_images[str(patient)].append(idx)
        patients = list(patient_images)
        patient_targets = np.asarray([dataset.targets[patient_images[p]].max(axis=0) for p in patients])
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        _, selected_patients = next(splitter.split(np.zeros(len(patients)), patient_targets))
        return np.asarray([idx for p in selected_patients for idx in patient_images[patients[p]]])

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    _, selected = next(splitter.split(np.zeros(len(dataset)), dataset.targets))
    return selected


def measure_model_throughput(root, dataset_cls, modality, labels, args, device):
    """Measure train and validation image throughput on real, transformed data."""
    train_ds = dataset_cls(root, make_transforms(args.image_size, True, modality))
    val_ds = dataset_cls(root, make_transforms(args.image_size, False, modality))
    probe_count = min(len(train_ds), args.batch_size * 5)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type in {"cuda", "xpu"},
        "persistent_workers": args.workers > 0,
        "worker_init_fn": _limit_worker_threads if args.workers > 0 else None,
    }
    train_loader = DataLoader(Subset(train_ds, np.arange(probe_count)), shuffle=False, **loader_options)
    val_loader = DataLoader(Subset(val_ds, np.arange(probe_count)), shuffle=False, **loader_options)
    model_modality = Modality.CHEST_XRAY if modality == "xray" else Modality.SKIN_LESION
    model = build_model(model_modality, len(labels), pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MultiLabelFocalLoss(args.focal_alpha, args.focal_gamma)
    amp = device.type in {"cuda", "xpu"} and not args.no_amp
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler = make_scaler(device, amp)

    train_iterator = iter(train_loader)
    warm_images, warm_targets = next(train_iterator)
    warm_images, warm_targets = warm_images.to(device), warm_targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
        warm_loss = criterion(model(warm_images), warm_targets)
    if scaler.is_enabled():
        scaler.scale(warm_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        warm_loss.backward()
        optimizer.step()
    sync_device(device)

    train_images = 0
    started = time.perf_counter()
    for images, targets in train_iterator:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            loss = criterion(model(images), targets)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        train_images += images.shape[0]
    sync_device(device)
    train_elapsed = max(time.perf_counter() - started, 1e-6)

    model.eval()
    val_iterator = iter(val_loader)
    next(val_iterator)  # warm up kernels and data access outside the timed interval
    sync_device(device)
    val_images = 0
    started = time.perf_counter()
    with torch.no_grad():
        for images, _ in val_iterator:
            model(images.to(device, non_blocking=True))
            val_images += images.shape[0]
    sync_device(device)
    val_elapsed = max(time.perf_counter() - started, 1e-6)

    del model, optimizer, scaler, train_loader, val_loader
    clear_device_cache(device)
    return train_images / train_elapsed, val_images / val_elapsed


def choose_budget_fraction(args, modalities, device):
    """Estimate a safe shared sampling fraction from measured model throughput."""
    full_seconds = 0.0
    sizes = {}
    for modality in modalities:
        root = Path(args.data_dir) / ("nih_chest_xray" if modality == "xray" else "isic_2019")
        labels = NIH_LABELS if modality == "xray" else ISIC_LABELS
        dataset_cls = NIHChestXrayDataset if modality == "xray" else ISIC2019Dataset
        dataset = dataset_cls(root)
        sizes[modality] = len(dataset)
        if modality == "skin" and dataset.skipped_unmapped:
            print(f"[skin] excluding {dataset.skipped_unmapped} unmapped images (including SCC)", flush=True)
        train_ips, val_ips = measure_model_throughput(root, dataset_cls, modality, labels, args, device)
        fold_cost = args.epochs * len(dataset) * (
            (args.folds - 1) / train_ips + 1.0 / val_ips
        )
        full_seconds += fold_cost
        print(
            f"[{modality}] calibration: train={train_ips:.2f} images/s, "
            f"validation={val_ips:.2f} images/s; full-data CV estimate={fold_cost / 3600:.2f} h",
            flush=True,
        )

    # Ensure even rare labels can populate every validation fold after sampling.
    # For NIH this count is measured across patients, not correlated images.
    minimum_examples = math.inf
    for modality in modalities:
        root = Path(args.data_dir) / ("nih_chest_xray" if modality == "xray" else "isic_2019")
        dataset = (NIHChestXrayDataset if modality == "xray" else ISIC2019Dataset)(root)
        if modality == "skin":
            counts = np.bincount(dataset.strata, minlength=len(ISIC_LABELS))
        else:
            patient_images = defaultdict(list)
            for idx, patient in enumerate(dataset.groups):
                patient_images[str(patient)].append(idx)
            patient_targets = np.asarray([
                dataset.targets[patient_images[p]].max(axis=0) for p in patient_images
            ])
            counts = patient_targets.sum(axis=0)
        positive_counts = counts[counts > 0]
        if positive_counts.size:
            minimum_examples = min(minimum_examples, int(positive_counts.min()))

    # Keep 10% of the requested wall time unallocated for thermal throttling,
    # calibration variance, checkpoint writes, and data-loader startup.
    usable_seconds = args.budget_hours * 3600 * 0.90
    estimated_fraction = min(1.0, usable_seconds / max(full_seconds, 1e-6))
    # A modest cushion over K examples per rare label reduces the chance a
    # random stratified subset leaves a fold without that label.
    minimum_fraction = min(1.0, (1.5 * args.folds / minimum_examples)
                           if math.isfinite(minimum_examples) and minimum_examples else 1.0)
    fraction = max(estimated_fraction, minimum_fraction)
    predicted_hours = full_seconds * fraction / 3600
    print(
        f"[budget] requested={args.budget_hours:.2f} h; sampled-fraction={fraction:.4f}; "
        f"estimated training time={predicted_hours:.2f} h (plus calibration; 10% reserve)",
        flush=True,
    )
    if fraction > estimated_fraction:
        print(
            f"[budget] rare-label fold coverage requires at least {minimum_fraction:.1%} of data; "
            "this may exceed the requested time budget.",
            flush=True,
        )
    for modality, count in sizes.items():
        print(f"[budget] {modality}: selecting about {max(1, round(count * fraction)):,} of {count:,} usable images", flush=True)
    return fraction


def train_modality(args, modality):
    root = Path(args.data_dir) / ("nih_chest_xray" if modality == "xray" else "isic_2019")
    labels = NIH_LABELS if modality == "xray" else ISIC_LABELS
    dataset_cls = NIHChestXrayDataset if modality == "xray" else ISIC2019Dataset
    base = dataset_cls(root)
    print(f"[{modality}] usable training images={len(base)}", flush=True)
    if modality == "skin" and base.skipped_unmapped:
        print(f"[skin] skipped {base.skipped_unmapped} images whose diagnosis is outside the seven-label head (including SCC)", flush=True)
    selected = select_dataset_indices(base, modality, args.sample_fraction, args.seed)
    sampled_view = DatasetIndexView(base, selected)
    folds = [
        (selected[train_idx], selected[val_idx])
        for train_idx, val_idx in make_folds(sampled_view, modality, args.folds, args.seed)
    ]
    print(f"[{modality}] selected={len(selected):,}/{len(base):,} images ({args.sample_fraction:.1%})", flush=True)
    device = resolve_device(args.device, args.cpu)
    amp = device.type in {"cuda", "xpu"} and not args.no_amp
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_overall, best_path = -math.inf, None
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        if args.deadline is not None and time.monotonic() >= args.deadline:
            print(f"[{modality}] time budget reached before fold {fold}; stopping cleanly", flush=True)
            break
        print(f"[{modality} fold {fold}/{len(folds)}] train={len(train_idx)} val={len(val_idx)} device={device}", flush=True)
        train_ds = dataset_cls(root, make_transforms(args.image_size, True, modality))
        val_ds = dataset_cls(root, make_transforms(args.image_size, False, modality))
        train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=amp, persistent_workers=args.workers > 0,
                                  worker_init_fn=_limit_worker_threads if args.workers > 0 else None)
        val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=amp, persistent_workers=args.workers > 0,
                                worker_init_fn=_limit_worker_threads if args.workers > 0 else None)
        model = build_model(Modality.CHEST_XRAY if modality == "xray" else Modality.SKIN_LESION,
                            len(labels), pretrained=args.pretrained).to(device)
        criterion = MultiLabelFocalLoss(args.focal_alpha, args.focal_gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Cosine warm restarts use eta_min + 1/2*(eta_max-eta_min)*
        # (1+cos(pi*T_cur/T_i)). LR smoothly decays from eta_max to eta_min;
        # at T_i it jumps back to eta_max. T_mult lengthens later cycles,
        # making restarts less frequent as training progresses.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.restart_period, T_mult=args.restart_multiplier, eta_min=args.min_lr)
        scaler = make_scaler(device, amp)
        stopper = EarlyStopping(args.patience, args.min_delta)
        fold_path = output / f"{modality}_fold_{fold}.pth"
        budget_expired = False
        for epoch in range(args.epochs):
            try:
                train_loss, train_auc = run_epoch(
                    model, train_loader, criterion, device, optimizer, scaler,
                    scheduler, amp, epoch, deadline=args.deadline,
                )
                with torch.no_grad():
                    val_loss, val_auc = run_epoch(
                        model, val_loader, criterion, device, amp=False, deadline=args.deadline,
                    )
            except TrainingBudgetExpired:
                print(f"[{modality}] time budget reached during fold {fold}; keeping completed checkpoints", flush=True)
                budget_expired = True
                break
            print(f"  epoch {epoch+1}/{args.epochs} loss={train_loss:.4f} auc={train_auc:.4f} "
                  f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
            improved, stop = stopper.update(val_auc)
            if improved:
                checkpoint = {"model_state_dict": model.state_dict(),
                              "architecture": "densenet121" if modality == "xray" else "efficientnet_b4",
                              "modality": modality, "labels": list(labels), "fold": fold,
                              "validation_macro_auc": val_auc, "image_size": args.image_size, "epoch": epoch + 1}
                torch.save(checkpoint, fold_path)
                if val_auc > best_overall:
                    best_overall, best_path = val_auc, fold_path
            if stop:
                print(f"  early stopping at epoch {epoch+1}; best val AUC={stopper.best:.4f}", flush=True)
                break
        del model, optimizer, scheduler, scaler
        clear_device_cache(device)
        if budget_expired:
            break
    if best_path is not None:
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        deployment = output / ("xray.pth" if modality == "xray" else "skin.pth")
        torch.save(checkpoint, deployment)
        print(f"[{modality}] selected fold checkpoint AUC={best_overall:.4f}: {deployment}", flush=True)
    else:
        print(f"[{modality}] no valid ROC-AUC checkpoint was produced", flush=True)


def _stratified_subset(indices, targets, fraction, seed):
    """Select a multilabel-stratified sample while keeping the full fold split.

    NIH folds are first split by patient in ``make_folds``. Subsampling occurs
    only inside each already-separated fold, so no patient can cross between
    the tuning train and validation partitions.
    """
    indices = np.asarray(indices)
    if fraction >= 0.999:
        return indices
    if not 0 < fraction < 1:
        raise ValueError("--tuning-sample-fraction must be in (0, 1]")
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=fraction, random_state=seed
    )
    _, selected = next(splitter.split(np.zeros(len(indices)), targets[indices]))
    return indices[selected]


def tune_modality(args, modality):
    """Use Optuna to tune optimizer/loss/scheduler settings on small CV subsets.

    The selected settings are saved as JSON and returned for the subsequent
    full training run. Tuning is deliberately separate from the held-out final
    evaluation: the final run still uses the requested ``--folds`` setting.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install Optuna with `pip install optuna` to use --tune") from exc

    modality_name = "xray" if modality == "xray" else "skin"
    root = Path(args.data_dir) / ("nih_chest_xray" if modality == "xray" else "isic_2019")
    labels = NIH_LABELS if modality == "xray" else ISIC_LABELS
    dataset_cls = NIHChestXrayDataset if modality == "xray" else ISIC2019Dataset
    base = dataset_cls(root)
    print(f"[{modality}] tuning images={len(base)}", flush=True)
    if modality == "skin" and base.skipped_unmapped:
        print(f"[skin] skipped {base.skipped_unmapped} images whose diagnosis is outside the seven-label head (including SCC)", flush=True)
    folds = make_folds(base, modality, args.tuning_folds, args.seed)
    device = resolve_device(args.device, args.cpu)
    amp = device.type in {"cuda", "xpu"} and not args.no_amp
    if device.type == "cpu":
        print(f"[{modality}] Hyperparameter tuning on CPU can take a long time.", flush=True)

    # Build each dataset once. Subsets below reference the same samples, but
    # train and validation retain distinct deterministic transforms.
    train_ds = dataset_cls(root, make_transforms(args.image_size, True, modality))
    val_ds = dataset_cls(root, make_transforms(args.image_size, False, modality))
    model_modality = Modality.CHEST_XRAY if modality == "xray" else Modality.SKIN_LESION
    max_restart = max(2, min(8, args.tuning_epochs))

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
            "focal_alpha": trial.suggest_float("focal_alpha", 0.25, 0.9),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            "restart_period": trial.suggest_int("restart_period", 2, max_restart),
            "min_lr_ratio": trial.suggest_float("min_lr_ratio", 1e-3, 0.1, log=True),
        }
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(folds, 1):
            seed = args.seed + trial.number * 100 + fold
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.manual_seed_all(seed)

            train_idx = _stratified_subset(
                train_idx, base.targets, args.tuning_sample_fraction, seed
            )
            val_idx = _stratified_subset(
                val_idx, base.targets, args.tuning_sample_fraction, seed + 1
            )
            train_loader = DataLoader(
                Subset(train_ds, train_idx), batch_size=params["batch_size"], shuffle=True,
                num_workers=args.workers, pin_memory=amp,
                persistent_workers=args.workers > 0,
                worker_init_fn=_limit_worker_threads if args.workers > 0 else None,
            )
            val_loader = DataLoader(
                Subset(val_ds, val_idx), batch_size=params["batch_size"], shuffle=False,
                num_workers=args.workers, pin_memory=amp,
                persistent_workers=args.workers > 0,
                worker_init_fn=_limit_worker_threads if args.workers > 0 else None,
            )
            model = build_model(model_modality, len(labels), pretrained=args.pretrained).to(device)
            criterion = MultiLabelFocalLoss(params["focal_alpha"], params["focal_gamma"])
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=params["restart_period"], T_mult=args.restart_multiplier,
                eta_min=params["lr"] * params["min_lr_ratio"],
            )
            scaler = make_scaler(device, amp)
            best_fold_auc = -math.inf
            for epoch in range(args.tuning_epochs):
                run_epoch(
                    model, train_loader, criterion, device, optimizer, scaler,
                    scheduler, amp, epoch,
                )
                with torch.no_grad():
                    _, val_auc = run_epoch(model, val_loader, criterion, device, amp=False)
                if math.isfinite(val_auc):
                    best_fold_auc = max(best_fold_auc, val_auc)
            fold_scores.append(best_fold_auc)
            del model, optimizer, scheduler, scaler, train_loader, val_loader
            clear_device_cache(device)
            score_so_far = float(np.mean([s for s in fold_scores if math.isfinite(s)]))
            if math.isfinite(score_so_far):
                trial.report(score_so_far, step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        valid_scores = [score for score in fold_scores if math.isfinite(score)]
        return float(np.mean(valid_scores)) if valid_scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        study_name=f"{modality_name}_training_hpo",
    )
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    best = dict(study.best_params)
    best["min_lr"] = best["lr"] * best.pop("min_lr_ratio")
    tune_dir = Path(args.output_dir) / "tuning"
    tune_dir.mkdir(parents=True, exist_ok=True)
    params_path = tune_dir / f"{modality_name}_best_params.json"
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump({"modality": modality_name, "best_macro_auc": study.best_value,
                   "trials": args.trials, "parameters": best}, handle, indent=2)
    trial_path = tune_dir / f"{modality_name}_trials.csv"
    trial_fields = ["number", "value", "state"] + sorted(
        {key for trial in study.trials for key in trial.params}
    )
    with trial_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_fields)
        writer.writeheader()
        for trial in study.trials:
            writer.writerow({"number": trial.number, "value": trial.value,
                             "state": trial.state.name, **trial.params})
    print(f"[{modality}] best tuning macro-AUC={study.best_value:.4f}; settings saved to {params_path}", flush=True)
    return best


def load_tuned_params(args, modality):
    """Load one modality's settings previously saved with ``--tune-only``."""
    modality_name = "xray" if modality == "xray" else "skin"
    path = Path(args.output_dir) / "tuning" / f"{modality_name}_best_params.json"
    if not path.is_file():
        raise FileNotFoundError(f"Tuned settings not found at {path}; run with --tune first")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    params = payload.get("parameters", {})
    required = {"lr", "weight_decay", "focal_alpha", "focal_gamma", "batch_size",
                "restart_period", "min_lr"}
    missing = required.difference(params)
    if missing:
        raise ValueError(f"Tuned settings file {path} is missing: {', '.join(sorted(missing))}")
    print(f"[{modality}] loading tuned settings from {path}", flush=True)
    return params


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("xray", "skin", "both"), default="both")
    parser.add_argument("--experiment-mode", choices=("full", "binary"), default="full",
                        help="full preserves the 14/7-label models; binary runs isolated two-class experiments")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts/weights")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="PyTorch CPU threads; 0 allocates logical CPUs after DataLoader workers")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto",
                        help="auto prefers CUDA, then Intel XPU, then CPU")
    parser.add_argument("--budget-hours", type=float, default=0.0,
                        help="full mode estimates subset size; binary mode applies a shared hard time limit")
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                        help="manual stratified fraction when --budget-hours is not used")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--restart-period", type=int, default=5)
    parser.add_argument("--restart-multiplier", type=int, default=2)
    parser.add_argument("--tune", action="store_true", help="run Optuna hyperparameter tuning before full training")
    parser.add_argument("--tune-only", action="store_true", help="tune and save settings without starting full training")
    parser.add_argument("--use-tuned", action="store_true", help="load saved Optuna settings and start full training")
    parser.add_argument("--trials", type=int, default=3, help="number of optional Optuna trials per modality")
    parser.add_argument("--tuning-epochs", type=int, default=3, help="short epochs per fold during tuning")
    parser.add_argument("--tuning-folds", type=int, default=2, help="CV folds used for tuning")
    parser.add_argument("--tuning-sample-fraction", type=float, default=0.2,
                        help="multilabel-stratified fraction used in each tuning split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-total-images", type=int, default=10000,
                        help="binary mode: unique selected images total per dataset")
    parser.add_argument("--max-train-per-class", type=int, default=0,
                        help="optional binary cap per training class; 0 keeps all selected training examples")
    parser.add_argument("--max-val-per-class", type=int, default=0,
                        help="optional binary cap per validation class; 0 keeps all selected validation examples")
    parser.add_argument("--run-tag", default="10k_v2",
                        help="binary checkpoint suffix; existing outputs with the same tag are never overwritten")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="binary mode: held-out validation fraction before per-class caps")
    parser.add_argument("--head-epochs", type=int, default=3,
                        help="binary mode: epochs training the classification head with frozen backbone")
    parser.add_argument("--fine-tune-epochs", type=int, default=1,
                        help="binary mode: optional epochs fine-tuning the final backbone stage; 0 disables")
    parser.add_argument("--resume", default=None,
                        help="binary mode: resume from a matching *_binary_<run-tag>_last.pth checkpoint")
    parser.add_argument("--evaluate", action="store_true",
                        help="binary mode: evaluate a saved best checkpoint on its recorded validation split")
    parser.add_argument("--checkpoint", default=None,
                        help="binary evaluation checkpoint; defaults to <modality>_binary_<run-tag>.pth")
    parser.add_argument("--pretrained", action="store_true", help="initialize with ImageNet weights (may download them)")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1 or args.restart_period < 1:
        raise SystemExit("folds must be >=2; epochs, batch-size and restart-period must be positive")
    if args.workers < 0 or args.cpu_threads < 0:
        raise SystemExit("workers and cpu-threads must be zero or greater")
    if args.budget_hours < 0 or not 0 < args.sample_fraction <= 1:
        raise SystemExit("budget-hours must be >=0 and sample-fraction must be in (0, 1]")
    if args.max_train_per_class < 0 or args.max_val_per_class < 0:
        raise SystemExit("max-train-per-class and max-val-per-class must be zero (uncapped) or greater")
    if args.target_total_images < 2:
        raise SystemExit("target-total-images must be at least 2")
    if not args.run_tag or not all(character.isalnum() or character in "_-" for character in args.run_tag):
        raise SystemExit("run-tag may contain only letters, numbers, underscore, and hyphen")
    if args.head_epochs < 1 or args.fine_tune_epochs < 0:
        raise SystemExit("head-epochs must be positive and fine-tune-epochs must be zero or greater")
    if not 0 < args.val_fraction < 0.5:
        raise SystemExit("val-fraction must be greater than 0 and less than 0.5")
    if args.tune_only:
        args.tune = True
    if args.tune and args.use_tuned:
        raise SystemExit("Use either --tune or --use-tuned, not both")
    if args.budget_hours and (args.tune or args.use_tuned):
        raise SystemExit("Time-budget mode skips Optuna so the available time goes to model training")
    if args.experiment_mode == "binary" and (args.tune or args.use_tuned):
        raise SystemExit("Optuna tuning currently applies to full-label training; binary experiments use their reproducible local defaults")
    if args.experiment_mode == "binary" and args.resume and args.modality == "both":
        raise SystemExit("Resume one binary modality at a time with --modality xray or --modality skin")
    if args.experiment_mode == "binary" and args.evaluate and args.modality == "both" and args.checkpoint:
        raise SystemExit("When evaluating both modalities, omit --checkpoint so each default binary checkpoint is selected")
    if args.experiment_mode == "binary" and args.evaluate and args.resume:
        raise SystemExit("Use either --evaluate or --resume, not both")
    if args.experiment_mode == "binary" and args.checkpoint and not args.evaluate:
        raise SystemExit("--checkpoint is only used with --evaluate")
    if args.experiment_mode == "full" and (args.evaluate or args.resume or args.checkpoint):
        raise SystemExit("--evaluate, --resume, and --checkpoint are supported in --experiment-mode binary")
    if args.trials < 1 or args.tuning_epochs < 1 or args.tuning_folds < 2:
        raise SystemExit("trials and tuning-epochs must be positive; tuning-folds must be >=2")
    if not 0 < args.tuning_sample_fraction <= 1:
        raise SystemExit("tuning-sample-fraction must be in (0, 1]")
    configure_cpu_parallelism(args)
    device = resolve_device(args.device, args.cpu)
    print(f"[runtime] selected device={device}", flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(args.seed)
    modalities = ("xray", "skin") if args.modality == "both" else (args.modality,)
    args.deadline = None
    if args.budget_hours:
        budget_started = time.monotonic()
        if args.experiment_mode == "full":
            args.sample_fraction = choose_budget_fraction(args, modalities, device)
        # In binary mode, class caps bound work; in full mode, throughput
        # calibration chooses a stratified subset. Keep one minute for saving.
        args.deadline = budget_started + args.budget_hours * 3600 - 60
    if args.experiment_mode == "binary":
        if args.evaluate:
            for modality in modalities:
                evaluate_args = argparse.Namespace(**vars(args))
                evaluate_binary_checkpoint(evaluate_args, modality)
        else:
            for modality in modalities:
                modality_args = argparse.Namespace(**vars(args))
                train_binary_experiment(modality_args, modality)
        return
    for modality in modalities:
        modality_args = argparse.Namespace(**vars(args))
        if args.tune or args.use_tuned:
            best = tune_modality(modality_args, modality) if args.tune else load_tuned_params(modality_args, modality)
            if args.tune_only:
                continue
            for key in ("lr", "weight_decay", "focal_alpha", "focal_gamma", "batch_size",
                        "restart_period", "min_lr"):
                setattr(modality_args, key, best[key])
        train_modality(modality_args, modality)


if __name__ == "__main__":
    main()
