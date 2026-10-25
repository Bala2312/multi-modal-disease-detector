import csv

import numpy as np
import pytest
from PIL import Image

import train_engine
from train_engine import (
    BinaryImageDataset,
    _atomic_torch_save,
    binary_metrics,
    cap_binary_split,
    inverse_frequency_class_weights,
    load_isic_binary_records,
    load_xray_binary_records,
    parse_args,
    select_binary_indices,
    select_binary_threshold,
    stratified_group_split,
    torch,
)
from vision_engine import Modality, VisionEngine


def _write_image(path, suffix=".png"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(40, 90, 130)).save(path.with_suffix(suffix))
    return path.with_suffix(suffix)


def test_nih_binary_filter_keeps_only_exact_labels_and_patient_groups(tmp_path):
    root = tmp_path / "nih_chest_xray"
    image_root = root / "images_001" / "images"
    image_root.mkdir(parents=True)
    rows = []
    for index in range(10):
        filename = f"normal_{index}.png"
        _write_image(image_root / filename)
        rows.append({"Image Index": filename, "Finding Labels": "No Finding", "Patient ID": f"nf{index}"})
    for index in range(10):
        filename = f"pneumonia_{index}.png"
        _write_image(image_root / filename)
        rows.append({"Image Index": filename, "Finding Labels": "Pneumonia", "Patient ID": f"p{index}"})
    rows.append({"Image Index": "mixed.png", "Finding Labels": "Pneumonia|Effusion", "Patient ID": "mixed"})
    with (root / "Data_Entry_2017.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Image Index", "Finding Labels", "Patient ID"])
        writer.writeheader()
        writer.writerows(rows)

    records, classes, exclusions = load_xray_binary_records(root)

    assert classes == ("No Finding", "Pneumonia")
    assert len(records) == 20
    assert exclusions["excluded_other_or_mixed"] == 1
    assert set(record[1] for record in records) == {0, 1}


def test_isic_binary_filter_requires_one_hot_mel_or_nv(tmp_path):
    root = tmp_path / "isic_2019"
    image_root = root / "images"
    image_root.mkdir(parents=True)
    columns = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    rows = []
    metadata = []
    for index in range(10):
        image_id = f"mel_{index}"
        _write_image(image_root / image_id, ".jpg")
        labels = {name: 0 for name in columns}
        labels["MEL"] = 1
        rows.append({"image": image_id, **labels})
        metadata.append({"image": image_id, "lesion_id": "shared-lesion" if index < 2 else f"lesion-{index}"})
    for index in range(10):
        image_id = f"nv_{index}"
        _write_image(image_root / image_id, ".jpg")
        labels = {name: 0 for name in columns}
        labels["NV"] = 1
        rows.append({"image": image_id, **labels})
        metadata.append({"image": image_id, "lesion_id": f"nv-lesion-{index}"})
    for image_id, enabled in (("bcc", ("BCC",)), ("mixed", ("MEL", "NV"))):
        _write_image(image_root / image_id, ".jpg")
        labels = {name: 0 for name in columns}
        for name in enabled:
            labels[name] = 1
        rows.append({"image": image_id, **labels})
        metadata.append({"image": image_id, "lesion_id": ""})
    with (root / "ISIC_2019_Training_GroundTruth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", *columns])
        writer.writeheader()
        writer.writerows(rows)
    with (root / "ISIC_2019_Training_Metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "lesion_id"])
        writer.writeheader()
        writer.writerows(metadata)

    records, classes, exclusions = load_isic_binary_records(root)

    assert classes == ("MEL", "NV")
    assert len(records) == 20
    assert exclusions["excluded_other_diagnoses"] == 1
    assert exclusions["excluded_ambiguous_labels"] == 1
    assert records[0][2] == "shared-lesion" or any(record[2] == "shared-lesion" for record in records)


def test_group_split_caps_are_disjoint_and_reproducible():
    records = []
    for label in (0, 1):
        for index in range(20):
            records.append((None, label, f"{label}-{index}", f"{label}-{index}"))
    dataset = BinaryImageDataset(records, ("MEL", "NV"))
    train, val = stratified_group_split(dataset, seed=21, val_fraction=0.2)
    capped_train = cap_binary_split(dataset, train, 6, seed=5)
    capped_val = cap_binary_split(dataset, val, 3, seed=6)

    assert set(dataset.groups[capped_train]).isdisjoint(set(dataset.groups[capped_val]))
    assert np.bincount(dataset.targets[capped_train]).tolist() == [6, 6]
    assert np.bincount(dataset.targets[capped_val]).tolist() == [3, 3]
    assert capped_train.tolist() == cap_binary_split(dataset, train, 6, seed=5).tolist()


def test_ten_thousand_selection_policy_is_unique_and_preserves_class_policy():
    records = []
    for index in range(4):
        records.append((None, 1, f"p-{index}", f"p-{index}"))
    for index in range(16):
        records.append((None, 0, f"n-{index}", f"n-{index}"))
    xray_view = BinaryImageDataset(records, ("No Finding", "Pneumonia"))
    xray_selected = select_binary_indices(xray_view, "xray", 10, seed=42)
    assert np.bincount(xray_view.targets[xray_selected]).tolist() == [6, 4]
    assert len(np.unique(xray_selected)) == 10

    skin_view = BinaryImageDataset(
        [(None, label, f"g-{i}", f"id-{i}") for i, label in enumerate([0] * 4 + [1] * 16)],
        ("MEL", "NV"),
    )
    skin_selected = select_binary_indices(skin_view, "skin", 10, seed=42)
    assert np.bincount(skin_view.targets[skin_selected]).tolist() == [2, 8]
    assert len(np.unique(skin_selected)) == 10
    assert skin_selected.tolist() == select_binary_indices(skin_view, "skin", 10, seed=42).tolist()


def test_binary_metrics_and_checkpoint_save(tmp_path):
    metrics = binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.9, 0.1]]),
        ("MEL", "NV"),
    )
    assert metrics["per_class"]["MEL"]["recall"] == 0.5
    assert metrics["positive_class"] == "MEL"
    assert len(metrics["confusion_matrix"]) == 2
    assert inverse_frequency_class_weights([0] * 9 + [1], 2).tolist() == pytest.approx([10 / 18, 5.0])
    threshold = select_binary_threshold(
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.9, 0.1]]),
        ("No Finding", "Pneumonia"),
    )
    threshold_metrics = binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.9, 0.1]]),
        ("No Finding", "Pneumonia"), threshold,
    )
    assert threshold_metrics["decision_threshold"] == threshold
    assert threshold_metrics["sensitivity"] > 0
    assert "specificity" in threshold_metrics

    path = tmp_path / "checkpoint.pth"
    _atomic_torch_save({"task": "binary_classification", "value": torch.tensor([3])}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["task"] == "binary_classification"
    assert loaded["value"].item() == 3


def test_binary_inference_uses_softmax_and_full_mode_remains_default(monkeypatch):
    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(3, 2)

        def forward(self, images):
            return self.classifier(images.mean(dim=(2, 3)))

    engine = VisionEngine(
        TinyClassifier(), TinyClassifier(),
        {Modality.CHEST_XRAY: ("No Finding", "Pneumonia"), Modality.SKIN_LESION: ("MEL", "NV")},
        device="cpu", output_modes={Modality.CHEST_XRAY: "binary", Modality.SKIN_LESION: "multilabel"},
        decision_thresholds={Modality.CHEST_XRAY: 0.35},
    )
    prediction = engine.predict(Image.new("RGB", (16, 16)), modality=Modality.CHEST_XRAY)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    expected = "Pneumonia" if prediction.probabilities["Pneumonia"] >= 0.35 else "No Finding"
    assert prediction.top_label == expected

    monkeypatch.setattr("sys.argv", ["train_engine.py"])
    assert parse_args().experiment_mode == "full"


def test_one_epoch_binary_training_smoke_saves_best_and_resume_checkpoints(tmp_path, monkeypatch):
    root = tmp_path / "data" / "nih_chest_xray"
    image_root = root / "images_001" / "images"
    image_root.mkdir(parents=True)
    rows = []
    for label_name, label, prefix in (("No Finding", 0, "nf"), ("Pneumonia", 1, "p")):
        for index in range(20):
            filename = f"{prefix}_{index}.png"
            _write_image(image_root / filename)
            rows.append({"Image Index": filename, "Finding Labels": label_name, "Patient ID": f"{prefix}{index}"})
    with (root / "Data_Entry_2017.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Image Index", "Finding Labels", "Patient ID"])
        writer.writeheader()
        writer.writerows(rows)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(3, 2)

        def forward(self, images):
            return self.classifier(images.mean(dim=(2, 3)))

    monkeypatch.setattr(train_engine, "build_model", lambda *_args, **_kwargs: TinyModel())
    torch.set_num_threads(2)
    args = type("Args", (), {
        "data_dir": str(tmp_path / "data"), "seed": 9, "val_fraction": 0.2,
        "max_train_per_class": 0, "max_val_per_class": 0, "target_total_images": 30,
        "run_tag": "smoke", "image_size": 16,
        "device": "cpu", "cpu": False, "no_amp": True, "workers": 0,
        "batch_size": 4, "pretrained": False, "head_epochs": 1,
        "fine_tune_epochs": 0, "lr": 1e-2, "weight_decay": 0.0,
        "output_dir": str(tmp_path / "artifacts" / "weights"), "resume": None,
        "patience": 2, "deadline": None,
    })()

    train_engine.train_binary_experiment(args, "xray")

    best = torch.load(tmp_path / "artifacts" / "weights" / "xray_binary_smoke.pth", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "artifacts" / "weights" / "xray_binary_smoke_last.pth", map_location="cpu", weights_only=False)
    assert best["task"] == "binary_classification"
    assert best["labels"] == ["No Finding", "Pneumonia"]
    assert "optimizer_state_dict" in last
    assert best["decision_threshold"] == best["metrics"]["decision_threshold"]
    assert (tmp_path / "artifacts" / "weights" / "xray_binary_smoke.json").is_file()
