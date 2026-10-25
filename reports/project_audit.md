# Capstone project audit

Audit snapshot: 25 October 2026. Dataset statistics and model results below were read from the files present in this workspace; they are not estimates. This is a research prototype, not a clinically validated device.

## Project intent versus local implementation

The pitch PDF is a 13-page rasterized slide deck, so text extraction yields no selectable text. Its rendered slides describe a dual-modality system with DenseNet-121, EfficientNet-B4, Grad-CAM, PubMed/WHO retrieval, triage, and a web dashboard. Slide 3 calls the chest data “NIH CheXpert” and claims “224,000+ labeled scans”; that combines two different dataset names. The files and loader here are **NIH ChestX-ray14**, not Stanford CheXpert. The local NIH corpus has 112,120 PNG images and `Data_Entry_2017.csv`. The ISIC corpus is the 2019 training split. The official ISIC challenge identifies 25,331 training images, eight populated training diagnosis categories, an additional OOD class for its test challenge, and a CC BY-NC license for the aggregate training data. Cite and comply with the source license and attribution requirements before redistribution.

The pitch's 92% chest ROC-AUC, 88% skin sensitivity, 81% Grad-CAM IoU, and 96% RAG factuality are **targets shown in the proposal**, not measured project results. There is no evidence in the current artifacts that these targets were reached.

## Dataset inventory and provenance

| Local data | Files and metadata | Verified counts | Notes |
|---|---|---:|---|
| NIH ChestX-ray14 | Kaggle ID `nih-chest-xrays/data`; `Data_Entry_2017.csv`; `train_val_list.txt`; `test_list.txt`; 12 image folders | 112,120 PNG images; 112,120 metadata rows; 30,805 patient IDs | Images are 1024 x 1024 in a deterministic 250-image full-decode sample; 249 were grayscale `L`, one `RGBA`; no decode failures. |
| ISIC 2019 | Kaggle ID `salviohexia/isic-2019-skin-lesion-images-for-classification`; ground-truth and metadata CSVs; diagnosis folders | 25,331 JPG images and ground-truth rows | Deterministic 250-image full-decode sample, all RGB; mixed native resolutions. Metadata has 25,331 rows and 23,247 non-empty lesion IDs. |

NIH CSV labels are report-derived weak labels; `No Finding` is the dataset's label and must not be equated with a definitive normal radiograph. Raw NIH positive counts include multi-finding rows: No Finding 60,361; Pneumonia 1,431; Infiltration 19,894; Effusion 13,317; Atelectasis 11,559; Nodule 6,331; Mass 5,782; Pneumothorax 5,302; Consolidation 4,667; Pleural Thickening 3,385; Cardiomegaly 2,776; Emphysema 2,516; Edema 2,303; Fibrosis 1,686; Hernia 227. Categories overlap across rows. The binary task keeps only exact `No Finding` and exact `Pneumonia`, so only 322 pneumonia images qualify; 51,437 other/mixed-label rows are excluded from this binary task.

ISIC ground truth columns are MEL 4,522; NV 12,875; BCC 3,323; BKL 2,624; AK 867; SCC 628; VASC 253; DF 239; UNK 0. The local CSV contains a zero-positive `UNK` column, not trainable positive examples. `AK` is mapped to the training/API label `AKIEC`. Binary MEL/NV excludes 7,934 other diagnoses. ISIC's official challenge training collection is CC BY-NC; image-level contributor attribution/licensing may apply, so retain the official citations and don't redistribute the data without checking rights.

## Quality checks, missing data, and leakage

- Binary loader/run metadata recorded zero missing image paths for the eligible binary records. Baseline training successfully consumed its selected examples. Separate random samples of 100 images per corpus decoded with PIL and had no failures. A full byte-for-byte decode of every image and whole-corpus exact/near-duplicate analysis have **not** been run.
- ISIC ground truth has no multi-positive or all-zero rows except the empty `UNK` output column (0 positive records). The loader requires one-hot labels for binary MEL/NV and excludes other diagnoses.
- For the saved 10,000-image binary run, train/validation samples had no repeated IDs across splits. NIH patient groups did not cross its selected train/validation split. ISIC lesion IDs did not cross its split. ISIC metadata contains no patient ID, so distinct lesions from the same person may still cross splits.
- NIH's official lists contain 86,524 `train_val` images and 25,596 `test` images, with no image-ID overlap. The existing 10,000-image random selection did **not** preserve that official holdout: 1,380 selected train images and 321 selected validation images occur in `test_list.txt`, involving 923 patient IDs in the saved selected subset. Do not use the NIH official test list to claim independent performance for this baseline. The project has not created a replacement untouched test partition.
- ISIC's downloaded Kaggle tree contains its 25,331 training images and labels; it does not contain the official 8,238-image labeled challenge test set. Existing ISIC validation metrics are not independent test metrics.
- Baseline thresholds and model selection used the validation data. That validation split must not later be relabeled as a test set.

## Models, runtime, and reproducibility

Binary checkpoints use ImageNet-pretrained DenseNet-121 (X-ray) and EfficientNet-B4 (skin), 224 x 224 RGB images, ImageNet normalization, weighted cross-entropy, 3 frozen-backbone head epochs and 1 fine-tuning epoch. Batch size is 8, AdamW learning rate 1e-4 and weight decay 1e-4. The saved run seed is 42. The fresh evaluation commands selected Intel XPU; training metadata records 8 logical CPU cores, 4 PyTorch threads, and 4 DataLoader workers. Existing tagged checkpoints are not overwritten by a new run tag.

Current selected binary run (`10k_v2`) splits:

| Task | Train | Validation | Total selected |
|---|---:|---:|---:|
| NIH No Finding / Pneumonia | 7,786 / 254 = 8,040 | 1,892 / 68 = 1,960 | 10,000 |
| ISIC MEL / NV | 2,104 / 5,928 = 8,032 | 495 / 1,473 = 1,968 | 10,000 |

Fresh checkpoint evaluations are in `artifacts/logs/*_binary_10k_v2_evaluation.json`; the epoch records are in the matching `*_training.jsonl` files. Per-image predictions are saved beside the evaluations as `*_validation_predictions.csv`. X-ray validation: ROC-AUC 0.6188 (95% patient-group bootstrap CI 0.5480-0.6893), PR-AUC 0.0673, Pneumonia sensitivity 14.71% (CI 7.14%-22.73%), specificity 97.67% (CI 96.63%-98.60%), accuracy 94.80%; confusion matrix `[[1848, 44], [58, 10]]`. Skin validation: ROC-AUC 0.7958 (95% lesion-group bootstrap CI 0.7595-0.8284), PR-AUC 0.6010, MEL sensitivity 75.35% (CI 69.94%-80.50%), specificity 68.77% (CI 65.63%-71.87%), accuracy 70.43%; confusion matrix `[[373, 122], [460, 1013]]`. The X-ray all-negative behavior has improved to nonzero sensitivity, but 58 of 68 pneumonia cases remain missed at the selected threshold. This does not establish that the model is reliable.

Optuna code exists for full-label training, but binary tuning is rejected by the CLI and no model comparison or hyperparameter search has been run. Only these two baseline architecture/configuration runs are documented. The seven-hour budget has not been used to claim a broader search.

## Module status

| Component | Present | Audit status |
|---|---|---|
| Training | `train_engine.py`; tagged binary weights, metadata, logs | Binary baseline exists. Full NIH is multi-label. ISIC full-label path has just been updated in code to include SCC and group splits by lesion; no new full-label checkpoint has been trained yet. |
| Inference API | `main.py`, `/health`, `/analyze` | FastAPI wiring exists. Current default checkpoint paths are not the binary checkpoint names; configure `.env`. Runtime needs model weights, usable RAG index, guidelines, and for generated summaries an OpenAI key. |
| Vision | `vision_engine.py` | DenseNet/EfficientNet loading, ImageNet transforms, sigmoid multi-label and softmax binary output. Automatic modality choice is an image saturation heuristic, not a validated modality classifier; explicit modality is safer. |
| Grad-CAM | `gradcam_engine.py`, `generate_gradcam_examples.py` | Hook-based class-specific overlays now save separate source image, heatmap and overlay. Eight real validation TP/FP/FN/TN examples were generated on XPU. No localization score was calculated. Visual spot checks show attention near image borders/markers in selected examples, so heatmaps do not establish clinically meaningful localization. |
| RAG | `rag_engine.py` | PDF loading, chunking, BioBERT embeddings, Chroma indexing/retrieval code exists. `data/guidelines/` is empty and no local Chroma index exists, so ingestion/retrieval has not run. BioBERT is not a sentence-embedding-tuned retriever; compare it empirically before claiming retrieval quality. |
| Summary/triage | `triage_llm.py` | Deterministic prototype thresholds and a context-only prompt exist. Thresholds are not calibrated or clinically reviewed. The API response currently omits structured source citations/evidence IDs and a dedicated uncertainty field. |
| Dashboard | None detected | A FastAPI endpoint exists, but no user-facing dashboard has been implemented. |

Automated checks: 18 tests passed on the Anaconda Python runtime outside the sandbox (PyTorch XPU DLL access is blocked in the sandbox). Tests cover binary selection/metrics, synthetic eight-class lesion-group folds, Grad-CAM sidecar outputs, basic API validation, and deterministic triage; there is no end-to-end test for live model + Grad-CAM + Chroma + LLM. A `.env` file is absent, and no guideline PDFs are present.

## Changes made after the binary baseline

The full-label ISIC loader now includes SCC as the eighth populated class and retains lesion IDs. The UNK class remains excluded because the supplied training ground truth has no positive UNK rows; the official test OOD class cannot be learned from zero training examples. Full-label ISIC cross-validation now uses `StratifiedGroupKFold` and verifies lesion groups are disjoint. A five-fold smoke split passed on full NIH and ISIC metadata; every ISIC validation fold included all eight populated classes. Binary class orders and binary checkpoints are unchanged. A fresh full-label ISIC run is required to produce an eight-output checkpoint.

## Primary sources and usage notes

- NIH ChestX-ray14 official source and original attribution: <https://nihcc.app.box.com/v/ChestXray-NIHCC> and Wang et al., “ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases,” CVPR 2017, <https://arxiv.org/abs/1705.02315>.
- Official ISIC 2019 challenge page, class definitions, collection size, test size, license and required attribution: <https://challenge.isic-archive.com/data/> and <https://challenge.isic-archive.com/landing/2019/>.
- Download script dataset IDs are Kaggle mirrors; official provenance/terms should take precedence over a mirror's description.

## Remaining work (not represented as completed)

1. Create and freeze clean, group-separated train/validation/test manifests; the previous baseline NIH official holdout is contaminated for this run, and no ISIC patient IDs are supplied.
2. Re-run full-label ISIC with eight outputs, add a real unknown/OOD strategy if one is in scope, and retain the existing binary baselines.
3. Implement bounded architecture/loss/augmentation/hyperparameter comparisons under a declared budget and record completed trial counts.
4. Add ROC, PR, threshold, calibration, confidence-interval and performance/resource plots from actual predictions; report threshold selection and confidence intervals without test leakage.
5. Grad-CAM artifacts are generated for validation TP/FP/FN/TN samples; evaluate localization only against valid annotations if available.
6. Obtain permitted, authoritative guideline PDFs, build and evaluate the retrieval index, preserve source citations, and test unsupported-evidence abstention.
7. Improve structured safety output, uncertainty/OOD handling, clinician review workflow, and add a dashboard only after integration checks.
8. Run end-to-end API tests with real configured checkpoints, documents, model key (or deterministic no-key abstention), invalid uploads, and failure cases.
