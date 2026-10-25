# Multi-Modal Medical Disease Detector & AI Clinical Assistant

Modular capstone scaffold for image classification, visual explanations, retrieval over supplied medical guidance, and guarded clinical-summary generation.

## Project structure

```text
.
├── main.py                 # FastAPI endpoint and pipeline orchestration (Step 6)
├── download_datasets.py    # Kaggle dataset download and extraction script
├── vision_engine.py        # Modality routing and DenseNet/EfficientNet inference (Step 2)
├── gradcam_engine.py       # Grad-CAM map generation and overlay (Step 3)
├── rag_engine.py           # PDF chunking, BioBERT embeddings, Chroma retrieval (Step 4)
├── triage_llm.py           # Deterministic triage and grounded LLM synthesis (Step 5)
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Test dependencies
├── .env.example            # Local configuration template
├── tests/                  # Deterministic triage and API validation tests
├── data/
│   └── guidelines/         # Operator-provided PDF guidance for indexing
├── artifacts/
│   ├── weights/            # Trained model checkpoints; not committed
│   └── chroma_db/           # Persistent local retrieval index
└── outputs/
    └── heatmaps/           # Generated explainability images; not committed
```

The proposed architecture is a research prototype. Model weights, dataset-specific label maps, validation, and clinical review are required before its predictions can be interpreted for patient care. The pitch PDF is treated as project context; implementation instructions come from the user's request.

## Configure and run

1. Install a PyTorch/torchvision pair matching the machine, then install `requirements.txt` (and `requirements-dev.txt` to run tests).
2. Copy `.env.example` to `.env`; set checkpoint paths, output labels in the exact training order, and `OPENAI_API_KEY`.
3. Put approved guideline PDFs in `data/guidelines/`. The first analysis indexes them into the local Chroma directory when `RAG_AUTO_INDEX=true`.
4. Start the API with `python main.py`, then open `http://127.0.0.1:8000/docs` to call `/analyze`.

`/analyze` requires both trained checkpoints, matching labels, indexed guideline PDFs, and an OpenAI API key. It returns HTTP 503 with setup details when the pipeline is not ready. Thresholds in `triage_llm.py` are prototype routing values, not validated clinical thresholds.

Run the available unit/API checks with `pytest`.

Download the two large Kaggle datasets with `python download_datasets.py`. The script reads `KAGGLE_API_TOKEN` from the environment or prompts for it without echoing input. Dataset files are ignored by Git because of their size.

## Local binary experiments

`train_engine.py` keeps the existing `--experiment-mode full` (the default) for the serving model outputs: 14 NIH findings and seven ISIC diagnoses. The separate `--experiment-mode binary` runs save to new tagged filenames and never overwrite the full-label checkpoints or the earlier untagged binary checkpoints. NIH keeps only exact `No Finding` and exact `Pneumonia` rows, uses patient IDs for a stratified group split, then samples without replacement. ISIC keeps unambiguous one-hot MEL/NV rows and preserves their source class ratio; repeated `lesion_id` values stay in one split. ISIC metadata has no patient identifier, so patient leakage cannot be ruled out.

Binary mode selects **10,000 images total per dataset** by default. With the current files and seed 42, NIH selects all 322 exact-label Pneumonia images plus 9,678 No Finding images. ISIC selects about 2,599 MEL and 7,401 NV images from its 4,522 MEL and 12,875 NV eligible cases. It then makes a group-aware 80/20 split, prints exact total/per-class/split counts before training, and verifies both classes occur in both splits. Optional `--max-train-per-class` and `--max-val-per-class` caps default to 0 (uncapped); specifying caps that reduce the 10,000 total prints a limitation warning. Images load on demand rather than being held as tensors in RAM.

The current seed-42 split yields 8,040 train / 1,960 validation X-rays (7,786/1,892 No Finding and 254/68 Pneumonia) and 8,032 train / 1,968 validation skin images (2,104/495 MEL and 5,928/1,473 NV). Both class distributions appear in both splits and the group intersection is empty. The exact split counts are printed by the script and can change with dataset or dependency updates.

X-ray training uses inverse-frequency weighted cross entropy, not replacement sampling, so the selected dataset remains imbalanced and contains no repeated samples. The code logs the output-index mapping and positive class. It selects an F1-maximizing decision threshold on validation probabilities, saves that threshold and positive label, reports sensitivity, specificity, precision, F1, ROC-AUC, confusion matrix, and per-class metrics, and applies the same threshold in binary API inference. The best checkpoint is selected by validation ROC-AUC. **The previous X-ray result is not considered fixed until a new evaluation shows improved Pneumonia sensitivity; validation metrics and a validation-selected threshold are not independent test results.** Reserve an untouched patient/lesion-separated test set for final reporting.

Defaults are 224-pixel inputs, batch size eight, three frozen-backbone head epochs, one final-stage fine-tuning epoch, seed 42, and early stopping. Outputs include `xray_binary_10k_v2.pth`, `skin_binary_10k_v2.pth`, tagged `_last.pth` resume checkpoints, tagged JSON metadata, and JSONL logs. If a tagged checkpoint already exists, the script refuses to overwrite it; pass a new `--run-tag` for a fresh run.

On Windows, install the project's Python dependencies first. For Intel Arc/XPU acceleration, install the matching Intel graphics driver and then the official PyTorch XPU wheels; the project's ordinary PyPI build may be CPU-only. Verify `torch.xpu.is_available()` before using `--device xpu`. PyTorch's official XPU installation and supported-device guidance is [here](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html).

```powershell
python -m pip install -r requirements.txt
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/xpu
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

If the availability check prints `False`, use `--device auto` to fall back to CPU or configure a compatible Intel driver and XPU-enabled PyTorch build. Intel integrated graphics do not provide CUDA.

Run both 10,000-image binary experiments under one shared seven-hour deadline:

```powershell
python train_engine.py --experiment-mode binary --modality both --device auto --pretrained --target-total-images 10000 --head-epochs 3 --fine-tune-epochs 1 --image-size 224 --batch-size 8 --workers 4 --patience 2 --budget-hours 7 --seed 42
```

To run only one modality, change `--modality both` to `xray` or `skin`. To resume an interrupted experiment, use its latest checkpoint and the same modality/settings, for example:

```powershell
python train_engine.py --experiment-mode binary --modality xray --device auto --pretrained --resume artifacts/weights/xray_binary_10k_v2_last.pth --target-total-images 10000 --head-epochs 3 --fine-tune-epochs 1 --batch-size 8 --workers 4 --patience 2 --budget-hours 2 --seed 42
python train_engine.py --experiment-mode binary --modality skin --device auto --pretrained --resume artifacts/weights/skin_binary_10k_v2_last.pth --target-total-images 10000 --head-epochs 3 --fine-tune-epochs 1 --batch-size 8 --workers 4 --patience 2 --budget-hours 2 --seed 42
```

Evaluate a saved best checkpoint on its recorded validation images:

```powershell
python train_engine.py --experiment-mode binary --modality xray --evaluate --checkpoint artifacts/weights/xray_binary_10k_v2.pth --device auto
python train_engine.py --experiment-mode binary --modality skin --evaluate --checkpoint artifacts/weights/skin_binary_10k_v2.pth --device auto
```

To use a binary checkpoint in the existing API, explicitly point `XRAY_CHECKPOINT` or `SKIN_CHECKPOINT` to its tagged file and set the matching JSON label order (`["No Finding","Pneumonia"]` or `["MEL","NV"]`). Checkpoint metadata selects softmax and the validation-selected binary threshold; full-label checkpoints continue to use sigmoid. API configuration is per modality, so the other modality still needs its own compatible checkpoint. These validation results are not independent test results and do not establish clinical performance.
