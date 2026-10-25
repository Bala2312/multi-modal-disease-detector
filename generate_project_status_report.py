"""Build a PDF status report from the saved baseline outputs and audit files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf" / "medical_ai_capstone_research_status_report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#10233F")
BLUE = colors.HexColor("#2D6CDF")
PALE = colors.HexColor("#EEF3F9")
MUTED = colors.HexColor("#5E6B7A")
RED = colors.HexColor("#B42318")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleA", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23, leading=28, textColor=NAVY, alignment=TA_LEFT, spaceAfter=7))
styles.add(ParagraphStyle(name="SubA", parent=styles["Normal"], fontSize=9.5, leading=14, textColor=MUTED, spaceAfter=12))
styles.add(ParagraphStyle(name="HeadA", parent=styles["Heading1"], fontSize=15, leading=19, textColor=NAVY, spaceBefore=4, spaceAfter=8))
styles.add(ParagraphStyle(name="SubHeadA", parent=styles["Heading2"], fontSize=10.5, leading=13, textColor=BLUE, spaceBefore=5, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyA", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=colors.HexColor("#263445"), spaceAfter=5))
styles.add(ParagraphStyle(name="SmallA", parent=styles["BodyText"], fontSize=7, leading=9, textColor=MUTED))
styles.add(ParagraphStyle(name="CellA", parent=styles["BodyText"], fontSize=7, leading=9))
styles.add(ParagraphStyle(name="HeadCellA", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=colors.white))
styles.add(ParagraphStyle(name="AlertA", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=RED, backColor=colors.HexColor("#FFF2F0"), borderColor=colors.HexColor("#F5B7B1"), borderWidth=.6, borderPadding=7, spaceBefore=3, spaceAfter=8))

def p(text, style="BodyA"):
    return Paragraph(text, styles[style])

def table(rows, widths):
    cells = [[Paragraph(str(value), styles["HeadCellA"] if ri == 0 else styles["CellA"]) for value in row]
             for ri, row in enumerate(rows)]
    obj = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    obj.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#CFD8E3")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PALE]),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return obj

def image(path: Path, width: float, height: float):
    flow = Image(str(path))
    scale = min(width / flow.imageWidth, height / flow.imageHeight)
    flow.drawWidth = flow.imageWidth * scale
    flow.drawHeight = flow.imageHeight * scale
    return flow

def page_frame(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY); canvas.rect(0, h-9*mm, w, 9*mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white); canvas.setFont("Helvetica-Bold", 7.5)
    canvas.drawString(16*mm, h-6.2*mm, "MEDICAL AI CAPSTONE | RESEARCH STATUS")
    canvas.setStrokeColor(colors.HexColor("#D9E1EA")); canvas.line(16*mm, 13*mm, w-16*mm, 13*mm)
    canvas.setFillColor(MUTED); canvas.setFont("Helvetica", 7)
    canvas.drawString(16*mm, 8*mm, "Prototype engineering report - validation only - 25 October 2026")
    canvas.drawRightString(w-16*mm, 8*mm, str(doc.page))
    canvas.restoreState()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

audit = read_json(ROOT / "reports" / "dataset_audit" / "dataset_audit.json")
evaluations = {
    key: read_json(ROOT / "artifacts" / "logs" / f"{key}_binary_10k_v2_evaluation.json")
    for key in ("xray", "skin")
}
metrics = {key: value["metrics"] for key, value in evaluations.items()}
examples = read_json(ROOT / "reports" / "experiments" / "binary_baselines" / "gradcam_examples" / "gradcam_examples.json")["examples"]
example_by_key = {(row["modality"], row["case"]): row for row in examples}
figdir = ROOT / "reports" / "experiments" / "binary_baselines"
audit_dir = ROOT / "reports" / "dataset_audit"

architecture_path = figdir / "pipeline_architecture.png"
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
boxes = [(0.1,"Image"),(2.1,"224x224\nnormalize"),(4.1,"DenseNet-121\nor EfficientNet-B4"),(6.1,"Binary scores\nthreshold"),(8.1,"Triage + RAG\nprototype API")]
for x,label in boxes:
    rect=plt.Rectangle((x,.95),1.65,1.05,facecolor="#EEF3F9",edgecolor="#2D6CDF",linewidth=1.5)
    ax.add_patch(rect); ax.text(x+.825,1.48,label,ha="center",va="center",fontsize=8,fontweight="bold",color="#10233F")
for i in range(len(boxes)-1):
    ax.annotate("",xy=(boxes[i+1][0]-.08,1.48),xytext=(boxes[i][0]+1.72,1.48),arrowprops=dict(arrowstyle="->",color="#0F9D91",lw=1.8))
ax.text(5,2.55,"Current binary inference path; RAG is not configured locally",ha="center",fontsize=10,fontweight="bold",color="#10233F")
fig.tight_layout(); fig.savefig(architecture_path,dpi=180,bbox_inches="tight"); plt.close(fig)

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
                        topMargin=15*mm, bottomMargin=18*mm,
                        title="Medical AI Capstone Research Status Report",
                        author="Medical AI Capstone Project")
story = [Spacer(1, 6*mm), p("Medical AI Capstone", "TitleA"),
         p("Architecture, dataset audit, baseline validation, Grad-CAM and implementation status", "SubA"),
         p("Executive summary", "HeadA"),
         p("The workspace contains a working binary research baseline for NIH ChestX-ray14 and ISIC 2019, with two pretrained CNN checkpoints, evaluation code, a FastAPI scaffold, a Grad-CAM engine, and a local-PDF RAG engine. I reloaded the saved checkpoints and re-evaluated their recorded validation splits on Intel Arc XPU. Dataset counts, prediction-level metrics, confidence intervals, curves and qualitative explanation examples in this report come from files generated from those runs.")]

story += [table([
    ["Positive class","Validation images","ROC-AUC (95% group CI)","PR-AUC","Sensitivity (95% group CI)","Specificity"],
    ["Pneumonia", f"{evaluations['xray']['validation_count']:,}",
     f"{metrics['xray']['roc_auc_positive_class']:.3f} (0.548-0.689)",
     f"{metrics['xray']['average_precision_positive_class']:.3f}",
     f"{metrics['xray']['sensitivity']:.1%} (7.1%-22.7%)", f"{metrics['xray']['specificity']:.1%}"],
    ["MEL", f"{evaluations['skin']['validation_count']:,}",
     f"{metrics['skin']['roc_auc_positive_class']:.3f} (0.760-0.828)",
     f"{metrics['skin']['average_precision_positive_class']:.3f}",
     f"{metrics['skin']['sensitivity']:.1%} (69.9%-80.5%)", f"{metrics['skin']['specificity']:.1%}"],
], [24*mm, 24*mm, 37*mm, 18*mm, 39*mm, 24*mm]), Spacer(1, 4*mm),
    p("Key result: X-ray pneumonia sensitivity is 14.7%; 58 of 68 validation pneumonia cases were missed. The prior all-negative behavior improved, but the result remains inadequate for clinical use. Skin MEL sensitivity is 75.4%, with 44.8% positive precision and 68.8% specificity.", "AlertA"),
    p("No independent test results are available. The validation threshold was selected on the same validation split. These are research metrics, not clinical validation.", "SmallA"), PageBreak()]

story += [p("1. Project scope and architecture", "HeadA"),
          p("The proposal describes chest X-ray and skin lesion classification, Grad-CAM, guideline-grounded retrieval, deterministic triage, clinician review and a user-facing dashboard. The current binary models use ImageNet-pretrained DenseNet-121 for chest images and EfficientNet-B4 for dermoscopy. Their binary heads return two logits and use softmax probabilities; the NIH full-label path uses independent sigmoid outputs. Input preprocessing resizes to 224 x 224 RGB and applies ImageNet mean/std normalization."),
          image(architecture_path, 178*mm, 52*mm),
          table([["Component","Implementation status"],
                 ["Binary image models","Present; tagged pretrained checkpoints; exact output order in metadata"],
                 ["Full-label models","NIH 14-label path exists; ISIC updated to eight populated labels and lesion-group folds; new full-label skin checkpoint not trained"],
                 ["Grad-CAM","Generated original/heatmap/overlay for eight actual validation cases; no localization metric"],
                 ["RAG","PDF/Chroma/BioBERT code exists; local guidelines folder is empty, so no index or retrieval evaluation"],
                 ["Clinical summary / triage","Prototype prompt and deterministic matrix exist; thresholds are unvalidated"],
                 ["API / dashboard","FastAPI routes exist; no configured .env, source documents, or user-facing dashboard"],
          ], [42*mm, 136*mm]),
          p("Hardware during fresh evaluation: Intel Arc XPU selected, 8 logical CPU cores, 4 PyTorch threads and 4 DataLoader workers. The environment has no project .env or OPENAI_API_KEY. The RAG source folder has no PDFs, so the full /analyze flow is not ready to serve grounded summaries.", "BodyA"), PageBreak()]

story += [p("2. Dataset provenance and split audit", "HeadA"),
          p("The pitch slide calls the chest data 'NIH CheXpert' and reports 224,000+ scans. Local files and the loader identify NIH ChestX-ray14: 112,120 PNG images, 112,120 label rows, and 30,805 patient IDs. NIH ChestX-ray14 and Stanford CheXpert are distinct datasets. The skin data is the ISIC 2019 training split: 25,331 JPEG images and matching labels."),
          image(audit_dir / "class_distribution.png", 178*mm, 67*mm),
          table([["Dataset / binary subset","Eligible data and selected counts","Train / validation split","Grouping"],
                 ["NIH ChestX-ray14","60,361 exact No Finding; 322 exact Pneumonia. Selected 9,678 + 322 = 10,000.","7,786 + 254 = 8,040 / 1,892 + 68 = 1,960","Patient groups kept separate in selected split"],
                 ["ISIC 2019","4,522 MEL; 12,875 NV. Selected 2,599 + 7,401 = 10,000.","2,104 + 5,928 = 8,032 / 495 + 1,473 = 1,968","Lesion groups kept separate; patient IDs unavailable"],
          ], [32*mm, 55*mm, 52*mm, 39*mm]),
          p("Image checks: deterministic samples of 250 per dataset were fully decoded. NIH sample images were 1024 x 1024, mostly grayscale; ISIC sample images were RGB at mixed resolutions. The label-to-image inventory reported zero missing paths. Full-corpus byte-level decoding and pixel/near-duplicate scans have not been performed.", "BodyA"),
          p("Leakage limitation: the NIH selection included 1,380 training and 321 validation images from NIH's official test_list.txt, involving 923 selected patient IDs. Therefore that official test list is not untouched for this baseline. ISIC's labeled official test set is not present locally. The saved baseline validation results must not be presented as independent test performance.", "AlertA"), PageBreak()]

story += [p("3. Training protocol and metrics", "HeadA"),
          p("The two saved binary runs each selected 10,000 unique images with seed 42. Both used pretrained weights, 3 frozen-backbone head epochs followed by 1 final-stage fine-tuning epoch, batch size 8, AdamW (learning rate 1e-4, weight decay 1e-4), inverse-frequency class-weighted cross-entropy, and early stopping/checkpoint selection by validation ROC-AUC. X-ray class order is No Finding / Pneumonia; skin order is MEL / NV. Validation-only F1 selection yielded thresholds 0.2653 and 0.4495, respectively."),
          image(figdir / "epoch_learning_curves.png", 178*mm, 66*mm),
          image(figdir / "validation_roc_curves.png", 178*mm, 65*mm),
          image(figdir / "validation_precision_recall_curves.png", 178*mm, 65*mm), PageBreak()]

story += [p("4. Validation outputs", "HeadA"),
          image(figdir / "validation_threshold_curves.png", 178*mm, 68*mm),
          image(figdir / "validation_calibration_curves.png", 178*mm, 68*mm),
          table([["Metric","X-ray / Pneumonia","Skin / MEL"],
                 ["Accuracy / balanced accuracy", f"{metrics['xray']['accuracy']:.1%} / {metrics['xray']['balanced_accuracy']:.1%}", f"{metrics['skin']['accuracy']:.1%} / {metrics['skin']['balanced_accuracy']:.1%}"],
                 ["Macro precision / recall / F1", f"{metrics['xray']['precision_macro']:.1%} / {metrics['xray']['recall_macro']:.1%} / {metrics['xray']['f1_macro']:.1%}", f"{metrics['skin']['precision_macro']:.1%} / {metrics['skin']['recall_macro']:.1%} / {metrics['skin']['f1_macro']:.1%}"],
                 ["Positive precision / F1", f"{metrics['xray']['positive_precision']:.1%} / {metrics['xray']['positive_f1']:.3f}", f"{metrics['skin']['positive_precision']:.1%} / {metrics['skin']['positive_f1']:.3f}"],
                 ["MCC / Brier / ECE-10", f"{metrics['xray']['matthews_correlation_coefficient']:.3f} / {metrics['xray']['brier_score_positive_class']:.3f} / {metrics['xray']['expected_calibration_error_10_bins']:.3f}", f"{metrics['skin']['matthews_correlation_coefficient']:.3f} / {metrics['skin']['brier_score_positive_class']:.3f} / {metrics['skin']['expected_calibration_error_10_bins']:.3f}"],
                 ["Confusion matrix (actual rows)", "[[1848, 44], [58, 10]]", "[[373, 122], [460, 1013]]"],
          ], [48*mm, 65*mm, 65*mm]),
          p("ROC-AUC 95% intervals use 1,000 patient-group resamples for NIH and lesion-group resamples for ISIC. Calibration curves are diagnostic only; no calibration model was fitted. X-ray PR-AUC 0.067 is only moderately above the positive prevalence (3.5%). Skin PR-AUC is 0.601 against 25.2% prevalence.", "BodyA"), PageBreak()]

xray_fn = example_by_key[("xray", "false_negative")]
skin_fp = example_by_key[("skin", "false_positive")]

story += [p("5. Grad-CAM examples and interpretation", "HeadA"),
          p("The explanation script selected the model's predicted class for each validation case and saved the original image, colorized heatmap and overlay. Target layers: DenseNet-121 `features.denseblock4`; EfficientNet-B4 `features.7`. The figure samples below include a pneumonia false negative and an MEL false positive. These are class-discriminative saliency maps, not validated lesion localization."),
          table([["X-ray pneumonia false negative","Skin MEL false positive"],
                 [image(ROOT / example_by_key[("xray", "false_negative")]["overlay_path"], 82*mm, 78*mm),
                  image(ROOT / example_by_key[("skin", "false_positive")]["overlay_path"], 82*mm, 78*mm)]],
                [88*mm, 88*mm]),
          table([["Example","True / predicted","Positive score / threshold","Grad-CAM target"],
                 ["X-ray FN", "Pneumonia / No Finding", f"{xray_fn['positive_probability']:.3f} / {xray_fn['decision_threshold']:.3f}", "No Finding (predicted class)"],
                 ["Skin FP", "NV / MEL", f"{skin_fp['positive_probability']:.3f} / {skin_fp['decision_threshold']:.3f}", "MEL (predicted class)"],
          ], [27*mm, 44*mm, 46*mm, 59*mm]),
          p("Visual spot checks show border/marker-area attention in selected examples. This is a warning signal for possible shortcuts and requires further systematic review. No IoU, pointing-game or perturbation metric is reported; the pitch's 81% IoU remains an unmeasured target.", "AlertA"),
          p("Other generated examples and the path manifest are in `reports/experiments/binary_baselines/gradcam_examples/`.", "SmallA"), PageBreak()]

story += [p("6. Grounded RAG, triage, application and next steps", "HeadA"),
          table([["Proposal target","Measured status"],
                 ["NIH ROC-AUC 92%","Baseline 61.9%; no multi-architecture comparison"],
                 ["ISIC sensitivity 88%","Baseline 75.4%"],
                 ["Grad-CAM localization IoU 81%","Not measured; no corresponding localization evaluation"],
                 ["RAG factuality 96%","Not measured; no guideline PDFs or index present"],
          ], [60*mm, 116*mm]),
          p("Code changes in this phase: full-label ISIC now includes SCC as an eighth populated class and uses lesion-aware StratifiedGroupKFold; the UNK column is not treated as trained because it has zero positive examples. Existing binary checkpoints and label orders were preserved. Evaluation now saves validation prediction CSVs, grouped-bootstrap confidence intervals, balanced accuracy, PR-AUC, MCC, Brier score and ECE. Dataset audit and chart scripts were added. Grad-CAM saves original/heatmap/overlay sidecars. The API response now includes checkpoint filename, entropy-based score uncertainty, evidence source metadata and limitations."),
          p("Verification: 18 tests pass; Python compilation passes; the full-dataset five-fold smoke check found zero patient/lesion overlap. The actual checkpoint evaluations and eight Grad-CAM examples ran on XPU. No new full model training, Optuna trial, RAG indexing, LLM call or dashboard run was performed in this audit."),
          p("Required next work", "SubHeadA"),
          p("1. Define a leakage-safe experiment protocol before new tuning. All exact-label NIH Pneumonia examples were consumed by the existing baseline; the official test list overlaps baseline patients. To obtain a valid positive-class test, revise the new task definition/split before training or use a new external cohort. ISIC's official test labels are not locally available and patient IDs are missing.") ,
          p("2. Compare imbalance strategies and architectures with patient/lesion-aware splits. Tune thresholds on validation only, then evaluate once on a truly untouched test. Preserve current baseline checkpoints.") ,
          p("3. Obtain licensed, authoritative guidelines for the RAG index; compare BioBERT with a sentence-embedding model; evaluate retrieval relevance, citation correctness, unsupported claims and abstention with human review.") ,
          p("4. Add system tests and a dashboard after configured checkpoints, guidelines and key handling are available. Have qualified clinicians review triage language and escalation rules."),
          p("Sources: <link href='https://nihcc.app.box.com/v/ChestXray-NIHCC' color='#2D6CDF'>NIH ChestX-ray data</link>; <link href='https://challenge.isic-archive.com/landing/2019/' color='#2D6CDF'>ISIC 2019 challenge definition</link>; <link href='https://challenge.isic-archive.com/data/' color='#2D6CDF'>ISIC data, license and attribution</link>.", "SmallA"),
          p("Output files: `reports/project_audit.md`; `reports/dataset_audit/`; `reports/experiments/binary_baselines/`; `artifacts/logs/*_binary_10k_v2_evaluation.json`; `artifacts/weights/*_binary_10k_v2.pth`.", "SmallA")]

doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
print(OUT)
