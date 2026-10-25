"""FastAPI application wiring image inference, Grad-CAM, RAG, and triage."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from triage_llm import ClinicalAssistant, determine_triage

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


logger = logging.getLogger("medical_capstone")

DEFAULT_XRAY_LABELS = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
)
DEFAULT_SKIN_LABELS = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "12000000"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
_VISION_LOCK = threading.Lock()


class AnalyzeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    diagnosis: str
    confidence: float
    triage_level: str
    triage_detail: dict
    heatmap_path: str
    original_image_path: str
    heatmap_image_path: str
    clinical_summary: str
    modality: str
    probabilities: dict[str, float]
    model_version: str
    uncertainty: float
    evidence_sources: list[dict]
    limitations: list[str]


@dataclass(frozen=True)
class Settings:
    xray_checkpoint: Path
    skin_checkpoint: Path
    xray_labels: tuple[str, ...]
    skin_labels: tuple[str, ...]
    chroma_dir: Path
    guidelines_dir: Path
    collection_name: str
    embedding_model: str
    llm_model: str
    openai_api_key: str | None
    rag_top_k: int
    auto_index: bool


@dataclass
class Services:
    vision: object
    rag: object
    model_versions: dict[str, str]


def _load_settings() -> Settings:
    """Read environment configuration, including a local optional .env file."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    def read_labels(variable: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
        raw = os.getenv(variable)
        if not raw:
            return fallback
        try:
            labels = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{variable} must be a JSON array of label names") from exc
        if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
            raise ValueError(f"{variable} must be a non-empty JSON array of strings")
        return tuple(labels)

    return Settings(
        xray_checkpoint=Path(os.getenv("XRAY_CHECKPOINT", "artifacts/weights/xray.pth")),
        skin_checkpoint=Path(os.getenv("SKIN_CHECKPOINT", "artifacts/weights/skin.pth")),
        xray_labels=read_labels("XRAY_LABELS", DEFAULT_XRAY_LABELS),
        skin_labels=read_labels("SKIN_LABELS", DEFAULT_SKIN_LABELS),
        chroma_dir=Path(os.getenv("CHROMA_DIR", "artifacts/chroma_db")),
        guidelines_dir=Path(os.getenv("GUIDELINES_DIR", "data/guidelines")),
        collection_name=os.getenv("CHROMA_COLLECTION", "clinical_guidelines"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "dmis-lab/biobert-base-cased-v1.1"),
        llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        rag_top_k=max(1, int(os.getenv("RAG_TOP_K", "4"))),
        auto_index=os.getenv("RAG_AUTO_INDEX", "true").lower() in {"true", "1", "yes"},
    )


@lru_cache(maxsize=1)
def _get_services() -> Services:
    """Load model checkpoints and open or build the RAG index on first request."""
    from rag_engine import MedicalRAG
    from vision_engine import Modality, VisionEngine, build_model, load_checkpoint

    settings = _load_settings()
    for path in (settings.xray_checkpoint, settings.skin_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(
                f"Trained checkpoint missing at {path}. Train/export checkpoints before using /analyze."
            )

    def load_configured_model(modality, path: Path, labels: tuple[str, ...]):
        # Training checkpoints include their output label order. Enforce it
        # when present so same-sized but differently ordered heads cannot be
        # interpreted as the wrong diseases at inference time.
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # Older PyTorch versions do not expose weights_only.
            checkpoint = torch.load(path, map_location="cpu")
        checkpoint_labels = checkpoint.get("labels") if isinstance(checkpoint, dict) else None
        task = checkpoint.get("task", "multilabel") if isinstance(checkpoint, dict) else "multilabel"
        output_mode = "binary" if task == "binary_classification" else "multilabel"
        decision_threshold = float(checkpoint.get("decision_threshold", 0.5)) if output_mode == "binary" else 0.5
        positive_label = (
            checkpoint.get("positive_label") or checkpoint.get("metrics", {}).get("positive_class")
            if output_mode == "binary" else None
        )
        if checkpoint_labels is not None and tuple(checkpoint_labels) != labels:
            raise ValueError(
                f"Checkpoint {path} label order does not match configured labels. "
                "Set XRAY_LABELS/SKIN_LABELS to the checkpoint order or retrain/export consistently."
            )
        model = build_model(modality, len(labels))
        return load_checkpoint(model, path), output_mode, decision_threshold, positive_label, path.name

    xray_model, xray_output_mode, xray_threshold, xray_positive, xray_version = load_configured_model(
        Modality.CHEST_XRAY, settings.xray_checkpoint, settings.xray_labels
    )
    skin_model, skin_output_mode, skin_threshold, skin_positive, skin_version = load_configured_model(
        Modality.SKIN_LESION, settings.skin_checkpoint, settings.skin_labels
    )
    vision = VisionEngine(
        xray_model,
        skin_model,
        {
            Modality.CHEST_XRAY: settings.xray_labels,
            Modality.SKIN_LESION: settings.skin_labels,
        },
        output_modes={Modality.CHEST_XRAY: xray_output_mode, Modality.SKIN_LESION: skin_output_mode},
        decision_thresholds={Modality.CHEST_XRAY: xray_threshold, Modality.SKIN_LESION: skin_threshold},
        positive_labels={key: value for key, value in {
            Modality.CHEST_XRAY: xray_positive, Modality.SKIN_LESION: skin_positive,
        }.items() if value is not None},
    )

    rag = MedicalRAG(
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
    )
    if settings.chroma_dir.exists():
        try:
            rag.open_index()
        except Exception:
            if not settings.auto_index:
                raise
            rag.index_pdfs(settings.guidelines_dir, reset=True)
    elif settings.auto_index:
        rag.index_pdfs(settings.guidelines_dir, reset=True)
    else:
        raise FileNotFoundError(
            f"Chroma index missing at {settings.chroma_dir}; index guidelines or enable RAG_AUTO_INDEX"
        )

    collection = getattr(rag.vectorstore, "_collection", None)
    if collection is not None and collection.count() == 0:
        if not settings.auto_index:
            raise FileNotFoundError("The configured Chroma collection is empty")
        rag.index_pdfs(settings.guidelines_dir, reset=True)

    return Services(vision=vision, rag=rag, model_versions={
        Modality.CHEST_XRAY.value: xray_version,
        Modality.SKIN_LESION.value: skin_version,
    })


def _probability_uncertainty(probabilities: dict[str, float]) -> float:
    """Normalized mean Bernoulli entropy; a score diagnostic, not calibration."""
    import math

    if not probabilities:
        return 1.0
    eps = 1e-12
    entropies = []
    for raw in probabilities.values():
        p = min(max(float(raw), eps), 1.0 - eps)
        entropies.append(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0))
    return float(sum(entropies) / len(entropies))


def _analyze_sync(image: Image.Image, requested_modality: str | None) -> AnalyzeResponse:
    try:
        services = _get_services()
        settings = _load_settings()
    except Exception as exc:
        logger.exception("Pipeline initialization failed")
        raise HTTPException(status_code=503, detail=f"Analysis pipeline is not ready: {exc}") from exc

    try:
        from gradcam_engine import GradCAM, default_target_layer
        from vision_engine import Modality

        with _VISION_LOCK:
            modality = Modality(requested_modality) if requested_modality else None
            prediction = services.vision.predict(image, modality=modality)
            triage = determine_triage(prediction.probabilities)

            model = services.vision.models[prediction.modality]
            label_order = services.vision.labels[prediction.modality]
            class_index = label_order.index(prediction.top_label)
            with GradCAM(model, default_target_layer(prediction.modality), device=services.vision.device) as cam:
                cam_result = cam.generate(image, class_index)

        context = services.rag.search(prediction.top_label, top_k=settings.rag_top_k)
        evidence_sources = []
        for document in context:
            metadata = getattr(document, "metadata", {}) or {}
            evidence_sources.append({
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page"),
                "title": metadata.get("title"),
            })
        if context:
            try:
                assistant = ClinicalAssistant(
                    model=settings.llm_model,
                    api_key=settings.openai_api_key,
                )
                summary = assistant.summarize(
                    prediction.top_label,
                    prediction.top_score,
                    triage,
                    context,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=f"Clinical summary service is not ready: {exc}") from exc
        else:
            # The assistant's empty-context response is deterministic and does
            # not require constructing a remote LLM client or an API key.
            summary = ClinicalAssistant.summarize(
                None,
                prediction.top_label,
                prediction.top_score,
                triage,
                context,
            )

        return AnalyzeResponse(
            diagnosis=prediction.top_label,
            confidence=prediction.top_score,
            triage_level=triage.description,
            triage_detail=triage.to_dict(),
            heatmap_path=str(cam_result.output_path.resolve()),
            original_image_path=str(cam_result.original_path.resolve()),
            heatmap_image_path=str(cam_result.heatmap_path.resolve()),
            clinical_summary=summary,
            modality=prediction.modality.value,
            probabilities=prediction.probabilities,
            model_version=services.model_versions[prediction.modality.value],
            uncertainty=_probability_uncertainty(prediction.probabilities),
            evidence_sources=evidence_sources,
            limitations=[
                "Research prototype: model scores are not diagnoses or calibrated clinical probabilities.",
                "A low-risk output does not rule out serious disease; urgent symptoms require appropriate clinical assessment.",
                "Triage thresholds are unvalidated and require clinician review.",
            ],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {exc}") from exc


def _validate_modality(modality: str | None) -> None:
    if modality is not None and modality not in {"chest_xray", "skin_lesion"}:
        raise HTTPException(status_code=422, detail="modality must be chest_xray or skin_lesion")


def _decode_uploaded_image(contents: bytes) -> Image.Image:
    if not contents:
        raise HTTPException(status_code=422, detail="The uploaded image is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES} byte upload limit")
    try:
        with Image.open(io.BytesIO(contents)) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds {MAX_IMAGE_PIXELS} pixel limit",
                )
            opened.load()
            return opened.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=415, detail="Upload must be a valid supported image") from exc


app = FastAPI(
    title="Multi-Modal Medical Disease Detector & AI Clinical Assistant",
    description=(
        "Research prototype API. Image predictions and triage rules are unvalidated "
        "and are not a substitute for clinical assessment."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness endpoint that does not download models or initialize the RAG index."""
    return {"status": "ok", "pipeline": "lazy initialization; use /analyze to check readiness"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(..., description="Medical image to analyze"),
    modality: str | None = Form(
        default=None,
        description="Optional override: chest_xray or skin_lesion",
    ),
) -> AnalyzeResponse:
    """Analyze one uploaded image and return prediction, triage, and explanation."""
    _validate_modality(modality)

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    image = _decode_uploaded_image(contents)

    return await run_in_threadpool(_analyze_sync, image, modality)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=False)
