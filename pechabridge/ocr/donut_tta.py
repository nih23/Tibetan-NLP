"""Test-time augmentation helpers for DONUT OCR line crops.

The functions in this module are intentionally independent of the CLI and
Workbench layers.  Callers provide the original page image, the detected line
box, the loaded DONUT runtime, and the preprocessing function that matches the
checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

BoxXYWH = Tuple[int, int, int, int]
BoxXYXY = Tuple[int, int, int, int]
PreprocessFn = Callable[[Image.Image], Image.Image]


@dataclass(frozen=True)
class TTABoxVariant:
    """One clipped TTA box variant in absolute page coordinates."""

    name: str
    box_xyxy: BoxXYXY

    @property
    def box_xywh(self) -> BoxXYWH:
        x1, y1, x2, y2 = self.box_xyxy
        return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


@dataclass(frozen=True)
class TTACandidate:
    """DONUT text hypothesis for one crop variant."""

    text: str
    confidence: float
    variant: str
    box_xyxy: BoxXYXY


@dataclass(frozen=True)
class TTACluster:
    """A Levenshtein cluster used for consensus voting."""

    index: int
    candidates: List[TTACandidate]

    @property
    def size(self) -> int:
        return len(self.candidates)

    @property
    def average_confidence(self) -> float:
        if not self.candidates:
            return 0.0
        return float(sum(c.confidence for c in self.candidates) / len(self.candidates))


@dataclass(frozen=True)
class TTAConsensusResult:
    """Final golden string plus full TTA diagnostics."""

    text: str
    confidence: float
    winner_cluster_index: int
    winner_cluster_size: int
    winner_cluster_average_confidence: float
    candidates: List[TTACandidate]
    clusters: List[TTACluster]

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "winner_cluster_index": self.winner_cluster_index,
            "winner_cluster_size": self.winner_cluster_size,
            "winner_cluster_average_confidence": self.winner_cluster_average_confidence,
            "candidates": [asdict(c) for c in self.candidates],
            "clusters": [
                {
                    "index": cluster.index,
                    "size": cluster.size,
                    "average_confidence": cluster.average_confidence,
                    "candidates": [asdict(c) for c in cluster.candidates],
                }
                for cluster in self.clusters
            ],
        }


def _coerce_image_size(image_size: Sequence[int]) -> Tuple[int, int]:
    if len(image_size) < 2:
        raise ValueError("image_size must contain width and height.")
    image_w = max(1, int(image_size[0]))
    image_h = max(1, int(image_size[1]))
    return image_w, image_h


def _clip_xyxy(x1: int, y1: int, x2: int, y2: int, image_size: Sequence[int]) -> Optional[BoxXYXY]:
    image_w, image_h = _coerce_image_size(image_size)
    cx1 = max(0, min(image_w - 1, int(round(x1))))
    cy1 = max(0, min(image_h - 1, int(round(y1))))
    cx2 = max(0, min(image_w, int(round(x2))))
    cy2 = max(0, min(image_h, int(round(y2))))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def _shift_box_vertically(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    delta_y: int,
    image_size: Sequence[int],
) -> Optional[BoxXYXY]:
    image_w, image_h = _coerce_image_size(image_size)
    _ = image_w
    height = max(1, int(y2 - y1))
    ny1 = int(y1 + delta_y)
    ny2 = int(y2 + delta_y)
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if ny2 > image_h:
        overflow = ny2 - image_h
        ny1 -= overflow
        ny2 = image_h
    if ny2 - ny1 != height:
        ny2 = ny1 + height
    return _clip_xyxy(x1, ny1, x2, ny2, image_size)


def xyxy_to_xywh(box_xyxy: Sequence[int]) -> BoxXYWH:
    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must contain exactly 4 values.")
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def xywh_to_xyxy(box_xywh: Sequence[int]) -> BoxXYXY:
    if len(box_xywh) != 4:
        raise ValueError("box_xywh must contain exactly 4 values.")
    x, y, w, h = [int(v) for v in box_xywh]
    return x, y, x + max(0, w), y + max(0, h)


def generate_tta_box_variants_xywh(
    box_xywh: Sequence[int],
    image_size: Sequence[int],
    variations: int = 7,
) -> List[TTABoxVariant]:
    """Return clipped TTA box variants for a YOLO ``(x, y, w, h)`` box.

    The default first seven variants are:
    original, +10%/+20% upward height, +10%/+20% downward height,
    +15% symmetric height, and a slight upward vertical shift.

    Duplicate boxes can occur near page boundaries after clipping; duplicates
    are removed so callers may receive fewer than ``variations`` usable crops.
    """

    if int(variations) <= 0:
        return []
    x1, y1, x2, y2 = xywh_to_xyxy(box_xywh)
    image_w, image_h = _coerce_image_size(image_size)
    box = _clip_xyxy(x1, y1, x2, y2, (image_w, image_h))
    if box is None:
        return []
    x1, y1, x2, y2 = box
    h = max(1, y2 - y1)

    specs: List[Tuple[str, Optional[BoxXYXY]]] = [
        ("original", _clip_xyxy(x1, y1, x2, y2, (image_w, image_h))),
        ("expand_up_10", _clip_xyxy(x1, y1 - int(round(h * 0.10)), x2, y2, (image_w, image_h))),
        ("expand_up_20", _clip_xyxy(x1, y1 - int(round(h * 0.20)), x2, y2, (image_w, image_h))),
        ("expand_down_10", _clip_xyxy(x1, y1, x2, y2 + int(round(h * 0.10)), (image_w, image_h))),
        ("expand_down_20", _clip_xyxy(x1, y1, x2, y2 + int(round(h * 0.20)), (image_w, image_h))),
        (
            "expand_symmetric_15",
            _clip_xyxy(
                x1,
                y1 - int(round(h * 0.15)),
                x2,
                y2 + int(round(h * 0.15)),
                (image_w, image_h),
            ),
        ),
        ("shift_up_10", _shift_box_vertically(x1, y1, x2, y2, -int(round(h * 0.10)), (image_w, image_h))),
        ("shift_down_10", _shift_box_vertically(x1, y1, x2, y2, int(round(h * 0.10)), (image_w, image_h))),
        (
            "expand_symmetric_25",
            _clip_xyxy(
                x1,
                y1 - int(round(h * 0.25)),
                x2,
                y2 + int(round(h * 0.25)),
                (image_w, image_h),
            ),
        ),
    ]

    out: List[TTABoxVariant] = []
    seen: set[BoxXYXY] = set()
    for name, maybe_box in specs:
        if maybe_box is None or maybe_box in seen:
            continue
        seen.add(maybe_box)
        out.append(TTABoxVariant(name=name, box_xyxy=maybe_box))
        if len(out) >= int(variations):
            break
    return out


def generate_tta_boxes_xywh(
    box_xywh: Sequence[int],
    image_size: Sequence[int],
    variations: int = 7,
) -> List[BoxXYWH]:
    """Return only ``(x, y, w, h)`` boxes for callers that do not need names."""

    return [variant.box_xywh for variant in generate_tta_box_variants_xywh(box_xywh, image_size, variations)]


def generate_tta_box_variants_xyxy(
    box_xyxy: Sequence[int],
    image_size: Sequence[int],
    variations: int = 7,
) -> List[TTABoxVariant]:
    """Return clipped TTA box variants for an internal ``(x1, y1, x2, y2)`` box."""

    return generate_tta_box_variants_xywh(xyxy_to_xywh(box_xyxy), image_size, variations)


def _ensure_pil_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image).astype(np.uint8, copy=False)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr).convert("RGB")


def crop_tta_variants_xyxy(
    image: Image.Image | np.ndarray,
    box_xyxy: Sequence[int],
    variations: int = 7,
) -> List[Tuple[TTABoxVariant, Image.Image]]:
    """Crop all TTA variants from the original scan/page image."""

    pil = _ensure_pil_rgb(image)
    variants = generate_tta_box_variants_xyxy(box_xyxy, pil.size, variations)
    crops: List[Tuple[TTABoxVariant, Image.Image]] = []
    for variant in variants:
        crops.append((variant, pil.crop(variant.box_xyxy).convert("RGB")))
    return crops


def _strip_special_token_strings(text: str, tokenizer: Any) -> str:
    out = str(text or "")
    try:
        special_toks = sorted(
            [str(t) for t in getattr(tokenizer, "all_special_tokens", []) if isinstance(t, str) and t],
            key=len,
            reverse=True,
        )
    except Exception:
        special_toks = []
    for tok in special_toks:
        out = out.replace(tok, "")
    return out.strip()


def build_donut_generate_kwargs(
    generate_config: Optional[Mapping[str, Any]],
    *,
    max_len: int = 0,
    force_scores: bool = True,
) -> Dict[str, Any]:
    """Build DONUT ``generate`` kwargs and optionally force score outputs."""

    cfg = generate_config or {}
    gen_kwargs: Dict[str, Any] = {}
    for key in (
        "decoder_start_token_id",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "max_length",
        "max_new_tokens",
        "min_new_tokens",
        "num_beams",
        "do_sample",
        "temperature",
        "top_p",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "length_penalty",
        "early_stopping",
        "bad_words_ids",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "use_cache",
    ):
        value = cfg.get(key)
        if value is not None:
            gen_kwargs[key] = value

    if int(max_len) > 0:
        gen_kwargs["max_length"] = int(max_len)
    elif "max_length" not in gen_kwargs and "max_new_tokens" not in gen_kwargs:
        gen_kwargs["max_length"] = 160

    if force_scores:
        # Required for confidence extraction from Hugging Face generation
        # outputs.  ``scores`` are step-wise logits; we convert them to token
        # probabilities after generation.
        gen_kwargs["return_dict_in_generate"] = True
        gen_kwargs["output_scores"] = True
    return gen_kwargs


def _generated_token_confidences(
    *,
    model: Any,
    generation_output: Any,
    tokenizer: Any,
) -> List[float]:
    """Return mean token probabilities for generated sequences.

    Hugging Face ``generate(..., return_dict_in_generate=True,
    output_scores=True)`` returns one logits tensor per generated decoding step
    in ``scores``.  The numerically safest path is
    ``model.compute_transition_scores(..., normalize_logits=True)``: it gathers
    the actually selected token at each step and returns log-probabilities,
    while also handling beam-search backtracking through ``beam_indices``.
    We exponentiate those log-probabilities and average over non-special
    generated tokens to obtain a scalar confidence per decoded string.
    """

    try:
        import torch
    except Exception:
        return []

    sequences = getattr(generation_output, "sequences", None)
    scores = getattr(generation_output, "scores", None)
    if sequences is None or scores is None or len(scores) == 0:
        if sequences is None:
            return []
        return [0.0 for _ in range(int(sequences.shape[0]))]

    try:
        beam_indices = getattr(generation_output, "beam_indices", None)
        transition_scores = model.compute_transition_scores(
            sequences,
            scores,
            beam_indices=beam_indices,
            normalize_logits=True,
        )
        transition_scores = transition_scores.detach().float()
    except Exception:
        # Greedy fallback.  This is exact for num_beams=1 and still gives a
        # useful defensive value if a model class lacks compute_transition_scores.
        rows = int(sequences.shape[0])
        out: List[float] = []
        seq_offset = max(0, int(sequences.shape[1]) - len(scores))
        for row_idx in range(rows):
            probs_for_row: List[float] = []
            for step_idx, logits in enumerate(scores):
                logits_row = logits[row_idx].detach().float()
                token_pos = seq_offset + step_idx
                if token_pos >= int(sequences.shape[1]):
                    continue
                token_id = int(sequences[row_idx, token_pos].detach().cpu().item())
                probs = torch.softmax(logits_row, dim=-1)
                probs_for_row.append(float(probs[token_id].detach().cpu().item()))
            out.append(float(sum(probs_for_row) / len(probs_for_row)) if probs_for_row else 0.0)
        return out

    step_count = len(scores)
    if int(transition_scores.shape[1]) > step_count:
        transition_scores = transition_scores[:, -step_count:]
    seq_offset = max(0, int(sequences.shape[1]) - int(transition_scores.shape[1]))
    token_ids = sequences[:, seq_offset : seq_offset + int(transition_scores.shape[1])]

    special_ids = {
        int(v)
        for v in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(getattr(model, "config", None), "decoder_start_token_id", None),
        )
        if v is not None and int(v) >= 0
    }

    confidences: List[float] = []
    probs = torch.exp(transition_scores)
    for row_idx in range(int(probs.shape[0])):
        row_probs = probs[row_idx]
        row_token_ids = token_ids[row_idx] if row_idx < int(token_ids.shape[0]) else None
        mask = torch.isfinite(row_probs) & (row_probs > 0.0) & (row_probs <= 1.0)
        if row_token_ids is not None and special_ids:
            special_mask = torch.zeros_like(mask, dtype=torch.bool)
            for special_id in special_ids:
                special_mask |= row_token_ids == int(special_id)
            mask &= ~special_mask
        selected = row_probs[mask]
        if int(selected.numel()) == 0:
            confidences.append(0.0)
        else:
            confidences.append(float(selected.mean().detach().cpu().item()))
    return confidences


def infer_donut_texts_with_confidence(
    images: Sequence[Image.Image],
    runtime: Mapping[str, Any],
    *,
    preprocess_fn: Optional[PreprocessFn] = None,
    generate_config: Optional[Mapping[str, Any]] = None,
    max_len: int = 0,
    batch_size: int = 0,
) -> List[Tuple[str, float]]:
    """Run DONUT on a list of crops and return ``[(text, confidence), ...]``."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for DONUT TTA inference.") from exc

    if not images:
        return []
    image_processor = runtime["image_processor"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]
    gen_kwargs = build_donut_generate_kwargs(
        generate_config if generate_config is not None else runtime.get("generate_config"),
        max_len=max_len,
        force_scores=True,
    )
    effective_batch = int(batch_size) if int(batch_size or 0) > 0 else len(images)

    results: List[Tuple[str, float]] = []
    for start in range(0, len(images), effective_batch):
        batch_raw = [_ensure_pil_rgb(img) for img in images[start : start + effective_batch]]
        if preprocess_fn is not None:
            batch_images = [preprocess_fn(img).convert("RGB") for img in batch_raw]
        else:
            batch_images = batch_raw

        pixel_values = image_processor(images=batch_images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(pixel_values=pixel_values, **gen_kwargs)

        sequences = getattr(generated, "sequences", generated)
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True) if len(sequences) else []
        confidences = _generated_token_confidences(model=model, generation_output=generated, tokenizer=tokenizer)
        if len(confidences) < len(texts):
            confidences.extend([0.0] * (len(texts) - len(confidences)))
        for text, confidence in zip(texts, confidences):
            results.append((_strip_special_token_strings(str(text or ""), tokenizer), float(confidence)))
    return results


def _levenshtein_distance_fallback(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def levenshtein_distance(a: str, b: str) -> int:
    """Use an installed Levenshtein backend when available, else pure Python."""

    try:
        import Levenshtein  # type: ignore

        return int(Levenshtein.distance(a, b))
    except Exception:
        pass
    try:
        import editdistance  # type: ignore

        return int(editdistance.eval(a, b))
    except Exception:
        pass
    return _levenshtein_distance_fallback(a, b)


def cluster_tta_candidates(
    candidates: Sequence[TTACandidate],
    *,
    max_distance: int = 2,
) -> List[TTACluster]:
    """Cluster candidates by character-level Levenshtein distance."""

    clusters: List[List[TTACandidate]] = []
    threshold = max(0, int(max_distance))
    for candidate in candidates:
        placed = False
        for cluster in clusters:
            if any(levenshtein_distance(candidate.text, other.text) <= threshold for other in cluster):
                cluster.append(candidate)
                placed = True
                break
        if not placed:
            clusters.append([candidate])
    return [TTACluster(index=i, candidates=list(cluster)) for i, cluster in enumerate(clusters)]


def select_consensus_candidate(
    candidates: Sequence[TTACandidate],
    *,
    max_distance: int = 2,
) -> TTAConsensusResult:
    """Majority vote by Levenshtein cluster, then highest confidence string."""

    if not candidates:
        return TTAConsensusResult(
            text="",
            confidence=0.0,
            winner_cluster_index=-1,
            winner_cluster_size=0,
            winner_cluster_average_confidence=0.0,
            candidates=[],
            clusters=[],
        )
    clusters = cluster_tta_candidates(candidates, max_distance=max_distance)
    winner_cluster = max(clusters, key=lambda cluster: (cluster.size, cluster.average_confidence))
    winner = max(winner_cluster.candidates, key=lambda candidate: candidate.confidence)
    return TTAConsensusResult(
        text=winner.text,
        confidence=float(winner.confidence),
        winner_cluster_index=int(winner_cluster.index),
        winner_cluster_size=int(winner_cluster.size),
        winner_cluster_average_confidence=float(winner_cluster.average_confidence),
        candidates=list(candidates),
        clusters=clusters,
    )


def run_donut_tta_on_page_box(
    page_image: Image.Image | np.ndarray,
    box_xyxy: Sequence[int],
    runtime: Mapping[str, Any],
    *,
    preprocess_fn: Optional[PreprocessFn] = None,
    variations: int = 7,
    max_distance: int = 2,
    max_len: int = 0,
    batch_size: int = 0,
    generate_config: Optional[Mapping[str, Any]] = None,
) -> TTAConsensusResult:
    """Crop TTA variants, run DONUT, and return the Levenshtein consensus."""

    variant_crops = crop_tta_variants_xyxy(page_image, box_xyxy, variations=variations)
    texts_and_conf = infer_donut_texts_with_confidence(
        [crop for _variant, crop in variant_crops],
        runtime,
        preprocess_fn=preprocess_fn,
        generate_config=generate_config,
        max_len=max_len,
        batch_size=batch_size,
    )
    candidates = [
        TTACandidate(
            text=text,
            confidence=float(confidence),
            variant=variant.name,
            box_xyxy=variant.box_xyxy,
        )
        for (variant, _crop), (text, confidence) in zip(variant_crops, texts_and_conf)
    ]
    return select_consensus_candidate(candidates, max_distance=max_distance)


__all__ = [
    "BoxXYWH",
    "BoxXYXY",
    "TTABBoxVariant",
    "TTABoxVariant",
    "TTACandidate",
    "TTACluster",
    "TTAConsensusResult",
    "build_donut_generate_kwargs",
    "cluster_tta_candidates",
    "crop_tta_variants_xyxy",
    "generate_tta_box_variants_xywh",
    "generate_tta_box_variants_xyxy",
    "generate_tta_boxes_xywh",
    "infer_donut_texts_with_confidence",
    "levenshtein_distance",
    "run_donut_tta_on_page_box",
    "select_consensus_candidate",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
]

# Backward-compatible alias in case external callers use the common BBox name.
TTABBoxVariant = TTABoxVariant
