from pechabridge.ocr.donut_tta import (
    TTACandidate,
    generate_tta_box_variants_xywh,
    generate_tta_boxes_xywh,
    levenshtein_distance,
    select_consensus_candidate,
)


def test_generate_tta_boxes_xywh_default_variants_are_clipped() -> None:
    boxes = generate_tta_boxes_xywh((10, 20, 30, 10), image_size=(100, 50), variations=7)

    assert boxes[0] == (10, 20, 30, 10)
    assert len(boxes) == 7
    for x, y, w, h in boxes:
        assert x >= 0
        assert y >= 0
        assert w > 0
        assert h > 0
        assert x + w <= 100
        assert y + h <= 50


def test_generate_tta_box_variants_remove_boundary_duplicates() -> None:
    variants = generate_tta_box_variants_xywh((0, 0, 20, 10), image_size=(25, 12), variations=9)

    assert variants[0].name == "original"
    assert len({variant.box_xyxy for variant in variants}) == len(variants)


def test_levenshtein_distance_fallback_contract() -> None:
    assert levenshtein_distance("pecha", "pecha") == 0
    assert levenshtein_distance("pecha", "pech") == 1
    assert levenshtein_distance("pecha", "pacha") == 1


def test_consensus_prefers_majority_cluster_then_best_candidate_confidence() -> None:
    candidates = [
        TTACandidate("abc", 0.60, "original", (0, 0, 10, 10)),
        TTACandidate("abd", 0.55, "expand", (0, 0, 10, 11)),
        TTACandidate("xyz", 0.99, "shift", (0, 1, 10, 11)),
    ]

    result = select_consensus_candidate(candidates, max_distance=1)

    assert result.text == "abc"
    assert result.confidence == 0.60
    assert result.winner_cluster_size == 2


def test_consensus_breaks_cluster_size_tie_by_average_confidence() -> None:
    candidates = [
        TTACandidate("abc", 0.10, "a", (0, 0, 10, 10)),
        TTACandidate("abd", 0.30, "b", (0, 0, 10, 11)),
        TTACandidate("wxyz", 0.80, "c", (0, 1, 10, 11)),
        TTACandidate("wxyy", 0.90, "d", (0, 1, 10, 12)),
    ]

    result = select_consensus_candidate(candidates, max_distance=1)

    assert result.text == "wxyy"
    assert result.winner_cluster_size == 2
