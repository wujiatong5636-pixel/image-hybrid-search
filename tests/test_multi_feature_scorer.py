"""多维评分测试。"""

from pathlib import Path

from PIL import Image

from src.multi_feature_scorer import MultiFeatureScorer


def test_identical_image_features_are_high(tmp_path):
    path = tmp_path / "same.png"
    Image.new("RGB", (100, 100), (120, 80, 40)).save(path)
    result = MultiFeatureScorer().compare(path, path)
    assert result["composition_similarity"] > 0.99
    assert result["shape_similarity"] > 0.99
    assert result["color_similarity"] > 0.99
    assert result["texture_similarity"] > 0.99


def test_select_returns_three_unique_candidates(tmp_path):
    sample = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), "white").save(sample)
    candidates = []
    for index, color in enumerate(("red", "green", "blue")):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (100, 100), color).save(path)
        candidates.append({"path": str(path), "semantic_score": 0.8})
    result = MultiFeatureScorer().select(sample, candidates)
    assert set(result) == set("ABC")
    assert len({result[group]["path"] for group in "ABC"}) == 3
