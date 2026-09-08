"""类别分类排序测试。"""

import numpy as np

from src.image_classifier import ImageCategoryClassifier


def test_classify_embedding_returns_highest_category():
    classifier = object.__new__(ImageCategoryClassifier)
    classifier.category_vectors = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
    }
    result = classifier.classify_embedding(np.array([0.9, 0.1], dtype=np.float32), top_n=2)
    assert result["category"] == "a"
    assert [item["category"] for item in result["top"]] == ["a", "b"]
