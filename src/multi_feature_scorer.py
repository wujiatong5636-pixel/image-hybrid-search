"""轻量多维图片特征与 A/B/C 独立评分。"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


class MultiFeatureScorer:
    """计算构图、造型、色彩、纹理特征并进行三分组选择。"""

    def __init__(self):
        self._descriptor_cache = {}

    @staticmethod
    def _cosine(a, b):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= 1e-8:
            return 1.0 if np.allclose(a, b) else 0.0
        return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))

    def _descriptors(self, image_path):
        key = str(Path(image_path).resolve())
        if key in self._descriptor_cache:
            return self._descriptor_cache[key]
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            gray = rgb.convert("L")

            layout = np.asarray(gray.resize((12, 12)), dtype=np.float32) / 255.0
            layout = layout.reshape(-1) - float(layout.mean())

            edges = gray.resize((64, 64)).filter(ImageFilter.FIND_EDGES)
            shape = np.asarray(edges, dtype=np.float32).reshape(-1) / 255.0

            hsv = np.asarray(rgb.resize((128, 128)).convert("HSV"), dtype=np.uint8)
            color_parts = []
            for channel, bins in ((0, 18), (1, 8), (2, 8)):
                hist, _ = np.histogram(hsv[:, :, channel], bins=bins, range=(0, 256))
                hist = hist.astype(np.float32)
                hist /= max(float(hist.sum()), 1.0)
                color_parts.append(hist)
            color = np.concatenate(color_parts)

            texture_hist, _ = np.histogram(
                np.asarray(edges, dtype=np.uint8), bins=32, range=(0, 256)
            )
            texture = texture_hist.astype(np.float32)
            texture /= max(float(texture.sum()), 1.0)

        value = {"layout": layout, "shape": shape, "color": color, "texture": texture}
        self._descriptor_cache[key] = value
        return value

    def compare(self, sample_path, candidate_path):
        sample = self._descriptors(sample_path)
        candidate = self._descriptors(candidate_path)
        composition = (self._cosine(sample["layout"], candidate["layout"]) + 1.0) / 2.0
        return {
            "composition_similarity": float(np.clip(composition, 0.0, 1.0)),
            "shape_similarity": float(np.clip(self._cosine(sample["shape"], candidate["shape"]), 0.0, 1.0)),
            "color_similarity": float(np.clip(self._cosine(sample["color"], candidate["color"]), 0.0, 1.0)),
            "texture_similarity": float(np.clip(self._cosine(sample["texture"], candidate["texture"]), 0.0, 1.0)),
        }

    @staticmethod
    def _group_scores(item):
        semantic = float(np.clip(item["semantic_score"], 0.0, 1.0))
        composition = item["composition_similarity"]
        shape = item["shape_similarity"]
        color_difference = 1.0 - item["color_similarity"]
        texture_difference = 1.0 - item["texture_similarity"]
        return {
            "A": 0.35 * semantic + 0.30 * composition + 0.20 * shape + 0.075 * color_difference + 0.075 * texture_difference,
            "B": 0.50 * semantic + 0.20 * (1.0 - composition) + 0.15 * (1.0 - shape) + 0.075 * color_difference + 0.075 * texture_difference,
            "C": 0.65 * semantic + 0.10 * (1.0 - composition) + 0.10 * (1.0 - shape) + 0.075 * color_difference + 0.075 * texture_difference,
        }

    def select(self, sample_path, candidates):
        enriched = []
        for candidate in candidates:
            item = dict(candidate)
            item.update(self.compare(sample_path, item["path"]))
            item["group_scores"] = self._group_scores(item)
            enriched.append(item)

        selected = {}
        used_paths = set()
        for group in "ABC":
            available = [item for item in enriched if item["path"] not in used_paths]
            if not available:
                selected[group] = None
                continue
            best = max(available, key=lambda item: item["group_scores"][group])
            result = dict(best)
            result["group"] = group
            result["multi_feature_score"] = best["group_scores"][group]
            selected[group] = result
            used_paths.add(best["path"])
        return selected
