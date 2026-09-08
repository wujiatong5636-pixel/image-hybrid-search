"""基于现有 CLIP 模型的装修图片零样本分类。"""

from typing import Dict

import numpy as np


CATEGORY_PROMPTS: Dict[str, list[str]] = {
    "living_room": ["a photo of a living room with a sofa", "living room interior design"],
    "bedroom": ["a photo of a bedroom with a bed", "bedroom interior design"],
    "kitchen_sink": ["a close photo of a kitchen sink", "kitchen sink and countertop"],
    "kitchen_cabinet": ["kitchen cabinets and kitchen layout", "a photo of kitchen cabinetry"],
    "storage_cabinet": ["built-in storage cabinets", "wardrobe and storage furniture"],
    "bathroom": ["a photo of a bathroom", "bathroom interior renovation"],
    "construction": ["home renovation construction site", "interior construction process"],
    "cleaning": ["home cleaning and maintenance tutorial", "cleaning household equipment"],
    "color_design": ["interior color palette design", "home color scheme presentation"],
    "furniture": ["a product photo of furniture", "sofa chair table furniture"],
    "art": ["artwork and art appreciation", "painting sculpture and decorative art"],
    "material": ["interior material sample and texture", "tile stone wood material sample"],
}

CATEGORY_COMPATIBILITY = {
    "living_room": {"living_room", "furniture", "color_design"},
    "bedroom": {"bedroom", "furniture", "storage_cabinet", "color_design"},
    "kitchen_sink": {"kitchen_sink", "kitchen_cabinet"},
    "kitchen_cabinet": {"kitchen_cabinet", "kitchen_sink", "storage_cabinet"},
    "storage_cabinet": {"storage_cabinet", "kitchen_cabinet"},
    "bathroom": {"bathroom"},
    "construction": {"construction", "material"},
    "cleaning": {"cleaning"},
    "color_design": {"color_design", "living_room", "bedroom"},
    "furniture": {"furniture", "living_room", "bedroom"},
    "art": {"art"},
    "material": {"material", "construction"},
}


def categories_compatible(sample_category: str, candidate_category: str) -> bool:
    """判断候选类别是否允许用于当前样图。"""
    return candidate_category in CATEGORY_COMPATIBILITY.get(
        sample_category, {sample_category}
    )


class ImageCategoryClassifier:
    """复用项目 CLIP 编码器进行类别预测。"""

    def __init__(self, clip_searcher, prompts=None):
        self.clip = clip_searcher
        self.prompts = prompts or CATEGORY_PROMPTS
        self.category_vectors = self._build_category_vectors()

    @staticmethod
    def _normalize(vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-8 else vector

    def _build_category_vectors(self):
        vectors = {}
        for category, prompts in self.prompts.items():
            prompt_vectors = [self.clip.encode_text(prompt) for prompt in prompts]
            vectors[category] = self._normalize(np.mean(prompt_vectors, axis=0))
        return vectors

    def classify_embedding(self, embedding, top_n=3):
        embedding = self._normalize(embedding)
        scores = [
            {"category": category, "score": float(np.dot(embedding, vector))}
            for category, vector in self.category_vectors.items()
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)
        return {"category": scores[0]["category"], "score": scores[0]["score"], "top": scores[:top_n]}

    def classify_image(self, image_path, top_n=3):
        return self.classify_embedding(self.clip.encode_image(image_path), top_n=top_n)
