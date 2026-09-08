"""
A、B、C三分组基础选择器。

第一阶段根据以下信息进行基础选择：
1. CLIP语义相似度；
2. 样图与候选图的pHash距离；
3. 候选之间是否重复；
4. 候选是否已经被其他分组占用。

注意：
第一阶段只能完成基础分组。
构图、造型、材质和色彩的独立评分在第二阶段实现。
"""

from typing import Dict, List, Optional, Set


class BasicGroupSelector:
    """从候选结果中选择A、B、C三张图片。"""

    def __init__(
        self,
        duplicate_phash_threshold: int = 6,
        minimum_semantic_score: float = 0.0,
    ):
        if duplicate_phash_threshold < 0:
            raise ValueError("重复图片pHash阈值不能小于0")

        self.duplicate_phash_threshold = (
            duplicate_phash_threshold
        )
        self.minimum_semantic_score = minimum_semantic_score

    @staticmethod
    def _candidate_id(candidate: dict) -> str:
        """获取候选图片的稳定身份。"""
        return str(candidate["path"])

    def _is_available(
        self,
        candidate: dict,
        used_ids: Set[str],
    ) -> bool:
        """检查候选图片是否可以参加选择。"""
        candidate_id = self._candidate_id(candidate)

        if candidate_id in used_ids:
            return False

        if candidate.get("is_duplicate", False):
            return False

        if (
            candidate.get("phash_distance", 0)
            <= self.duplicate_phash_threshold
        ):
            return False

        if (
            candidate.get("semantic_score", 0.0)
            < self.minimum_semantic_score
        ):
            return False

        return True

    def _pick_first_available(
        self,
        candidates: List[dict],
        used_ids: Set[str],
    ) -> Optional[dict]:
        """从候选列表中选择第一个可用结果。"""
        for candidate in candidates:
            if self._is_available(candidate, used_ids):
                return candidate

        return None

    @staticmethod
    def _with_group(
        candidate: Optional[dict],
        group: str,
        description: str,
    ) -> Optional[dict]:
        """给选中的候选增加分组信息。"""
        if candidate is None:
            return None

        result = dict(candidate)
        result["group"] = group
        result["group_description"] = description
        return result

    def select(
        self,
        candidates: List[dict],
        used_candidate_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Optional[dict]]:
        """
        选择A、B、C三组结果。

        候选格式：
        {
            "path": "候选图片路径",
            "semantic_score": 0.85,
            "phash_distance": 20,
            "is_duplicate": False
        }
        """
        used_ids = set(used_candidate_ids or set())

        valid_candidates = [
            dict(candidate)
            for candidate in candidates
            if self._is_available(candidate, used_ids)
        ]

        # CLIP语义相似度从高到低排列
        valid_candidates.sort(
            key=lambda item: item.get(
                "semantic_score",
                0.0,
            ),
            reverse=True,
        )

        if not valid_candidates:
            return {
                "A": None,
                "B": None,
                "C": None,
            }

        total = len(valid_candidates)

        # A 选择语义相似度最高的有效候选。
        group_a = valid_candidates[0]

        # B 选择排序后位于中间位置的候选。
        if total >= 2:
            middle_index = total // 2
            group_b = valid_candidates[middle_index]
        else:
            group_b = None

        # C 选择排序末尾的候选，确保与 A、B 使用不同图片。
        if total >= 3:
            group_c = valid_candidates[-1]
        else:
            group_c = None

        return {
            "A": self._with_group(
                group_a,
                "A",
                "高语义相似、排除重复",
            ),
            "B": self._with_group(
                group_b,
                "B",
                "中等语义相似、与A不同",
            ),
            "C": self._with_group(
                group_c,
                "C",
                "保持主题相关、视觉差异更大",
            ),
        }
