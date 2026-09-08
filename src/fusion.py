"""
融合排序模块
支持两种融合策略：
  1. 加权求和（weighted）：对各路分数归一化后加权求和，适合分数质量稳定的场景
  2. RRF 倒数排名融合（rrf）：基于排名而非绝对分值，适合不同候选来源

统一输入格式：
  每路结果为 [(doc_id, score), ...]，按分数降序排列。
  对于无分值的候选列表，score可传0，排名由位置决定。
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# 统一结果类型：(doc_id, score)
SearchResult = List[Tuple[str, float]]


class FusionEngine:
    """多路检索结果融合引擎。"""

    def __init__(
        self,
        strategy: str = "weighted",
        weights: Dict[str, float] = None,
        rrf_k: int = 60,
        final_top_k: int = 10,
    ):
        """
        Args:
            strategy: "weighted" 或 "rrf"
            weights: 各候选通道权重，例如{"clip_image": 1.0}
            rrf_k: RRF 常数 k，通常取 60
            final_top_k: 最终返回数量
        """
        self.strategy = strategy.lower()
        self.weights = weights or {"clip_image": 1.0}
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k

        if self.strategy not in ("weighted", "rrf"):
            raise ValueError(f"不支持的融合策略: {strategy}")

        # 权重归一化
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info("融合引擎初始化 | strategy=%s | weights=%s | rrf_k=%d",
                     self.strategy, self.weights, self.rrf_k)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def fuse(self, results: Dict[str, SearchResult]) -> SearchResult:
        """
        融合多路检索结果。

        Args:
            results: {"通道名": [(doc_id, score), ...], ...}
                     通道名需与 weights 中的 key 对应

        Returns:
            [(doc_id, fused_score), ...] 按融合分数降序，最多 final_top_k 条
        """
        if not results:
            return []

        # 过滤掉空结果通道
        results = {k: v for k, v in results.items() if v}
        if not results:
            return []

        if self.strategy == "weighted":
            fused = self._weighted_sum(results)
        else:
            fused = self._rrf(results)

        # 按分数降序排列，取 top_k
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[: self.final_top_k]

    # ------------------------------------------------------------------
    # 策略1：加权求和
    # ------------------------------------------------------------------

    def _weighted_sum(self, results: Dict[str, SearchResult]) -> SearchResult:
        """
        加权求和融合。
        对每路分数做 min-max 归一化到 [0, 1]，然后按权重加权。
        某 doc 只在部分通道出现时，缺失通道的分数计为 0。
        """
        # 收集所有 doc_id
        all_docs = set()
        for channel_results in results.values():
            for doc_id, _ in channel_results:
                all_docs.add(doc_id)

        # 对每路做 min-max 归一化
        normalized = {}
        for channel, channel_results in results.items():
            if not channel_results:
                continue
            scores = [s for _, s in channel_results]
            min_s = min(scores)
            max_s = max(scores)
            score_range = max_s - min_s

            norm_map = {}
            for doc_id, score in channel_results:
                if score_range > 1e-8:
                    norm_map[doc_id] = (score - min_s) / score_range
                else:
                    norm_map[doc_id] = 1.0  # 所有分数相同时，归一化为 1
            normalized[channel] = norm_map

        # 加权求和
        fused = []
        for doc_id in all_docs:
            total_score = 0.0
            for channel, weight in self.weights.items():
                channel_score = normalized.get(channel, {}).get(doc_id, 0.0)
                total_score += weight * channel_score
            fused.append((doc_id, total_score))

        return fused

    # ------------------------------------------------------------------
    # 策略2：RRF 倒数排名融合
    # ------------------------------------------------------------------

    def _rrf(self, results: Dict[str, SearchResult]) -> SearchResult:
        """
        RRF (Reciprocal Rank Fusion) 倒数排名融合。
        RRF_score(d) = Σ 1 / (k + rank_i(d))

        优势：
          - 无需归一化分数
          - 对绝对分值不敏感
          - 适合分值量纲不同的候选来源
        """
        # 收集所有 doc_id
        all_docs = set()
        for channel_results in results.values():
            for doc_id, _ in channel_results:
                all_docs.add(doc_id)

        # 构建每路的排名映射：doc_id -> rank (从 1 开始)
        rank_maps = {}
        for channel, channel_results in results.items():
            rank_map = {}
            for rank, (doc_id, _) in enumerate(channel_results, start=1):
                rank_map[doc_id] = rank
            rank_maps[channel] = rank_map

        # 计算 RRF 分数
        fused = []
        for doc_id in all_docs:
            rrf_score = 0.0
            for channel in results.keys():
                rank = rank_maps[channel].get(doc_id)
                if rank is not None:
                    rrf_score += 1.0 / (self.rrf_k + rank)
            fused.append((doc_id, rrf_score))

        return fused

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def build_result_from_urls(urls: List[str]) -> SearchResult:
        """
        将无分值的 URL 列表转换为 SearchResult 格式。
        排名越靠前，伪分数越高（用 1/rank 作为伪分数，仅用于加权模式）。

        Args:
            urls: URL 列表，按相关度降序

        Returns:
            [(url, pseudo_score), ...]
        """
        return [(url, 1.0 / (i + 1)) for i, url in enumerate(urls)]
