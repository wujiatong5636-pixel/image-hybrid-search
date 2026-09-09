"""
本地CLIP+FAISS检索提供器。
封装现有的 CLIPSearcher 和 VectorStore，输出统一的 Candidate 格式。
"""
import logging
from pathlib import Path
from typing import List

from .base import Candidate, BaseProvider

logger = logging.getLogger(__name__)


class LocalFAISSProvider(BaseProvider):
    """本地CLIP+FAISS检索。"""

    def __init__(self, clip_searcher, vector_store):
        """
        Args:
            clip_searcher: 已初始化的 CLIPSearcher 实例
            vector_store: 已加载的 VectorStore 实例
        """
        self.clip = clip_searcher
        self.vector_store = vector_store

    def search(self, sample_path: str, top_k: int = 98) -> List[Candidate]:
        """
        执行本地FAISS检索。

        Args:
            sample_path: 样图路径
            top_k: 召回数量

        Returns:
            Candidate 列表（原始召回，未过滤）
        """
        sample_path = str(Path(sample_path).resolve())

        # CLIP编码样图
        query_vector = self.clip.encode_image(sample_path)

        # FAISS检索
        raw_results = self.vector_store.search(
            query_vector,
            top_k=top_k,
            threshold=0.0,
        )

        candidates = []
        for rank, (path, score, internal_id) in enumerate(raw_results, start=1):
            candidate = Candidate(
                source="local",
                source_id=str(Path(path).resolve()),
                source_url=None,
                local_path=str(Path(path).resolve()),
                source_rank=rank,
                source_score=float(score),
                match_type="local",
            )
            # 附带内部ID，方便后续重建向量
            candidate.internal_id = internal_id
            candidates.append(candidate)

        logger.info("本地FAISS召回 %d 条候选", len(candidates))
        return candidates

    @property
    def is_available(self) -> bool:
        return len(self.vector_store) > 0
