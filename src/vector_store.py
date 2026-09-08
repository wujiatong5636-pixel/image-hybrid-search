"""
FAISS 向量存储模块
支持两种索引：
  - IndexFlatIP：精确内积搜索，适合 < 5 万张，查询最快
  - IndexIVFFlat：倒排聚类加速，适合 5 万+ 规模
所有向量在添加前必须 L2 归一化，此时内积 = 余弦相似度。
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS 向量索引封装，支持增删查、持久化和 ID↔路径映射。"""
    def __init__(
        self,
        dim: int = 512,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        index_path: str = "data/faiss_index.bin",
        mapping_path: str = "data/id_to_path.json",
    ):
        """
        Args:
            dim: 向量维度
            index_type: "flat" 或 "ivf"
            nlist: IVF 聚类中心数（仅 ivf 模式）
            nprobe: IVF 查询探测聚类数（仅 ivf 模式）
            index_path: 索引文件持久化路径
            mapping_path: ID→路径映射文件路径
        """
        self.dim = dim
        self.index_type = index_type.lower()
        self.nlist = nlist
        self.nprobe = nprobe
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        # id_to_path: {int_id: "path/to/image.jpg"}
        # path_to_id: {"path/to/image.jpg": int_id}
        self.id_to_path: Dict[int, str] = {}
        self.path_to_id: Dict[str, int] = {}
        self._next_id = 0
        self.index = self._build_index()
        self._trained = (self.index_type == "flat")  # flat 无需训练

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------
    def _build_index(self) -> faiss.Index:
        """构建 FAISS 索引。"""
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dim)
            logger.info("构建 IndexFlatIP (dim=%d)", self.dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = self.nprobe
            logger.info("构建 IndexIVFFlat (dim=%d, nlist=%d, nprobe=%d)",
                         self.dim, self.nlist, self.nprobe)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        return index

    def train(self, embeddings: np.ndarray):
        """
        训练 IVF 索引（flat 模式无需调用）。
        必须在 add 之前调用，使用一批代表性向量。
        Args:
            embeddings: shape=(N, dim)，已 L2 归一化
        """
        if self.index_type == "flat":
            logger.info("flat 索引无需训练，跳过")
            return
        if self._trained:
            logger.warning("索引已训练，跳过重复训练")
            return
        if len(embeddings) < self.nlist:
            logger.warning("训练样本数 (%d) 少于 nlist (%d)，可能影响效果",
                           len(embeddings), self.nlist)
        self.index.train(embeddings.astype(np.float32))
        self._trained = True
        logger.info("IVF 索引训练完成，样本数=%d", len(embeddings))

    # ------------------------------------------------------------------
    # 增删操作
    # ------------------------------------------------------------------
    def add(self, embeddings: np.ndarray, paths: List[str]) -> List[int]:
        """
        添加向量到索引，自动跳过已存在路径。
        Args:
            embeddings: shape=(N, dim)，已 L2 归一化
            paths: 对应图片路径列表
        Returns:
            分配的内部 ID 列表
        """
        if not self._trained:
            raise RuntimeError("IVF 索引尚未训练，请先调用 train()")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配: 期望 {self.dim}, 实际 {embeddings.shape[1]}")
        if len(embeddings) != len(paths):
            raise ValueError("向量数量与路径数量不一致")

        # 过滤重复路径，只新增未存在的
        new_embeds = []
        new_paths = []
        ids_out = []
        for vec, p in zip(embeddings, paths):
            if p in self.path_to_id:
                ids_out.append(self.path_to_id[p])
                continue
            idx = self._next_id
            self.id_to_path[idx] = p
            self.path_to_id[p] = idx
            ids_out.append(idx)
            new_embeds.append(vec)
            new_paths.append(p)
            self._next_id += 1

        if len(new_embeds) > 0:
            arr = np.stack(new_embeds, axis=0)
            self.index.add(arr)
            logger.info("新增 %d 条向量，索引总量=%d", len(new_embeds), self.index.ntotal)
        else:
            logger.info("所有传入路径均已存在，无新增向量")

        # ✅ 关键一致性检查（修复点）
        if self.index.ntotal != len(self.id_to_path):
            raise RuntimeError(
                f"索引与路径映射不一致：index.ntotal={self.index.ntotal}, "
                f"映射长度={len(self.id_to_path)}"
            )
        return ids_out

    def add_one(self, embedding: np.ndarray, path: str) -> int:
        """添加单条向量。"""
        return self.add(embedding.reshape(1, -1), [path])[0]

    def __len__(self) -> int:
        return self.index.ntotal

    def reset(self):
        """清空全部索引与映射，重建空索引"""
        self.index.reset()
        self.id_to_path.clear()
        self.path_to_id.clear()
        self._next_id = 0
        self._trained = (self.index_type == "flat")
        logger.info("索引已重置清空")

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, int]]:
        """
        相似度查询。
        Args:
            query_vec: shape=(dim,) 查询向量，已 L2 归一化
            top_k: 返回数量
            threshold: 相似度阈值（低于此值的结果被过滤）
        Returns:
            [(path, similarity_score, internal_id), ...] 按相似度降序
        """
        if self.index.ntotal == 0:
            return []
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score < threshold:
                continue
            path = self.id_to_path.get(int(idx), f"unknown_{idx}")
            results.append((path, float(score), int(idx)))
        return results

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self):
        """保存索引和映射到磁盘。"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id_to_path": {str(k): v for k, v in self.id_to_path.items()},
                    "next_id": self._next_id,
                    "index_type": self.index_type,
                    "dim": self.dim,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("索引已保存: %s (%d 条)", self.index_path, self.index.ntotal)

    def load(self) -> bool:
        """
        从磁盘加载索引和映射。
        Returns:
            是否加载成功
        """
        if not self.index_path.exists() or not self.mapping_path.exists():
            logger.info("未找到已保存的索引，将创建新索引")
            return False
        self.index = faiss.read_index(str(self.index_path))
        if self.index_type == "ivf":
            self.index.nprobe = self.nprobe
            self._trained = True
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.id_to_path = {int(k): v for k, v in data.get("id_to_path", {}).items()}
        self.path_to_id = {v: k for k, v in self.id_to_path.items()}
        self._next_id = data.get("next_id", len(self.id_to_path))
        # ✅ 一致性检查（加载时也检查，若不一致则抛出异常）
        if self.index.ntotal != len(self.id_to_path):
            raise RuntimeError(
                f"加载后索引与映射不一致：index.ntotal={self.index.ntotal}, "
                f"映射长度={len(self.id_to_path)}"
            )
        logger.info("索引已加载: %s (%d 条)", self.index_path, self.index.ntotal)
        return True