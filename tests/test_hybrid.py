"""
单元测试：融合引擎和限流器（不依赖外部模型/API，可直接运行）
运行: python -m pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion import FusionEngine
from src.rate_limiter import RateLimiter
from src.vector_store import VectorStore


# ============================================================
# FusionEngine 测试
# ============================================================

class TestFusionEngine:
    def test_weighted_sum_basic(self):
        """加权求和：两路结果有交集时，交集项分数更高。"""
        fusion = FusionEngine(strategy="weighted",
                              weights={"a": 0.5, "b": 0.5},
                              final_top_k=10)
        results = {
            "a": [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)],
            "b": [("doc2", 0.95), ("doc4", 0.6)],
        }
        fused = fusion.fuse(results)
        ids = [doc_id for doc_id, _ in fused]
        # doc2 在两路都出现且排名靠前，应排第一
        assert ids[0] == "doc2"
        assert len(fused) == 4  # 去重后 4 个

    def test_rrf_basic(self):
        """RRF：排名越靠前贡献越大。"""
        fusion = FusionEngine(strategy="rrf", rrf_k=60, final_top_k=10)
        results = {
            "a": [("doc1", 0), ("doc2", 0), ("doc3", 0)],
            "b": [("doc2", 0), ("doc1", 0)],
        }
        fused = fusion.fuse(results)
        ids = [doc_id for doc_id, _ in fused]
        # doc1: rank1 in a, rank2 in b → 1/61 + 1/62
        # doc2: rank2 in a, rank1 in b → 1/62 + 1/61
        # 两者分数相同，doc3 只在 a 中排第三
        assert set(ids[:2]) == {"doc1", "doc2"}

    def test_empty_results(self):
        """空输入返回空列表。"""
        fusion = FusionEngine()
        assert fusion.fuse({}) == []
        assert fusion.fuse({"a": []}) == []

    def test_top_k_truncation(self):
        """结果数量不超过 final_top_k。"""
        fusion = FusionEngine(final_top_k=2)
        results = {"a": [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]}
        fused = fusion.fuse(results)
        assert len(fused) == 2

    def test_build_result_from_urls(self):
        """URL 列表转 SearchResult 格式。"""
        urls = ["http://a.com", "http://b.com", "http://c.com"]
        result = FusionEngine.build_result_from_urls(urls)
        assert len(result) == 3
        assert result[0][1] > result[1][1] > result[2][1]  # 伪分数递减

    def test_weights_normalized(self):
        """权重自动归一化。"""
        fusion = FusionEngine(weights={"a": 2, "b": 2})
        assert abs(fusion.weights["a"] - 0.5) < 1e-6
        assert abs(fusion.weights["b"] - 0.5) < 1e-6


# ============================================================
# RateLimiter 测试
# ============================================================

class TestRateLimiter:
    def test_allow_within_quota(self):
        """配额内允许调用。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rl = RateLimiter(max_per_month=100, alert_threshold=0.8,
                             usage_path=str(Path(tmpdir) / "usage.json"))
            for _ in range(79):
                assert rl.allow_vision_call() is True
            assert rl.used == 79

    def test_degrade_at_threshold(self):
        """达到阈值后拒绝调用。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rl = RateLimiter(max_per_month=100, alert_threshold=0.8,
                             usage_path=str(Path(tmpdir) / "usage.json"))
            for _ in range(80):
                rl.allow_vision_call()
            # 第 81 次应被拒绝（80/100 = 0.8 达到阈值）
            assert rl.allow_vision_call() is False

    def test_persistence(self):
        """用量持久化后可恢复。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "usage.json")
            rl1 = RateLimiter(max_per_month=100, usage_path=path)
            for _ in range(10):
                rl1.allow_vision_call()

            rl2 = RateLimiter(max_per_month=100, usage_path=path)
            assert rl2.used == 10

    def test_manual_reset(self):
        """手动重置。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rl = RateLimiter(max_per_month=100, usage_path=str(Path(tmpdir) / "u.json"))
            for _ in range(50):
                rl.allow_vision_call()
            rl.reset()
            assert rl.used == 0

    def test_get_usage(self):
        """用量信息正确。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rl = RateLimiter(max_per_month=100, alert_threshold=0.8,
                             usage_path=str(Path(tmpdir) / "u.json"))
            for _ in range(50):
                rl.allow_vision_call()
            info = rl.get_usage()
            assert info["used"] == 50
            assert info["remaining"] == 50
            assert abs(info["usage_rate"] - 0.5) < 1e-6
            assert info["degraded"] is False


# ============================================================
# VectorStore 测试
# ============================================================

class TestVectorStore:
    def test_add_and_search_flat(self):
        """Flat 索引：添加后可检索到最相似的向量。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(dim=8, index_type="flat",
                             index_path=str(Path(tmpdir) / "idx.bin"),
                             mapping_path=str(Path(tmpdir) / "map.json"))
            # 创建归一化向量
            v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            vs.add(np.stack([v1, v2]), ["img1.jpg", "img2.jpg"])

            # 用 v1 查询，应返回 img1.jpg 排第一
            results = vs.search(v1, top_k=2)
            assert results[0][0] == "img1.jpg"
            assert abs(results[0][1] - 1.0) < 1e-6  # 自身相似度=1

    def test_save_and_load(self):
        """索引持久化和加载。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = str(Path(tmpdir) / "idx.bin")
            map_path = str(Path(tmpdir) / "map.json")

            vs1 = VectorStore(dim=8, index_path=idx_path, mapping_path=map_path)
            v = np.random.randn(5, 8).astype(np.float32)
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            v = v / norms
            vs1.add(v, [f"img{i}.jpg" for i in range(5)])
            vs1.save()

            vs2 = VectorStore(dim=8, index_path=idx_path, mapping_path=map_path)
            assert vs2.load() is True
            assert len(vs2) == 5
            assert vs2.id_to_path[0] == "img0.jpg"

    def test_empty_search(self):
        """空索引搜索返回空列表。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(dim=8, index_path=str(Path(tmpdir) / "i.bin"),
                             mapping_path=str(Path(tmpdir) / "m.json"))
            q = np.ones(8, dtype=np.float32) / np.sqrt(8)
            assert vs.search(q) == []

    def test_ivf_train_and_search(self):
        """IVF 索引：训练后可检索。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(dim=8, index_type="ivf", nlist=3, nprobe=3,
                             index_path=str(Path(tmpdir) / "idx.bin"),
                             mapping_path=str(Path(tmpdir) / "map.json"))
            # 生成足够的训练数据
            np.random.seed(42)
            v = np.random.randn(20, 8).astype(np.float32)
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            v = v / norms

            vs.train(v)
            vs.add(v, [f"img{i}.jpg" for i in range(20)])

            q = v[0]
            results = vs.search(q, top_k=3)
            assert len(results) > 0
            assert results[0][0] == "img0.jpg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
