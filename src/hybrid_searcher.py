"""
HybridSearcher 主入口
整合 CLIP 语义检索、Google Vision 全网溯源、FAISS 向量存储和融合排序，
提供统一的以图搜图 / 以文搜图接口。

核心流程（以图搜图）：
  1. CLIP 编码查询图 → FAISS 检索本地库 → clip_image 结果
  2. Google Vision Web Detection → 全网匹配 URL → vision 结果
  3. 融合引擎合并两路 → 最终排序结果

降级策略：
  - Vision 不可用 / 限流 / 密钥缺失 → 自动降级为纯 CLIP 模式
  - CLIP 不可用 → 仅 Vision（无本地语义检索）
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .clip_searcher import CLIPSearcher
from .fusion import FusionEngine, SearchResult
from .rate_limiter import RateLimiter
from .vector_store import VectorStore
from .vision_searcher import VisionWebDetector

logger = logging.getLogger(__name__)


class HybridSearcher:
    """混合图片检索主类。"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: YAML 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.config_path = config_path

        # 初始化各模块
        self.clip = self._init_clip()
        self.vector_store = self._init_vector_store()
        self.vision = self._init_vision()
        self.fusion = self._init_fusion()
        self.rate_limiter = self._init_rate_limiter()

        logger.info("HybridSearcher 初始化完成 | Vision可用=%s | 索引量=%d",
                     self.vision.is_available if self.vision else False,
                     len(self.vector_store))

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_clip(self) -> CLIPSearcher:
        cfg = self.config["clip"]
        return CLIPSearcher(
            backend=cfg.get("backend", "onnx"),
            onnx_model=cfg.get("onnx_model", "openai/clip-vit-base-patch32"),
            openclip_model=cfg.get("openclip_model", "ViT-B-32"),
            openclip_pretrained=cfg.get("openclip_pretrained", "laion2b_s34b_b79k"),
            embedding_dim=cfg.get("embedding_dim", 512),
            device=cfg.get("device", "cpu"),
        )

    def _init_vector_store(self) -> VectorStore:
        cfg = self.config["vector_store"]
        vs = VectorStore(
            dim=self.config["clip"].get("embedding_dim", 512),
            index_type=cfg.get("index_type", "flat"),
            nlist=cfg.get("nlist", 100),
            nprobe=cfg.get("nprobe", 10),
            index_path=cfg.get("index_path", "data/faiss_index.bin"),
            mapping_path=cfg.get("mapping_path", "data/id_to_path.json"),
        )
        vs.load()
        return vs

    def _init_vision(self) -> Optional[VisionWebDetector]:
        cfg = self.config.get("google_vision", {})
        if not cfg.get("enabled", True):
            logger.info("Google Vision 已在配置中禁用")
            return None
        try:
            return VisionWebDetector(
                credentials_path=cfg.get("credentials_path", "credentials/google_vision_key.json"),
                proxy=cfg.get("proxy"),
                timeout=cfg.get("timeout", 30),
            )
        except Exception as e:
            logger.warning("Vision 初始化失败，将以纯 CLIP 模式运行: %s", e)
            return None

    def _init_fusion(self) -> FusionEngine:
        cfg = self.config["fusion"]
        return FusionEngine(
            strategy=cfg.get("strategy", "weighted"),
            weights=cfg.get("weights", {"vision": 0.3, "clip_image": 0.5, "clip_text": 0.2}),
            rrf_k=cfg.get("rrf_k", 60),
            final_top_k=cfg.get("final_top_k", 10),
        )

    def _init_rate_limiter(self) -> RateLimiter:
        cfg = self.config["rate_limiter"]
        return RateLimiter(
            max_per_month=cfg.get("max_per_month", 1000),
            alert_threshold=cfg.get("alert_threshold", 0.8),
            usage_path=cfg.get("usage_path", "data/vision_usage.json"),
        )

    # ------------------------------------------------------------------
    # 核心检索接口
    # ------------------------------------------------------------------

    def search_by_image(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        use_vision: bool = True,
    ) -> Dict:
        """
        以图搜图：CLIP 本地语义检索 + Google Vision 全网溯源，融合输出。

        Args:
            image_path: 查询图片路径
            top_k: 返回数量，默认取配置中的 final_top_k
            use_vision: 是否启用 Vision（False 时强制纯 CLIP）

        Returns:
            {
                "query": "查询图片路径",
                "mode": "hybrid" | "clip_only",
                "results": [
                    {"id": "doc_id(路径或URL)", "score": 融合分数, "source": "clip_image"|"vision"|"both"},
                    ...
                ],
                "clip_results": [(path, score), ...],  # 原始 CLIP 结果
                "vision_result": {...} | None,          # 原始 Vision 结果
            }
        """
        top_k = top_k or self.config["fusion"].get("final_top_k", 10)
        self.fusion.final_top_k = top_k

        # --- 1. CLIP 图像检索 ---
        clip_results = self._clip_image_search(image_path, top_k)

        # --- 2. Google Vision 全网溯源 ---
        vision_result = None
        vision_results: SearchResult = []
        mode = "clip_only"

        if use_vision and self.vision and self.vision.is_available:
            if self.rate_limiter.allow_vision_call():
                vision_result = self.vision.detect(image_path)
                vision_urls = self.vision.get_all_image_urls(vision_result)
                vision_results = FusionEngine.build_result_from_urls(vision_urls)
                mode = "hybrid"
            else:
                logger.info("Vision 已限流，本次使用纯 CLIP 模式")
        elif use_vision and not self.vision:
            logger.info("Vision 不可用，使用纯 CLIP 模式")

        # --- 3. 融合 ---
        all_results = {"clip_image": clip_results}
        if vision_results:
            all_results["vision"] = vision_results

        fused = self.fusion.fuse(all_results)

        # --- 4. 标注来源 ---
        clip_ids = {doc_id for doc_id, _ in clip_results}
        vision_ids = {doc_id for doc_id, _ in vision_results}

        annotated = []
        for doc_id, score in fused:
            if doc_id in clip_ids and doc_id in vision_ids:
                source = "both"
            elif doc_id in vision_ids:
                source = "vision"
            else:
                source = "clip_image"
            annotated.append({"id": doc_id, "score": round(score, 6), "source": source})

        return {
            "query": image_path,
            "mode": mode,
            "results": annotated,
            "clip_results": [(p, round(s, 6)) for p, s in clip_results],
            "vision_result": vision_result,
        }

    def search_by_text(
        self,
        text: str,
        top_k: Optional[int] = None,
    ) -> Dict:
        """
        以文搜图：仅使用 CLIP 文本编码在本地库中检索。
        （Vision 不支持文本检索，故无 Vision 通道）

        Args:
            text: 查询文本
            top_k: 返回数量

        Returns:
            {
                "query": "文本",
                "mode": "clip_text",
                "results": [{"id": path, "score": 相似度, "source": "clip_text"}, ...],
            }
        """
        top_k = top_k or self.config["fusion"].get("final_top_k", 10)

        text_vec = self.clip.encode_text(text)
        raw_results = self.vector_store.search(text_vec, top_k=top_k)

        results = [
            {"id": path, "score": round(score, 6), "source": "clip_text"}
            for path, score, _ in raw_results
        ]

        return {
            "query": text,
            "mode": "clip_text",
            "results": results,
        }

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------

    def build_index(
        self,
        image_dir: str,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        batch_size: int = 32,
        train_sample_size: int = 5000,
    ):
        """
        批量构建向量索引。

        Args:
            image_dir: 图片目录
            extensions: 支持的图片扩展名
            batch_size: 编码批大小
            train_sample_size: IVF 训练样本数
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")

        # 收集图片
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.rglob(f"*{ext}"))
            image_paths.extend(image_dir.rglob(f"*{ext.upper()}"))
        image_paths = sorted(set(str(p) for p in image_paths))

        if not image_paths:
            logger.warning("目录中未找到图片: %s", image_dir)
            return

        logger.info("找到 %d 张图片，开始编码...", len(image_paths))

        # 批量编码
        embeddings = self.clip.encode_images_batch(image_paths, batch_size=batch_size)

        # IVF 模式需要先训练
        if self.vector_store.index_type == "ivf" and not self.vector_store._trained:
            sample_size = min(train_sample_size, len(embeddings))
            logger.info("IVF 索引训练中，样本数=%d...", sample_size)
            self.vector_store.train(embeddings[:sample_size])

        # 添加到索引
        self.vector_store.add(embeddings, image_paths)

        # 持久化
        self.vector_store.save()
        logger.info("索引构建完成，总量=%d", len(self.vector_store))

    def add_image(self, image_path: str):
        """增量添加单张图片到索引。"""
        embedding = self.clip.encode_image(image_path)
        self.vector_store.add_one(embedding, image_path)
        self.vector_store.save()
        logger.info("已添加图片到索引: %s", image_path)

    def save(self):
        """保存索引和用量。"""
        self.vector_store.save()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _clip_image_search(self, image_path: str, top_k: int) -> SearchResult:
        """CLIP 图像检索：编码查询图 → FAISS 搜索。"""
        query_vec = self.clip.encode_image(image_path)
        raw = self.vector_store.search(
            query_vec,
            top_k=self.config["fusion"].get("top_k_per_source", 20),
        )
        return [(path, score) for path, score, _ in raw]
