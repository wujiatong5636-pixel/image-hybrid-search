"""
CLIP 轻量封装模块
支持两种推理后端：
  - ONNX Runtime（modern-onnx-clip）：CPU 友好，适合生产部署
  - OpenCLIP：开发测试，有 GPU 更佳

统一接口：encode_image / encode_text，输出均为 L2 归一化向量。
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPSearcher:
    """CLIP 模型封装，支持 ONNX / OpenCLIP 双后端切换。"""

    def __init__(
        self,
        backend: str = "onnx",
        onnx_model: str = "openai/clip-vit-base-patch32",
        openclip_model: str = "ViT-B-32",
        openclip_pretrained: str = "laion2b_s34b_b79k",
        embedding_dim: int = 512,
        device: str = "cpu",
    ):
        """
        Args:
            backend: 推理后端，"onnx" 或 "openclip"
            onnx_model: ONNX 模型标识（modern-onnx-clip 支持的模型名）
            openclip_model: OpenCLIP 模型架构名
            openclip_pretrained: OpenCLIP 预训练权重标识
            embedding_dim: 输出向量维度
            device: 推理设备 "cpu" / "cuda"
        """
        self.backend = backend.lower()
        self.embedding_dim = embedding_dim
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

        if self.backend == "onnx":
            self._init_onnx(onnx_model)
        elif self.backend == "openclip":
            self._init_openclip(openclip_model, openclip_pretrained)
        else:
            raise ValueError(f"不支持的后端: {backend}，请使用 'onnx' 或 'openclip'")

        logger.info("CLIP 初始化完成 | backend=%s | dim=%d | device=%s",
                     self.backend, self.embedding_dim, self.device)

    # ------------------------------------------------------------------
    # 后端初始化
    # ------------------------------------------------------------------

    def _init_onnx(self, model_name: str):
        """初始化 ONNX Runtime 后端（modern-onnx-clip）。"""
        try:
            from modern_onnx_clip import CLIPModel
        except ImportError:
            raise ImportError(
                "未安装 modern-onnx-clip，请运行: pip install modern-onnx-clip onnxruntime"
            )
        logger.info("加载 ONNX CLIP 模型: %s", model_name)
        self._model = CLIPModel.from_pretrained(model_name)

    def _init_openclip(self, model_name: str, pretrained: str):
        """初始化 OpenCLIP 后端。"""
        try:
            import open_clip
            import torch
        except ImportError:
            raise ImportError(
                "未安装 open-clip-torch，请运行: pip install open-clip-torch"
            )
        logger.info("加载 OpenCLIP 模型: %s / %s", model_name, pretrained)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()
        self._torch = torch

    # ------------------------------------------------------------------
    # 图像编码
    # ------------------------------------------------------------------

    def encode_image(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        编码单张图片，返回 L2 归一化向量。

        Args:
            image_input: 图片路径、Path 对象或 PIL Image

        Returns:
            shape=(embedding_dim,) 的 float32 向量
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"不支持的图像输入类型: {type(image_input)}")

        if self.backend == "onnx":
            embedding = self._model.encode_image(image)
            # modern-onnx-clip 可能返回 torch.Tensor 或 numpy
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()
            embedding = np.asarray(embedding, dtype=np.float32).flatten()
        else:
            import torch
            with torch.no_grad():
                x = self._preprocess(image).unsqueeze(0).to(self.device)
                embedding = self._model.encode_image(x)
                embedding = embedding.cpu().numpy().flatten()

        return self._normalize(embedding)

    def encode_images_batch(
        self, image_paths: List[Union[str, Path]], batch_size: int = 32
    ) -> np.ndarray:
        """
        批量编码图片。

        Args:
            image_paths: 图片路径列表
            batch_size: 批大小

        Returns:
            shape=(N, embedding_dim) 的 float32 矩阵，每行已 L2 归一化
        """
        embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            for path in batch:
                try:
                    emb = self.encode_image(path)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning("编码失败 %s: %s，跳过", path, e)
                    embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
            logger.info("已编码 %d / %d", min(i + batch_size, len(image_paths)), len(image_paths))

        return np.stack(embeddings) if embeddings else np.empty((0, self.embedding_dim))

    # ------------------------------------------------------------------
    # 文本编码
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> np.ndarray:
        """
        编码文本，返回 L2 归一化向量。

        Args:
            text: 输入文本

        Returns:
            shape=(embedding_dim,) 的 float32 向量
        """
        if self.backend == "onnx":
            embedding = self._model.encode_text(text)
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()
            embedding = np.asarray(embedding, dtype=np.float32).flatten()
        else:
            import torch
            with torch.no_grad():
                tokens = self._tokenizer([text]).to(self.device)
                embedding = self._model.encode_text(tokens)
                embedding = embedding.cpu().numpy().flatten()

        return self._normalize(embedding)

    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本。"""
        embeddings = [self.encode_text(t) for t in texts]
        return np.stack(embeddings) if embeddings else np.empty((0, self.embedding_dim))

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2 归一化，零向量保持不变。"""
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            return vec / norm
        return vec

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算两个向量的余弦相似度（输入已归一化时等价于内积）。"""
        return float(np.dot(vec_a, vec_b))
