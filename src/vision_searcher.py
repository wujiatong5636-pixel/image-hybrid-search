"""
Google Vision Web Detection 模块
调用 Google Cloud Vision API 的 WEB_DETECTION 功能，返回全网图片溯源结果。

结果类型：
  - full_matching_images：完全匹配（同图）
  - partial_matching_images：部分匹配（裁剪/局部）
  - pages_with_matching_images：包含该图的网页
  - visually_similar_images：视觉相似图片
  - web_entities：网络实体标签
  - best_guess_labels：最佳猜测标签

注意：Web Detection 不返回相似度分值，只有 URL 列表，融合时使用排名而非分值。
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VisionWebDetector:
    """Google Vision Web Detection 封装。"""

    def __init__(
        self,
        credentials_path: str = "credentials/google_vision_key.json",
        proxy: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ):
        """
        Args:
            credentials_path: Google Cloud 服务账号 JSON 密钥路径
            proxy: 代理配置 {"http": "http://...", "https": "http://..."}
            timeout: 请求超时（秒）
        """
        self.credentials_path = Path(credentials_path)
        self.proxy = proxy or {}
        self.timeout = timeout
        self._client = None
        self._available = False

        self._init_client()

    def _init_client(self):
        """初始化 Vision API 客户端。"""
        try:
            from google.cloud import vision
        except ImportError:
            logger.warning("未安装 google-cloud-vision，Vision 模块不可用")
            return

        if not self.credentials_path.exists():
            logger.warning("未找到 Google Vision 密钥文件: %s，Vision 模块不可用",
                           self.credentials_path)
            return

        # 设置凭证环境变量
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(self.credentials_path.resolve())

        # 配置代理（通过环境变量）
        if self.proxy.get("http"):
            os.environ["HTTP_PROXY"] = self.proxy["http"]
        if self.proxy.get("https"):
            os.environ["HTTPS_PROXY"] = self.proxy["https"]

        try:
            self._client = vision.ImageAnnotatorClient()
            self._available = True
            logger.info("Google Vision 客户端初始化成功")
        except Exception as e:
            logger.error("Google Vision 客户端初始化失败: %s", e)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def detect(self, image_path: str) -> Dict:
        """
        对单张图片执行 Web Detection。

        Args:
            image_path: 图片路径

        Returns:
            {
                "full_matches": [{"url": ..., "score": ...}],
                "partial_matches": [...],
                "similar_images": [...],
                "pages_with_matches": [{"url": ..., "page_title": ..., "full_matches": [...], "partial_matches": [...]}],
                "web_entities": [{"description": ..., "score": ...}],
                "best_guess_labels": [{"label": ..., "language_code": ...}],
            }
        """
        if not self._available:
            logger.warning("Vision 模块不可用，返回空结果")
            return self._empty_result()

        try:
            from google.cloud import vision

            with open(image_path, "rb") as f:
                content = f.read()

            image = vision.Image(content=content)
            response = self._client.web_detection(image=image, timeout=self.timeout)
            web = response.web_detection

            result = {
                "full_matches": self._parse_images(web.full_matching_images),
                "partial_matches": self._parse_images(web.partial_matching_images),
                "similar_images": self._parse_images(web.visually_similar_images),
                "pages_with_matches": self._parse_pages(web.pages_with_matching_images),
                "web_entities": [
                    {"description": e.description, "score": e.score}
                    for e in web.web_entities
                ],
                "best_guess_labels": [
                    {"label": l.label, "language_code": l.language_code}
                    for l in web.best_guess_labels
                ],
            }

            logger.info(
                "Web Detection 完成 | full=%d partial=%d similar=%d pages=%d entities=%d",
                len(result["full_matches"]),
                len(result["partial_matches"]),
                len(result["similar_images"]),
                len(result["pages_with_matches"]),
                len(result["web_entities"]),
            )
            return result

        except Exception as e:
            logger.error("Web Detection 调用失败 %s: %s", image_path, e)
            return self._empty_result()

    def get_all_image_urls(self, result: Dict) -> List[str]:
        """
        从检测结果中提取所有图片 URL（full + partial + similar），按优先级去重。

        Args:
            result: detect() 返回的结果字典

        Returns:
            URL 列表，按 full → partial → similar 顺序排列
        """
        urls = []
        seen = set()
        for key in ["full_matches", "partial_matches", "similar_images"]:
            for item in result.get(key, []):
                url = item.get("url", "")
                if url and url not in seen:
                    urls.append(url)
                    seen.add(url)
        return urls

    # ------------------------------------------------------------------
    # 内部解析方法
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_images(images) -> List[Dict]:
        """解析图片匹配列表。"""
        return [
            {
                "url": img.url,
                "score": getattr(img, "score", 0.0) or 0.0,
            }
            for img in images
        ]

    @staticmethod
    def _parse_pages(pages) -> List[Dict]:
        """解析包含匹配图片的网页列表。"""
        result = []
        for page in pages:
            result.append({
                "url": page.url,
                "page_title": page.page_title,
                "full_matches": [img.url for img in page.full_matching_images],
                "partial_matches": [img.url for img in page.partial_matching_images],
            })
        return result

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "full_matches": [],
            "partial_matches": [],
            "similar_images": [],
            "pages_with_matches": [],
            "web_entities": [],
            "best_guess_labels": [],
        }
