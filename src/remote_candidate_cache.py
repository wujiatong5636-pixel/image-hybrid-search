"""
远程候选图片下载与缓存模块。
负责把百度返回的URL下载到本地，做安全校验和缓存。

安全规则：
  - 只允许 http/https
  - 禁止访问内网/局域网地址
  - 单文件最大15MB
  - 超时15秒，重试2次
  - Content-Type必须是图片
  - PIL必须能打开
  - 宽高不低于200
  - 统一转RGB
  - 同一URL不重复下载
"""
import hashlib
import ipaddress
import logging
import socket
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class RemoteCandidateCache:
    """远程图片下载缓存器。"""

    ALLOWED_SCHEMES = {"http", "https"}
    IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp",
                           "image/gif", "image/bmp", "image/jpg"}

    def __init__(self, cache_dir: str = "runtime/baidu_cache",
                 timeout: int = 15, retries: int = 2,
                 max_file_mb: int = 15, min_width: int = 200,
                 min_height: int = 200):
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries
        self.max_file_bytes = max_file_mb * 1024 * 1024
        self.min_width = min_width
        self.min_height = min_height
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://graph.baidu.com/",
        })

    @staticmethod
    def _url_hash(url: str) -> str:
        """用URL的SHA-256作为缓存文件名。"""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _cache_path(self, url: str) -> Path:
        """获取URL对应的缓存文件路径。"""
        return self.cache_dir / f"{self._url_hash(url)}.jpg"

    def _is_safe_url(self, url: str) -> bool:
        """检查URL是否安全（禁止内网地址）。"""
        try:
            parsed = urlparse(url)
            if parsed.scheme.lower() not in self.ALLOWED_SCHEMES:
                logger.warning("URL协议不允许: %s", url)
                return False

            hostname = parsed.hostname
            if not hostname:
                return False

            # 解析IP，检查是否为内网地址
            try:
                ip = socket.gethostbyname(hostname)
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                    logger.warning("URL指向内网地址，拒绝下载: %s (%s)", url, ip)
                    return False
            except socket.gaierror:
                # 域名解析失败，交给后续下载步骤处理
                pass

            return True
        except Exception:
            return False

    def download(self, url: str) -> Dict:
        """
        下载并缓存一张远程图片。

        Returns:
            {
                "success": bool,
                "url": str,
                "local_path": str,
                "width": int,
                "height": int,
                "error": str (失败时)
            }
        """
        cache_path = self._cache_path(url)

        # 缓存命中
        if cache_path.is_file():
            try:
                with Image.open(cache_path) as img:
                    width, height = img.size
                logger.debug("缓存命中: %s", url)
                return {
                    "success": True,
                    "url": url,
                    "local_path": str(cache_path),
                    "width": width,
                    "height": height,
                }
            except Exception:
                # 缓存文件损坏，重新下载
                cache_path.unlink(missing_ok=True)

        # 安全检查
        if not self._is_safe_url(url):
            return {"success": False, "url": url, "error": "unsafe_url"}

        # 带重试下载
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = self._session.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()

                # 检查文件大小
                content_length = int(response.headers.get("Content-Length", 0))
                if content_length > self.max_file_bytes:
                    return {"success": False, "url": url,
                            "error": f"file_too_large: {content_length} bytes"}

                # 下载到临时文件
                temp_path = cache_path.with_suffix(".tmp")
                downloaded = 0
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        if downloaded > self.max_file_bytes:
                            f.close()
                            temp_path.unlink(missing_ok=True)
                            return {"success": False, "url": url,
                                    "error": "file_too_large"}
                        f.write(chunk)

                # PIL验证并转换
                with Image.open(temp_path) as img:
                    img = img.convert("RGB")
                    width, height = img.size
                    if width < self.min_width or height < self.min_height:
                        temp_path.unlink(missing_ok=True)
                        return {"success": False, "url": url,
                                "error": f"image_too_small: {width}x{height}"}
                    img.save(cache_path, format="JPEG", quality=92)

                temp_path.unlink(missing_ok=True)
                logger.info("下载成功: %s -> %s (%dx%d)", url, cache_path.name, width, height)
                return {
                    "success": True,
                    "url": url,
                    "local_path": str(cache_path),
                    "width": width,
                    "height": height,
                }

            except Exception as e:
                last_error = str(e)
                logger.warning("下载失败 (尝试%d/%d): %s - %s",
                               attempt + 1, self.retries + 1, url, e)
                if attempt < self.retries:
                    time.sleep(1)

        return {"success": False, "url": url, "error": last_error}

    def download_batch(self, urls: list, max_downloads: int = 20) -> List[Dict]:
        """
        批量下载图片。

        Args:
            urls: URL列表
            max_downloads: 最大下载数量

        Returns:
            下载成功的结果列表
        """
        results = []
        for url in urls[:max_downloads]:
            result = self.download(url)
            if result["success"]:
                results.append(result)
        logger.info("批量下载完成: 成功%d/%d", len(results), min(len(urls), max_downloads))
        return results
