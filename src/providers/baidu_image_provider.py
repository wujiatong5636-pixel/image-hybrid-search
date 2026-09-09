"""
百度识图提供器。
支持两种模式：
  - http：直接发HTTP请求解析结果（快，但百度反爬严格时可能被拒）
  - browser：用Playwright浏览器自动化（优先驱动系统Edge，免下载Chromium）
  - auto：先试HTTP，失败再用浏览器

注意：百度识图是网页逆向实现，接口随时可能变化。
本模块只负责获取图片URL列表，下载和评分由其他模块处理。
"""
import html as html_lib
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import requests

from .base import Candidate, BaseProvider

logger = logging.getLogger(__name__)


class BaiduImageProvider(BaseProvider):
    """百度识图提供器。"""

    # 百度识图上传接口
    UPLOAD_URL = "https://graph.baidu.com/upload"
    HOME_URL = "https://graph.baidu.com/pcpage/index?tpl_from=pc"
    MAIN_URL = "https://www.baidu.com/"

    BROWSER_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
    )

    def __init__(self, config: dict):
        """
        Args:
            config: config.yaml 中的 baidu_image_search 配置段
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.mode = self.config.get("mode", "auto")
        self.timeout = self.config.get("timeout", 30)
        self.max_results = self.config.get("max_results", 30)
        self.retries = self.config.get("retries", 1)
        self.min_interval = self.config.get("min_interval_seconds", 3)
        self.circuit_breaker_failures = self.config.get("circuit_breaker_failures", 3)
        # 浏览器渠道：优先系统Edge，其次Chromium
        self.browser_channel = self.config.get("browser_channel", "msedge")

        # 熔断状态
        self._consecutive_failures = 0
        self._last_request_time = 0.0
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        """构建带浏览器UA的请求会话。"""
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.BROWSER_UA,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        })
        return session

    @property
    def is_circuit_open(self) -> bool:
        """熔断器是否打开（连续失败过多时暂时停止调用）。"""
        return self._consecutive_failures >= self.circuit_breaker_failures

    def _record_success(self):
        """记录一次成功调用，重置失败计数。"""
        self._consecutive_failures = 0

    def _record_failure(self):
        """记录一次失败调用。"""
        self._consecutive_failures += 1
        logger.warning("百度识图连续失败 %d/%d 次",
                       self._consecutive_failures, self.circuit_breaker_failures)

    def _wait_interval(self):
        """两次请求之间的最小间隔，防止被封。"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    def search(self, sample_path: str, top_k: int = 30) -> List[Candidate]:
        """
        百度识图检索。

        Args:
            sample_path: 样图路径
            top_k: 返回URL数量上限

        Returns:
            Candidate 列表（只有URL，尚未下载）
        """
        if not self.enabled:
            logger.info("百度识图已在配置中禁用")
            return []

        if self.is_circuit_open:
            logger.warning("百度识图熔断器已打开（连续失败%d次），跳过本次调用",
                           self._consecutive_failures)
            return []

        sample_path = str(Path(sample_path).resolve())
        if not Path(sample_path).is_file():
            logger.error("样图不存在: %s", sample_path)
            return []

        urls: List[str] = []

        # 根据模式选择执行方式
        if self.mode in ("http", "auto"):
            urls = self._search_http(sample_path)
            if urls:
                self._record_success()
            elif self.mode == "auto":
                logger.info("HTTP模式无结果，自动切换浏览器模式")
                urls = self._search_browser(sample_path)
                if urls:
                    self._record_success()
                else:
                    self._record_failure()
            else:
                self._record_failure()
        elif self.mode == "browser":
            urls = self._search_browser(sample_path)
            if urls:
                self._record_success()
            else:
                self._record_failure()

        # 限制返回数量
        urls = urls[:min(top_k, self.max_results)]

        # 转换为统一Candidate格式
        candidates = []
        for rank, url in enumerate(urls, start=1):
            candidate = Candidate(
                source="baidu",
                source_id=url,
                source_url=url,
                local_path="",  # 下载后才填充
                source_rank=rank,
                source_score=None,  # 百度分数不采信
                match_type="visually_similar",
            )
            candidates.append(candidate)

        logger.info("百度识图返回 %d 个URL", len(candidates))
        return candidates

    def _warmup_cookies(self):
        """上传前先访问百度主站和识图页，预热cookie。"""
        for url in (self.MAIN_URL, self.HOME_URL):
            try:
                self._session.get(url, timeout=self.timeout)
            except Exception as e:
                logger.debug("cookie预热失败 %s: %s", url, e)

    def _search_http(self, sample_path: str) -> List[str]:
        """HTTP模式：上传图片并解析结果。"""
        try:
            self._wait_interval()
            self._warmup_cookies()

            # 上传图片（带完整浏览器头）
            self._session.headers.update({
                "Origin": "https://graph.baidu.com",
                "Referer": self.HOME_URL,
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json, text/javascript, */*; q=0.01",
            })
            with open(sample_path, "rb") as f:
                files = {"image": (Path(sample_path).name, f, "image/png")}
                form = {"range": '{"page_from": "searchIndex"}'}
                response = self._session.post(
                    self.UPLOAD_URL, files=files, data=form, timeout=self.timeout,
                )
            response.raise_for_status()
            data = response.json()

            # 百度返回 status=0 才是成功；data 可能为 None
            if not isinstance(data, dict) or data.get("status") != 0:
                logger.warning("百度上传被拒: status=%s msg=%s",
                               data.get("status") if isinstance(data, dict) else "?",
                               data.get("msg") if isinstance(data, dict) else "?")
                return []

            data_section = data.get("data") or {}
            search_url = data_section.get("url", "")
            if not search_url:
                logger.warning("百度上传返回中没有搜索URL: %s", str(data)[:200])
                return []

            # 访问搜索结果页
            self._wait_interval()
            self._session.headers.pop("X-Requested-With", None)
            response = self._session.get(search_url, timeout=self.timeout)
            response.raise_for_status()

            # 从HTML中提取图片URL
            urls = self._extract_urls_from_html(response.text)
            return urls

        except Exception as e:
            logger.warning("百度HTTP模式失败: %s", e)
            return []

    # URL中包含这些词的判定为图标/UI元素，不是结果图
    _JUNK_KEYWORDS = (
        "icon", "logo", "loading", "sprite", "button", "arrow", "blank",
        "default", "avatar", "pc-icon", "textocr", "static/", "/style/",
        "placeholder", "gif", "btn_", "_btn", "qrcode", "qr_code",
    )
    # 百度结果图床域名特征
    _RESULT_HOST_KEYS = ("baidu.com/it/", "bcebos.com/", "bdimg.com/", "hiphotos")

    def _is_valid_result_url(self, url: str) -> bool:
        """判断URL是否为真正的结果图（过滤图标、Logo、UI元素）。"""
        if not url or not url.startswith("http"):
            return False
        lower = url.lower()
        # 过滤图标类
        if any(junk in lower for junk in self._JUNK_KEYWORDS):
            return False
        return True

    def _extract_urls_from_html(self, html: str) -> List[str]:
        """从百度搜索结果HTML中提取图片URL。"""
        urls = []

        def _add(url: str):
            # 还原HTML实体（&amp; -> &）和转义斜杠
            url = html_lib.unescape(url).replace("\\/", "/").replace("\\u002f", "/")
            if self._is_valid_result_url(url) and url not in urls:
                urls.append(url)

        # 尝试从JSON数据中提取（百度页面通常把结果嵌在script标签里）
        json_patterns = [
            r'"thumbUrl"\s*:\s*"([^"]+)"',
            r'"objUrl"\s*:\s*"([^"]+)"',
            r'"middleUrl"\s*:\s*"([^"]+)"',
            r'"hoverUrl"\s*:\s*"([^"]+)"',
            r'"imgUrl"\s*:\s*"([^"]+)"',
            r'"originUrl"\s*:\s*"([^"]+)"',
            r'"image"\s*:\s*"(https?://[^"]+)"',
        ]
        for pattern in json_patterns:
            for match in re.finditer(pattern, html):
                _add(match.group(1))

        # 尝试从img标签提取
        for match in re.finditer(r'<img[^>]+(?:data-imgurl|data-src|src)="(https?://[^"]+)"', html):
            _add(match.group(1))

        # 结果图床优先排序（真正的相似图排前面）
        def _priority(u: str) -> int:
            low = u.lower()
            if any(k in low for k in self._RESULT_HOST_KEYS):
                return 0
            return 1
        urls.sort(key=_priority)
        return urls

    def _launch_context(self, playwright, profile_dir: str):
        """启动浏览器上下文：优先系统Edge，失败回退Chromium。"""
        launch_kwargs = dict(
            user_data_dir=profile_dir,
            headless=True,
            viewport={"width": 1280, "height": 800},
        )
        # 先尝试系统Edge（免下载）
        if self.browser_channel:
            try:
                return playwright.chromium.launch_persistent_context(
                    channel=self.browser_channel, **launch_kwargs
                )
            except Exception as e:
                logger.info("系统%s启动失败，回退Chromium: %s", self.browser_channel, e)
        # 回退到playwright自带chromium
        return playwright.chromium.launch_persistent_context(**launch_kwargs)

    def _search_browser(self, sample_path: str) -> List[str]:
        """
        浏览器模式：打开百度识图页面上传图片，读取结果。
        优先驱动系统Edge，无需下载Chromium。
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.warning("未安装playwright，浏览器模式不可用。"
                           "请运行: pip install playwright")
            return []

        profile_dir = str(Path(self.config.get(
            "browser_profile_dir", "runtime/baidu_browser_profile"
        )).resolve())
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        try:
            self._wait_interval()
            with sync_playwright() as p:
                context = self._launch_context(p, profile_dir)
                page = context.new_page()

                # 打开百度识图首页
                page.goto(self.HOME_URL, timeout=self.timeout * 1000, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)

                # 上传图片
                file_input = page.locator('input[type="file"]').first
                file_input.wait_for(state="attached", timeout=self.timeout * 1000)
                file_input.set_input_files(sample_path)

                # 等待结果页加载（上传后会自动跳转）
                try:
                    page.wait_for_load_state("networkidle", timeout=self.timeout * 1000)
                except Exception:
                    pass
                # 额外等待动态图片加载
                page.wait_for_timeout(5000)

                # 滚动页面触发懒加载
                for _ in range(3):
                    page.mouse.wheel(0, 1500)
                    page.wait_for_timeout(800)

                # 提取页面中的图片URL
                html = page.content()
                urls = self._extract_urls_from_html(html)

                # 如果HTML提取不足，尝试从网络请求里抓图片响应
                if len(urls) < 3:
                    urls = self._extract_from_images(page, urls)

                context.close()
                logger.info("浏览器模式提取到 %d 个URL", len(urls))
                return urls

        except Exception as e:
            logger.warning("百度浏览器模式失败: %s", e)
            return []

    def _extract_from_images(self, page, existing: List[str]) -> List[str]:
        """从页面img元素的src/data-src属性补充提取图片URL。"""
        urls = list(existing)
        try:
            src_list = page.eval_on_selector_all(
                "img",
                "els => els.map(e => e.src || e.getAttribute('data-src') || '')"
            )
            for url in src_list:
                url = html_lib.unescape(url or "")
                if self._is_valid_result_url(url) and url not in urls:
                    urls.append(url)
        except Exception:
            pass
        return urls

    @property
    def is_available(self) -> bool:
        return self.enabled and not self.is_circuit_open

    def reset_circuit_breaker(self):
        """手动重置熔断器。"""
        self._consecutive_failures = 0
