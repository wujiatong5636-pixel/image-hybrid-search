"""Google Vision Web Detection 候选提供器。"""

import logging
from pathlib import Path
from typing import List, Optional

from .base import BaseProvider, Candidate

logger = logging.getLogger(__name__)


class GoogleVisionProvider(BaseProvider):
    """仅在本地与百度候选仍不足时提供视觉相似图片 URL。"""

    def __init__(self, detector, rate_limiter=None, config: Optional[dict] = None):
        self.detector = detector
        self.rate_limiter = rate_limiter
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.max_results = int(self.config.get("max_results", 30))
        self.include_partial_matches = bool(
            self.config.get("include_partial_matches", False)
        )
        self.last_status = "not_called"
        self.last_error = ""

    @property
    def is_available(self) -> bool:
        return bool(
            self.enabled
            and self.detector is not None
            and getattr(self.detector, "is_available", False)
        )

    def search(self, sample_path: str, top_k: int = 30) -> List[Candidate]:
        self.last_error = ""
        if not self.enabled:
            self.last_status = "disabled"
            return []
        if not self.is_available:
            self.last_status = "unavailable"
            return []
        if not Path(sample_path).is_file():
            self.last_status = "invalid_sample"
            return []
        if self.rate_limiter and not self.rate_limiter.allow_vision_call():
            self.last_status = "quota_limited"
            return []

        try:
            result = self.detector.detect(sample_path)
            image_items = list(result.get("similar_images", []))
            if self.include_partial_matches:
                image_items.extend(result.get("partial_matches", []))

            seen = set()
            candidates = []
            limit = min(int(top_k), self.max_results)
            for item in image_items:
                url = item.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                candidates.append(
                    Candidate(
                        source="google",
                        source_id=url,
                        source_url=url,
                        source_rank=len(candidates) + 1,
                        source_score=item.get("score") or None,
                        match_type="visually_similar",
                    )
                )
                if len(candidates) >= limit:
                    break

            self.last_status = "success" if candidates else "empty"
            return candidates
        except Exception as error:
            self.last_status = "error"
            self.last_error = str(error)
            logger.exception("Google Vision 候选检索失败: %s", error)
            return []
