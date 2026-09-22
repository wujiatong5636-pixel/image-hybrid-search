"""Google Vision 降级提供器与触发策略测试。"""

from pathlib import Path

import pytest

from src.fallback_policy import needs_fallback
from src.providers.google_vision_provider import GoogleVisionProvider


class FakeDetector:
    is_available = True

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def detect(self, _sample_path):
        self.calls += 1
        return self.result


class FakeLimiter:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.calls = 0

    def allow_vision_call(self):
        self.calls += 1
        return self.allowed


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, True), (1, True), (2, True), (3, False), (8, False)],
)
def test_fallback_only_when_candidates_are_less_than_three(count, expected):
    assert needs_fallback(count, 3) is expected


def test_google_provider_uses_similar_images_and_excludes_full_matches(tmp_path):
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"sample")
    detector = FakeDetector(
        {
            "full_matches": [{"url": "https://example.com/full.jpg"}],
            "partial_matches": [{"url": "https://example.com/partial.jpg"}],
            "similar_images": [
                {"url": "https://example.com/a.jpg", "score": 0.8},
                {"url": "https://example.com/a.jpg", "score": 0.8},
                {"url": "https://example.com/b.jpg", "score": 0.7},
            ],
        }
    )
    limiter = FakeLimiter()
    provider = GoogleVisionProvider(
        detector,
        limiter,
        {"enabled": True, "max_results": 10, "include_partial_matches": False},
    )

    results = provider.search(str(sample), top_k=10)

    assert [item.source_url for item in results] == [
        "https://example.com/a.jpg",
        "https://example.com/b.jpg",
    ]
    assert all(item.source == "google" for item in results)
    assert provider.last_status == "success"
    assert detector.calls == 1
    assert limiter.calls == 1


def test_google_provider_skips_call_when_quota_is_limited(tmp_path):
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"sample")
    detector = FakeDetector({"similar_images": []})
    provider = GoogleVisionProvider(detector, FakeLimiter(False), {"enabled": True})

    assert provider.search(str(sample)) == []
    assert provider.last_status == "quota_limited"
    assert detector.calls == 0


def test_google_provider_is_unavailable_without_detector():
    provider = GoogleVisionProvider(None, None, {"enabled": True})
    assert provider.is_available is False
    assert provider.search(str(Path("missing.png"))) == []
    assert provider.last_status == "unavailable"
