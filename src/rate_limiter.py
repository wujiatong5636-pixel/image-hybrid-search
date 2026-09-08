"""
限流降级模块
监控 Google Vision API 调用量，达到免费额度阈值时自动降级到纯 CLIP 模式。

特性：
  - 用量持久化（JSON 文件）
  - 按月自动重置
  - 阈值告警 + 自动降级
  - 支持手动重置和查询
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Google Vision API 限流器，支持月度配额管理和自动降级。"""

    def __init__(
        self,
        max_per_month: int = 1000,
        alert_threshold: float = 0.8,
        usage_path: str = "data/vision_usage.json",
    ):
        """
        Args:
            max_per_month: 每月最大调用次数（免费额度）
            alert_threshold: 告警阈值比例（0-1），达到后自动降级
            usage_path: 用量记录文件路径
        """
        self.max_per_month = max_per_month
        self.alert_threshold = alert_threshold
        self.usage_path = Path(usage_path)
        self.used = 0
        self.month = self._current_month()
        self._load_usage()

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def allow_vision_call(self) -> bool:
        """
        检查是否允许本次 Vision 调用。
        达到阈值时返回 False，调用方应降级到纯 CLIP 模式。

        Returns:
            True = 允许调用；False = 已超限，应降级
        """
        # 跨月重置
        current_month = self._current_month()
        if current_month != self.month:
            logger.info("跨月检测：%s -> %s，重置用量", self.month, current_month)
            self._reset(current_month)

        if self.used / self.max_per_month >= self.alert_threshold:
            logger.warning(
                "Vision 用量告警: %d/%d (%.1f%%)，已达到阈值 %.0f%%，降级到纯 CLIP 模式",
                self.used, self.max_per_month,
                self.used / self.max_per_month * 100,
                self.alert_threshold * 100,
            )
            return False

        self.used += 1
        self._save_usage()
        return True

    def record_call(self):
        """记录一次调用（用于调用方在 allow 之后手动确认）。"""
        self.used += 1
        self._save_usage()

    def get_usage(self) -> dict:
        """获取当前用量信息。"""
        return {
            "month": self.month,
            "used": self.used,
            "max": self.max_per_month,
            "remaining": max(0, self.max_per_month - self.used),
            "usage_rate": self.used / self.max_per_month if self.max_per_month > 0 else 0,
            "degraded": self.used / self.max_per_month >= self.alert_threshold,
            "alert_threshold": self.alert_threshold,
        }

    def reset(self):
        """手动重置当月用量。"""
        self._reset(self._current_month())
        logger.info("用量已手动重置")

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _load_usage(self):
        """从磁盘加载用量记录。"""
        if not self.usage_path.exists():
            self.used = 0
            self.month = self._current_month()
            return

        try:
            with open(self.usage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.used = data.get("used", 0)
            self.month = data.get("month", self._current_month())

            # 跨月重置
            if self.month != self._current_month():
                logger.info("加载时检测到跨月，自动重置")
                self._reset(self._current_month())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("用量文件读取失败，重新初始化: %s", e)
            self.used = 0
            self.month = self._current_month()

    def _save_usage(self):
        """保存用量到磁盘。"""
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "used": self.used,
            "month": self.month,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            with open(self.usage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error("用量保存失败: %s", e)

    def _reset(self, month: str):
        """重置用量到指定月份。"""
        self.used = 0
        self.month = month
        self._save_usage()

    @staticmethod
    def _current_month() -> str:
        """获取当前月份字符串，格式 YYYY-MM。"""
        return datetime.now().strftime("%Y-%m")
