"""
统一候选数据结构定义。
无论候选来自本地还是百度，进入后续评分前都必须转换成这个格式。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Candidate:
    """统一候选数据结构。"""
    # 来源标识："local" 或 "baidu"
    source: str
    # 来源ID：本地为文件路径，百度为原始URL
    source_id: str
    # 百度原始URL（本地候选为None）
    source_url: Optional[str] = None
    # 本地文件路径（百度候选下载后的缓存路径）
    local_path: str = ""
    # 在来源中的排名（从1开始）
    source_rank: int = 0
    # 来源给出的分数（百度为None，本地为CLIP分数）
    source_score: Optional[float] = None
    # 匹配类型："local" 或 "visually_similar"
    match_type: str = "local"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "local_path": self.local_path,
            "source_rank": self.source_rank,
            "source_score": self.source_score,
            "match_type": self.match_type,
        }


class BaseProvider:
    """检索提供器基类，定义统一接口。"""

    def search(self, sample_path: str, top_k: int = 98) -> list:
        """
        检索候选图片。

        Args:
            sample_path: 样图路径
            top_k: 返回候选数量

        Returns:
            Candidate 对象列表
        """
        raise NotImplementedError("子类必须实现 search 方法")

    @property
    def is_available(self) -> bool:
        """提供器是否可用。"""
        return True
