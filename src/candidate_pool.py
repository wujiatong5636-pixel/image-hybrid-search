"""
候选池合并与去重模块。
负责把本地候选和百度候选合并，执行两层去重：
  1. 和样图比较（完全重复/近重复）
  2. 候选之间互相比较（SHA256相同/pHash过小/全局占用）

来源优先规则：内容相同时优先保留本地候选。
"""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class CandidatePool:
    """候选池合并与去重器。"""

    def __init__(self, duplicate_detector, global_registry=None):
        """
        Args:
            duplicate_detector: DuplicateDetector 实例
            global_registry: GlobalCandidateRegistry 实例（可选）
        """
        self.detector = duplicate_detector
        self.registry = global_registry

    def merge_and_deduplicate(
        self,
        sample_path: str,
        local_candidates: List[dict],
        baidu_candidates: List[dict],
    ) -> List[dict]:
        """
        合并本地和百度候选，执行去重。

        Args:
            sample_path: 样图路径
            local_candidates: 本地候选列表（已过滤）
            baidu_candidates: 百度候选列表（已下载、已重新评分）

        Returns:
            去重后的合并候选列表
        """
        sample_path = str(Path(sample_path).resolve())

        # 第一层：和样图去重
        local_candidates = self._filter_against_sample(sample_path, local_candidates)
        baidu_candidates = self._filter_against_sample(sample_path, baidu_candidates)

        # 合并：本地候选在前（优先级高）
        all_candidates = local_candidates + baidu_candidates

        # 第二层：候选之间去重
        result = self._deduplicate_among_candidates(all_candidates)

        logger.info("候选池合并完成: 本地%d + 百度%d -> 合并后%d",
                     len(local_candidates), len(baidu_candidates), len(result))
        return result

    def _filter_against_sample(self, sample_path: str, candidates: List[dict]) -> List[dict]:
        """过滤掉和样图重复的候选。"""
        result = []
        for cand in candidates:
            cand_path = cand.get("path", cand.get("local_path", ""))
            if not cand_path or not Path(cand_path).is_file():
                continue
            try:
                check = self.detector.compare(sample_path, cand_path)
                if check["is_duplicate"]:
                    logger.debug("候选与样图重复，剔除: %s", cand_path)
                    continue
                cand["phash_distance"] = int(check["phash_distance"])
                result.append(cand)
            except Exception as e:
                logger.warning("候选与样图比较失败，剔除: %s - %s", cand_path, e)
        return result

    def _deduplicate_among_candidates(self, candidates: List[dict]) -> List[dict]:
        """候选之间互相去重。"""
        result = []
        seen_hashes = set()

        for cand in candidates:
            cand_path = cand.get("path", cand.get("local_path", ""))
            if not cand_path or not Path(cand_path).is_file():
                continue

            # 全局占用检查
            if self.registry:
                try:
                    if self.registry.is_used(Path(cand_path)):
                        logger.debug("候选已被全局占用，剔除: %s", cand_path)
                        continue
                except Exception:
                    pass

            # SHA256去重
            try:
                cand_hash = self.detector.calculate_sha256(Path(cand_path))
            except Exception:
                continue

            if cand_hash in seen_hashes:
                logger.debug("候选SHA256重复，剔除: %s", cand_path)
                continue

            # pHash近重复检查（和已保留的候选比较）
            is_near_dup = False
            for kept in list(result):
                kept_path = kept.get("path", kept.get("local_path", ""))
                if not kept_path:
                    continue
                try:
                    check = self.detector.compare(cand_path, kept_path)
                    if check["is_duplicate"]:
                        # 近重复时保留质量更高的（语义分数更高的）
                        cand_score = cand.get("semantic_score", 0)
                        kept_score = kept.get("semantic_score", 0)
                        if cand_score > kept_score:
                            # 新候选质量更高，替换旧的
                            result.remove(kept)
                            try:
                                seen_hashes.discard(
                                    self.detector.calculate_sha256(Path(kept_path))
                                )
                            except Exception:
                                pass
                            logger.debug("近重复候选替换: %s (%.3f) > %s (%.3f)",
                                         cand_path, cand_score, kept_path, kept_score)
                        else:
                            is_near_dup = True
                            logger.debug("候选与已保留候选近重复，剔除: %s", cand_path)
                        break
                except Exception:
                    continue

            if not is_near_dup:
                seen_hashes.add(cand_hash)
                result.append(cand)

        return result
