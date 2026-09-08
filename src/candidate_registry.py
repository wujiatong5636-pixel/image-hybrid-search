"""
全局候选图片占用登记模块。

功能：
1. 使用SHA-256识别候选图片；
2. 防止同一图片被不同样图重复使用；
3. 将占用状态保存到JSON；
4. 支持程序中断后恢复。
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class GlobalCandidateRegistry:
    """管理整个批量任务中已经使用的候选图片。"""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.records: Dict[str, dict] = {}
        self._hash_cache: Dict[str, str] = {}
        self.load()

    @staticmethod
    def _path_key(image_path: Path) -> str:
        """生成文件路径缓存键。"""
        return str(Path(image_path).resolve())

    def calculate_sha256(self, image_path: Path) -> str:
        """计算候选图片的SHA-256。"""
        image_path = Path(image_path).resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"候选图片不存在：{image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"候选路径不是文件：{image_path}"
            )

        cache_key = self._path_key(image_path)

        if cache_key in self._hash_cache:
            return self._hash_cache[cache_key]

        digest = hashlib.sha256()

        with image_path.open("rb") as file:
            while True:
                block = file.read(1024 * 1024)

                if not block:
                    break

                digest.update(block)

        result = digest.hexdigest()
        self._hash_cache[cache_key] = result
        return result

    def load(self) -> None:
        """读取已有的全局候选占用记录。"""
        if not self.registry_path.exists():
            self.records = {}
            return

        try:
            with self.registry_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"候选占用记录已损坏："
                f"{self.registry_path}，错误：{error}"
            ) from error

        if not isinstance(data, dict):
            raise ValueError(
                "候选占用记录格式错误，最外层必须是字典"
            )

        records = data.get("records", {})

        if not isinstance(records, dict):
            raise ValueError(
                "候选占用记录中的records格式错误"
            )

        self.records = records

    def save(self) -> None:
        """
        保存占用记录。

        先写入临时文件，再替换正式文件，
        避免程序突然退出导致JSON只写了一半。
        """
        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = self.registry_path.with_suffix(
            self.registry_path.suffix + ".tmp"
        )

        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "record_count": len(self.records),
            "records": self.records,
        }

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(temp_path, self.registry_path)

    def is_used(self, image_path: Path) -> bool:
        """判断候选图片是否已经被使用。"""
        image_hash = self.calculate_sha256(image_path)
        return image_hash in self.records

    def get_record(
        self,
        image_path: Path,
    ) -> Optional[dict]:
        """查询候选图片的占用信息。"""
        image_hash = self.calculate_sha256(image_path)
        record = self.records.get(image_hash)

        if record is None:
            return None

        return dict(record)

    def reserve(
        self,
        image_path: Path,
        sample_sequence: int,
        sample_name: str,
        group: str,
    ) -> bool:
        """
        尝试占用一张候选图片。

        返回True：
        候选此前没有使用，本次占用成功。

        返回False：
        候选已经被其他结果使用，本次不再占用。
        """
        image_path = Path(image_path).resolve()
        image_hash = self.calculate_sha256(image_path)

        if image_hash in self.records:
            return False

        self.records[image_hash] = {
            "candidate_hash": image_hash,
            "candidate_path": str(image_path),
            "candidate_name": image_path.name,
            "sample_sequence": sample_sequence,
            "sample_name": sample_name,
            "group": group,
            "reserved_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        self.save()
        return True

    def release(self, image_path: Path) -> bool:
        """
        释放候选图片。

        只有在结果保存失败或任务回滚时使用。
        """
        image_hash = self.calculate_sha256(image_path)

        if image_hash not in self.records:
            return False

        del self.records[image_hash]
        self.save()
        return True

    def count(self) -> int:
        """返回已占用候选图片数量。"""
        return len(self.records)

    def get_used_hashes(self) -> set:
        """返回全部已使用图片的SHA-256集合。"""
        return set(self.records.keys())