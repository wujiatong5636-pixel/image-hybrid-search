"""
图片重复检测模块。

检测方法：
1. SHA-256：识别文件内容完全相同的图片；
2. pHash：识别缩放、压缩、轻微调色后的近重复图片。

本模块只负责检测，不会删除或移动图片。
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import imagehash
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


class DuplicateDetector:
    """图片重复检测器。"""

    def __init__(self, phash_threshold: int = 6):
        if phash_threshold < 0:
            raise ValueError("pHash阈值不能小于0")

        self.phash_threshold = phash_threshold
        self._sha_cache: Dict[str, str] = {}
        self._phash_cache: Dict[str, imagehash.ImageHash] = {}

    @staticmethod
    def _cache_key(image_path: Path) -> str:
        return str(image_path.resolve())

    def calculate_sha256(self, image_path: Path) -> str:
        """计算文件SHA-256。"""
        image_path = Path(image_path)
        cache_key = self._cache_key(image_path)

        if cache_key in self._sha_cache:
            return self._sha_cache[cache_key]

        digest = hashlib.sha256()

        with image_path.open("rb") as file:
            while True:
                block = file.read(1024 * 1024)

                if not block:
                    break

                digest.update(block)

        result = digest.hexdigest()
        self._sha_cache[cache_key] = result
        return result

    def calculate_phash(
        self,
        image_path: Path,
    ) -> imagehash.ImageHash:
        """计算图片感知哈希。"""
        image_path = Path(image_path)
        cache_key = self._cache_key(image_path)

        if cache_key in self._phash_cache:
            return self._phash_cache[cache_key]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            result = imagehash.phash(image)

        self._phash_cache[cache_key] = result
        return result

    def compare(
        self,
        source_path: Path,
        candidate_path: Path,
    ) -> dict:
        """
        比较一张样图和一张候选图。

        返回：
        {
            "is_duplicate": 是否属于重复图片,
            "duplicate_type": exact、near或none,
            "sha256_equal": SHA-256是否相同,
            "phash_distance": pHash距离
        }
        """
        source_path = Path(source_path)
        candidate_path = Path(candidate_path)

        if not source_path.exists():
            raise FileNotFoundError(f"样图不存在：{source_path}")

        if not candidate_path.exists():
            raise FileNotFoundError(
                f"候选图片不存在：{candidate_path}"
            )

        source_sha = self.calculate_sha256(source_path)
        candidate_sha = self.calculate_sha256(candidate_path)
        sha256_equal = source_sha == candidate_sha

        if sha256_equal:
            return {
                "source": str(source_path.resolve()),
                "candidate": str(candidate_path.resolve()),
                "is_duplicate": True,
                "duplicate_type": "exact",
                "sha256_equal": True,
                "phash_distance": 0,
            }

        source_phash = self.calculate_phash(source_path)
        candidate_phash = self.calculate_phash(candidate_path)
        phash_distance = source_phash - candidate_phash

        is_near_duplicate = (
            phash_distance <= self.phash_threshold
        )

        return {
            "source": str(source_path.resolve()),
            "candidate": str(candidate_path.resolve()),
            "is_duplicate": is_near_duplicate,
            "duplicate_type": (
                "near" if is_near_duplicate else "none"
            ),
            "sha256_equal": False,
            "phash_distance": phash_distance,
        }


def collect_images(directory: Path) -> List[Path]:
    """收集目录内支持的图片文件。"""
    return [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="扫描样图与候选图库中的重复图片"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="样图目录",
    )

    parser.add_argument(
        "--candidate_dir",
        required=True,
        help="候选图库目录",
    )

    parser.add_argument(
        "--phash_threshold",
        type=int,
        default=6,
        help="pHash近重复阈值，默认6",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()

    if not input_dir.is_dir():
        print(f"[错误] 样图目录不存在：{input_dir}")
        return 1

    if not candidate_dir.is_dir():
        print(f"[错误] 候选图库不存在：{candidate_dir}")
        return 1

    sample_images = collect_images(input_dir)
    candidate_images = collect_images(candidate_dir)

    detector = DuplicateDetector(
        phash_threshold=args.phash_threshold
    )

    exact_results = []
    near_results = []
    failed_results = []

    print("=" * 60)
    print("开始扫描重复图片")
    print(f"样图数量：{len(sample_images)}")
    print(f"候选数量：{len(candidate_images)}")
    print(
        f"比较次数："
        f"{len(sample_images) * len(candidate_images)}"
    )
    print(
        f"pHash近重复阈值："
        f"{args.phash_threshold}"
    )
    print("=" * 60)

    for sample_path in sample_images:
        for candidate_path in candidate_images:
            try:
                result = detector.compare(
                    sample_path,
                    candidate_path,
                )
            except Exception as error:
                failed_results.append(
                    {
                        "source": sample_path.name,
                        "candidate": candidate_path.name,
                        "error": str(error),
                    }
                )
                continue

            if result["duplicate_type"] == "exact":
                exact_results.append(result)
            elif result["duplicate_type"] == "near":
                near_results.append(result)

    print("\n扫描完成")
    print(f"完全重复：{len(exact_results)}")
    print(f"近似重复：{len(near_results)}")
    print(f"读取失败：{len(failed_results)}")

    if exact_results:
        print("\n完全重复图片：")

        for result in exact_results:
            print(
                f"样图：{Path(result['source']).name} "
                f"<=> 候选："
                f"{Path(result['candidate']).name}"
            )

    if near_results:
        print("\n近似重复图片：")

        for result in near_results:
            print(
                f"样图：{Path(result['source']).name} "
                f"<=> 候选："
                f"{Path(result['candidate']).name} "
                f"| pHash距离="
                f"{result['phash_distance']}"
            )

    if failed_results:
        print("\n读取失败的图片：")

        for result in failed_results:
            print(
                f"样图：{result['source']} "
                f"| 候选：{result['candidate']} "
                f"| 错误：{result['error']}"
            )

    print("\n提示：本程序只检测，不移动、不删除图片。")
    return 0


if __name__ == "__main__":
    sys.exit(main())