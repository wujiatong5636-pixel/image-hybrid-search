"""验证一张真实样图的完整候选选择链路。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_registry import GlobalCandidateRegistry
from src.duplicate_detector import DuplicateDetector
from src.group_selector import BasicGroupSelector
from src.hybrid_searcher import HybridSearcher


def main() -> int:
    parser = argparse.ArgumentParser(description="真实图片三分组联调")
    parser.add_argument("--query", required=True, help="样图路径")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument(
        "--registry",
        default=str(PROJECT_ROOT / "runtime/tasks/step12_test_registry.json"),
        help="本次联调使用的候选占用记录",
    )
    parser.add_argument(
        "--reset_registry",
        action="store_true",
        help="运行前清空第十二步测试占用记录",
    )
    args = parser.parse_args()

    query_path = Path(args.query).resolve()
    config_path = Path(args.config).resolve()
    registry_path = Path(args.registry).resolve()

    if not query_path.is_file():
        print(f"[错误] 样图不存在：{query_path}")
        return 1
    if not config_path.is_file():
        print(f"[错误] 配置文件不存在：{config_path}")
        return 1
    if args.top_k < 3:
        print("[错误] top_k至少为3")
        return 1

    if args.reset_registry:
        registry_path.unlink(missing_ok=True)
        registry_path.with_suffix(registry_path.suffix + ".tmp").unlink(
            missing_ok=True
        )

    print("正在加载CLIP模型和FAISS索引，请稍候……")
    searcher = HybridSearcher(config_path=str(config_path))
    detector = DuplicateDetector(phash_threshold=6)
    selector = BasicGroupSelector(
        duplicate_phash_threshold=6,
        minimum_semantic_score=0.0,
    )
    registry = GlobalCandidateRegistry(str(registry_path))

    query_vector = searcher.clip.encode_image(query_path)
    raw_results = searcher.vector_store.search(
        query_vector,
        top_k=args.top_k,
        threshold=0.0,
    )

    candidates = []
    missing_count = 0
    duplicate_count = 0
    globally_used_count = 0

    for candidate_path_text, semantic_score, internal_id in raw_results:
        candidate_path = Path(candidate_path_text).resolve()
        if not candidate_path.is_file():
            missing_count += 1
            continue

        duplicate_result = detector.compare(query_path, candidate_path)
        if duplicate_result["is_duplicate"]:
            duplicate_count += 1
            continue

        if registry.is_used(candidate_path):
            globally_used_count += 1
            continue

        candidates.append(
            {
                "path": str(candidate_path),
                "internal_id": internal_id,
                "semantic_score": float(semantic_score),
                "phash_distance": int(duplicate_result["phash_distance"]),
                "is_duplicate": False,
            }
        )

    selected = selector.select(candidates)

    print("\n" + "=" * 70)
    print("真实检索联调结果")
    print("=" * 70)
    print(f"样图：{query_path}")
    print(f"FAISS召回：{len(raw_results)}")
    print(f"路径失效：{missing_count}")
    print(f"样图重复：{duplicate_count}")
    print(f"已被全局占用：{globally_used_count}")
    print(f"进入分组候选：{len(candidates)}")
    print("-" * 70)

    selected_count = 0
    for group in ("A", "B", "C"):
        item = selected[group]
        if item is None:
            print(f"{group}：候选不足")
            continue

        reserved = registry.reserve(
            image_path=Path(item["path"]),
            sample_sequence=1,
            sample_name=query_path.name,
            group=group,
        )
        if not reserved:
            print(f"{group}：占用冲突，请重新运行")
            continue

        selected_count += 1
        print(
            f"{group}：{Path(item['path']).name} | "
            f"CLIP原始分={item['semantic_score']:.6f} | "
            f"pHash距离={item['phash_distance']}"
        )

    print("-" * 70)
    print(f"成功选择：{selected_count}/3")
    print(f"全局占用总数：{registry.count()}")
    print(f"测试登记文件：{registry_path}")
    print("=" * 70)

    return 0 if selected_count == 3 else 2


if __name__ == "__main__":
    sys.exit(main())
