"""正式批量执行本地 CLIP 候选选择并保存 A、B、C 结果。"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_registry import GlobalCandidateRegistry
from src.duplicate_detector import DuplicateDetector
from src.hybrid_searcher import HybridSearcher
from src.image_classifier import ImageCategoryClassifier, categories_compatible
from src.multi_feature_scorer import MultiFeatureScorer
from src.task_store import TaskStore

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [(0, int(x)) if x.isdigit() else (1, x.casefold()) for x in parts]


def collect_images(directory: Path):
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=natural_sort_key,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: str) -> str:
    stem = Path(name).stem
    value = re.sub(r'[<>:"/\\|?*]+', "_", stem).strip(" .")
    return value or "sample"


def main() -> int:
    parser = argparse.ArgumentParser(description="正式批量图片基础匹配")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument(
        "--absolute_min_score",
        type=float,
        default=0.50,
        help="候选CLIP原始分的绝对最低值",
    )
    parser.add_argument(
        "--relative_min_ratio",
        type=float,
        default=0.70,
        help="候选分数至少达到当前第一名的比例",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    registry_path = PROJECT_ROOT / "runtime/tasks/used_candidates.json"
    task_path = PROJECT_ROOT / "runtime/tasks/batch_task.json"

    if not input_dir.is_dir():
        print(f"[错误] 样图目录不存在：{input_dir}")
        return 1
    if not config_path.is_file():
        print(f"[错误] 配置文件不存在：{config_path}")
        return 1
    if args.top_k < 3:
        print("[错误] top_k至少为3")
        return 1
    if not 0.0 <= args.absolute_min_score <= 1.0:
        print("[错误] absolute_min_score必须在0到1之间")
        return 1
    if not 0.0 < args.relative_min_ratio <= 1.0:
        print("[错误] relative_min_ratio必须大于0且不超过1")
        return 1

    samples = collect_images(input_dir)
    if len(samples) < 10:
        print(f"[错误] 当前只有{len(samples)}张样图，至少需要10张")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    registry = GlobalCandidateRegistry(str(registry_path))
    task_store = TaskStore(str(task_path))
    detector = DuplicateDetector(phash_threshold=6)
    selector = MultiFeatureScorer()

    print("正在加载CLIP模型和FAISS索引，请稍候……")
    searcher = HybridSearcher(config_path=str(config_path))
    if len(searcher.vector_store) == 0:
        print("[错误] FAISS索引为空，请先构建候选图库索引")
        return 1
    classifier = ImageCategoryClassifier(searcher.clip)

    completed = 0
    skipped = 0
    failed = 0

    for sequence, sample_path in enumerate(samples, start=1):
        source_hash = sha256_file(sample_path)
        previous = task_store.get(source_hash)
        if args.resume and previous and task_store.outputs_exist(previous):
            print(f"[{sequence}/{len(samples)}] 跳过已完成：{sample_path.name}")
            skipped += 1
            continue

        print(f"[{sequence}/{len(samples)}] 正在处理：{sample_path.name}")
        task_store.update(
            source_hash,
            {
                "sequence": sequence,
                "source_name": sample_path.name,
                "source_path": str(sample_path),
                "status": "processing",
                "results": {},
            },
        )

        try:
            query_vector = searcher.clip.encode_image(sample_path)
            sample_prediction = classifier.classify_embedding(query_vector)
            sample_category = sample_prediction["category"]
            raw_results = searcher.vector_store.search(
                query_vector,
                top_k=args.top_k,
                threshold=0.0,
            )
            candidates = []
            category_rejected = 0
            for candidate_text, score, internal_id in raw_results:
                candidate_path = Path(candidate_text).resolve()
                if not candidate_path.is_file() or registry.is_used(candidate_path):
                    continue
                check = detector.compare(sample_path, candidate_path)
                if check["is_duplicate"]:
                    continue
                try:
                    candidate_vector = searcher.vector_store.index.reconstruct(internal_id)
                except Exception:
                    candidate_vector = searcher.clip.encode_image(candidate_path)
                candidate_prediction = classifier.classify_embedding(candidate_vector)
                candidate_category = candidate_prediction["category"]
                if not categories_compatible(sample_category, candidate_category):
                    category_rejected += 1
                    continue
                candidates.append(
                    {
                        "path": str(candidate_path),
                        "internal_id": internal_id,
                        "semantic_score": float(score),
                        "phash_distance": int(check["phash_distance"]),
                        "is_duplicate": False,
                        "category": candidate_category,
                        "category_score": float(candidate_prediction["score"]),
                    }
                )

            if not candidates:
                raise RuntimeError("重复过滤后没有可用候选")

            best_score = max(item["semantic_score"] for item in candidates)
            effective_min_score = max(
                args.absolute_min_score,
                best_score * args.relative_min_ratio,
            )
            candidates = [
                item
                for item in candidates
                if item["semantic_score"] >= effective_min_score
            ]

            selected = selector.select(sample_path, candidates)
            if any(selected[group] is None for group in ("A", "B", "C")):
                raise RuntimeError(
                    f"质量门槛={effective_min_score:.4f}，"
                    f"过滤后只有{len(candidates)}张合格候选，无法输出3张"
                )

            sample_output = output_dir / f"{sequence:04d}_{safe_name(sample_path.name)}"
            sample_output.mkdir(parents=True, exist_ok=True)
            source_output = sample_output / f"source{sample_path.suffix.lower()}"
            shutil.copy2(sample_path, source_output)

            saved_results = {}
            reserved_paths = []
            try:
                for group in ("A", "B", "C"):
                    item = selected[group]
                    candidate_path = Path(item["path"])
                    if not registry.reserve(
                        candidate_path,
                        sample_sequence=sequence,
                        sample_name=sample_path.name,
                        group=group,
                    ):
                        raise RuntimeError(f"{group}组候选发生全局占用冲突")
                    reserved_paths.append(candidate_path)
                    destination = sample_output / f"{sequence:04d}_{group}{candidate_path.suffix.lower()}"
                    shutil.copy2(candidate_path, destination)
                    saved_results[group] = {
                        "candidate_path": str(candidate_path),
                        "output_path": str(destination.resolve()),
                        "semantic_score": round(item["semantic_score"], 6),
                        "phash_distance": item["phash_distance"],
                        "category": item["category"],
                        "category_score": round(item["category_score"], 6),
                        "composition_similarity": round(item["composition_similarity"], 6),
                        "shape_similarity": round(item["shape_similarity"], 6),
                        "color_similarity": round(item["color_similarity"], 6),
                        "texture_similarity": round(item["texture_similarity"], 6),
                        "multi_feature_score": round(item["multi_feature_score"], 6),
                    }
            except Exception:
                for reserved_path in reserved_paths:
                    registry.release(reserved_path)
                raise

            task_store.update(
                source_hash,
                {
                    "sequence": sequence,
                    "source_name": sample_path.name,
                    "source_path": str(sample_path),
                    "source_output": str(source_output.resolve()),
                    "status": "completed",
                    "sample_category": sample_category,
                    "sample_category_score": round(sample_prediction["score"], 6),
                    "category_rejected": category_rejected,
                    "effective_min_score": round(effective_min_score, 6),
                    "results": saved_results,
                },
            )
            completed += 1
            print(
                f"    完成：A={Path(saved_results['A']['candidate_path']).name} | "
                f"B={Path(saved_results['B']['candidate_path']).name} | "
                f"C={Path(saved_results['C']['candidate_path']).name}"
            )
        except Exception as error:
            failed += 1
            task_store.update(
                source_hash,
                {
                    "sequence": sequence,
                    "source_name": sample_path.name,
                    "source_path": str(sample_path),
                    "status": "retryable_error",
                    "error": str(error),
                    "results": {},
                },
            )
            print(f"    [失败] {error}")

    summary = {
        "total": len(samples),
        "completed_this_run": completed,
        "skipped": skipped,
        "failed": failed,
        "global_used": registry.count(),
        "task_file": str(task_path),
        "registry_file": str(registry_path),
    }
    print("\n" + "=" * 70)
    print("批量任务完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 70)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
