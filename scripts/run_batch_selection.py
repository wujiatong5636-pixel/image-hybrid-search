"""正式批量执行本地 CLIP 候选选择并保存 A、B、C 结果。

1.2版：本地CLIP+FAISS作为主检索；本地不足3张时调用百度；
百度不可用、无结果或过滤后仍不足3张时，再调用Google Vision；
所有网络候选下载到本地后重新执行CLIP和多维评分，不采信网络排名。
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_registry import GlobalCandidateRegistry
from src.candidate_pool import CandidatePool
from src.duplicate_detector import DuplicateDetector
from src.fallback_policy import needs_fallback
from src.hybrid_searcher import HybridSearcher
from src.image_classifier import ImageCategoryClassifier, categories_compatible
from src.multi_feature_scorer import MultiFeatureScorer
from src.providers.baidu_image_provider import BaiduImageProvider
from src.providers.google_vision_provider import GoogleVisionProvider
from src.providers.local_faiss_provider import LocalFAISSProvider
from src.remote_candidate_cache import RemoteCandidateCache
from src.task_store import TaskStore

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_batch")
logger.setLevel(logging.INFO)

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


def score_remote_candidates(
    raw_candidates,
    remote_cache,
    max_downloads,
    searcher,
    query_vector,
    classifier,
    sample_category,
    effective_min_score,
    source,
):
    """下载网络候选并使用本地模型统一评分和过滤。"""
    urls = [item.source_url for item in raw_candidates if item.source_url]
    downloads = remote_cache.download_batch(urls, max_downloads=max_downloads)
    candidates = []
    for rank, item in enumerate(downloads, start=1):
        path = item["local_path"]
        try:
            vector = searcher.clip.encode_image(path)
            score = float(searcher.clip.similarity(query_vector, vector))
            prediction = classifier.classify_embedding(vector)
            if not categories_compatible(sample_category, prediction["category"]):
                continue
            if score < effective_min_score:
                continue
            candidates.append(
                {
                    "path": path,
                    "semantic_score": score,
                    "phash_distance": 0,
                    "is_duplicate": False,
                    "category": prediction["category"],
                    "category_score": float(prediction["score"]),
                    "source": source,
                    "source_url": item["url"],
                    "source_rank": rank,
                }
            )
        except Exception as error:
            logger.warning("%s候选评分失败 %s: %s", source, path, error)
    return candidates, len(downloads)


def main() -> int:
    parser = argparse.ArgumentParser(description="正式批量图片匹配（本地→百度→Google）")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--top_k", type=int, default=98)
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
    parser.add_argument(
        "--test_local_limit",
        type=int,
        default=0,
        help="测试用：强制限制本地有效候选数量（0表示不限制）",
    )
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

    # ===== 1.2版：初始化三级检索提供器 =====
    provider_cfg = searcher.config.get("search_provider", {})
    fallback_trigger_count = provider_cfg.get("fallback_trigger_count", 3)
    local_top_k = provider_cfg.get("local_top_k", 98)
    local_provider = LocalFAISSProvider(searcher.clip, searcher.vector_store)
    baidu_provider = BaiduImageProvider(searcher.config.get("baidu_image_search", {}))
    baidu_cfg = searcher.config.get("baidu_image_search", {})
    google_cfg = searcher.config.get("google_fallback", {})
    google_provider = GoogleVisionProvider(
        searcher.vision,
        searcher.rate_limiter,
        google_cfg,
    )
    remote_cfg = searcher.config.get("remote_download", {})
    remote_cache = RemoteCandidateCache(
        cache_dir=remote_cfg.get("cache_dir", "runtime/remote_candidate_cache"),
        timeout=remote_cfg.get("timeout", 15),
        retries=remote_cfg.get("retries", 2),
        max_file_mb=remote_cfg.get("max_file_mb", 15),
        min_width=remote_cfg.get("min_width", 200),
        min_height=remote_cfg.get("min_height", 200),
    )
    candidate_pool = CandidatePool(detector, registry)
    baidu_max_downloads = baidu_cfg.get("max_downloads", 20)
    google_max_downloads = google_cfg.get("max_downloads", 20)

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

        # 统计字段初始化（保证异常分支也能引用）
        baidu_called = False
        baidu_url_count = 0
        baidu_downloaded_count = 0
        baidu_valid_count = 0
        google_called = False
        google_status = "not_called"
        google_url_count = 0
        google_downloaded_count = 0
        google_valid_count = 0
        local_valid_count = 0
        low_diversity_pool = False
        category_rejected = 0
        effective_min_score = args.absolute_min_score

        try:
            query_vector = searcher.clip.encode_image(sample_path)
            sample_prediction = classifier.classify_embedding(query_vector)
            sample_category = sample_prediction["category"]

            # ===== 1. 本地FAISS检索 =====
            raw_local = local_provider.search(str(sample_path), top_k=local_top_k)

            # ===== 2. 本地候选五道过滤：路径/全局占用/pHash重复/类别/（质量门槛稍后） =====
            candidates = []
            for cand in raw_local:
                candidate_path = Path(cand.local_path).resolve()
                if not candidate_path.is_file() or registry.is_used(candidate_path):
                    continue
                check = detector.compare(sample_path, candidate_path)
                if check["is_duplicate"]:
                    continue
                try:
                    candidate_vector = searcher.vector_store.index.reconstruct(cand.internal_id)
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
                        "internal_id": cand.internal_id,
                        "semantic_score": cand.source_score or 0.0,
                        "phash_distance": int(check["phash_distance"]),
                        "is_duplicate": False,
                        "category": candidate_category,
                        "category_score": float(candidate_prediction["score"]),
                        "source": "local",
                        "source_url": None,
                        "source_rank": cand.source_rank,
                    }
                )

            # 测试用：强制限制本地候选数量
            if args.test_local_limit > 0:
                candidates = candidates[:args.test_local_limit]

            # 动态质量门槛
            if candidates:
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

            local_valid_count = len(candidates)
            print(f"    本地有效候选：{local_valid_count} 张（质量门槛={effective_min_score:.4f}）")

            # ===== 3. 本地不足时先百度，百度补充后仍不足再Google =====
            if needs_fallback(local_valid_count, fallback_trigger_count):
                baidu_called = True
                missing_count = fallback_trigger_count - local_valid_count
                download_target = max(10, missing_count * 5)
                print(f"    本地不足{fallback_trigger_count}张（缺{missing_count}），触发百度识图，计划下载{download_target}张……")

                try:
                    baidu_raw = baidu_provider.search(str(sample_path), top_k=baidu_cfg.get("max_results", 30))
                    baidu_url_count = len(baidu_raw)
                    print(f"    百度返回URL：{baidu_url_count} 个")

                    if baidu_raw:
                        baidu_candidates, baidu_downloaded_count = score_remote_candidates(
                            baidu_raw,
                            remote_cache,
                            max_downloads=min(download_target, baidu_max_downloads),
                            searcher=searcher,
                            query_vector=query_vector,
                            classifier=classifier,
                            sample_category=sample_category,
                            effective_min_score=effective_min_score,
                            source="baidu",
                        )
                        print(f"    百度下载成功：{baidu_downloaded_count} 张")
                        baidu_valid_count = len(baidu_candidates)
                        print(f"    百度过滤后有效：{baidu_valid_count} 张")
                        if baidu_candidates:
                            candidates = candidate_pool.merge_and_deduplicate(
                                str(sample_path), candidates, baidu_candidates
                            )
                            print(f"    合并去重后总候选：{len(candidates)} 张")

                except Exception as e:
                    print(f"    [警告] 百度识图调用失败：{e}")

                if needs_fallback(len(candidates), fallback_trigger_count):
                    google_called = True
                    print(f"    百度补充后仍只有{len(candidates)}张，触发Google Vision……")
                    try:
                        google_raw = google_provider.search(
                            str(sample_path),
                            top_k=google_cfg.get("max_results", 30),
                        )
                        google_status = google_provider.last_status
                        google_url_count = len(google_raw)
                        print(f"    Google状态：{google_status}，返回URL：{google_url_count} 个")
                        if google_raw:
                            missing_count = fallback_trigger_count - len(candidates)
                            download_target = max(10, missing_count * 5)
                            google_candidates, google_downloaded_count = score_remote_candidates(
                                google_raw,
                                remote_cache,
                                max_downloads=min(download_target, google_max_downloads),
                                searcher=searcher,
                                query_vector=query_vector,
                                classifier=classifier,
                                sample_category=sample_category,
                                effective_min_score=effective_min_score,
                                source="google",
                            )
                            google_valid_count = len(google_candidates)
                            print(
                                f"    Google下载成功：{google_downloaded_count} 张，"
                                f"过滤后有效：{google_valid_count} 张"
                            )
                            if google_candidates:
                                candidates = candidate_pool.merge_and_deduplicate(
                                    str(sample_path), candidates, google_candidates
                                )
                                print(f"    Google合并去重后总候选：{len(candidates)} 张")
                    except Exception as error:
                        google_status = "error"
                        print(f"    [警告] Google Vision调用失败：{error}")
            else:
                # 本地正好等于阈值时，标记低多样性提醒
                if local_valid_count == fallback_trigger_count:
                    low_diversity_pool = True
                    print(f"    本地候选正好{fallback_trigger_count}张，未调用百度（低多样性提醒）")

            if not candidates:
                raise RuntimeError("过滤后没有可用候选（本地、百度和Google均无有效结果）")

            # ===== 4. A/B/C多维评分选择 =====
            selected = selector.select(sample_path, candidates)
            if any(selected[group] is None for group in ("A", "B", "C")):
                raise RuntimeError(
                    f"质量门槛={effective_min_score:.4f}，"
                    f"最终只有{len(candidates)}张合格候选，无法输出3张"
                )

            # ===== 5. 全局占用并输出 =====
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
                        candidate_source=item.get("source", "local"),
                        candidate_url=item.get("source_url"),
                    ):
                        raise RuntimeError(f"{group}组候选发生全局占用冲突")
                    reserved_paths.append(candidate_path)
                    destination = sample_output / f"{sequence:04d}_{group}{candidate_path.suffix.lower()}"
                    shutil.copy2(candidate_path, destination)
                    saved_results[group] = {
                        "candidate_path": str(candidate_path),
                        "output_path": str(destination.resolve()),
                        "source": item.get("source", "local"),
                        "source_url": item.get("source_url"),
                        "source_rank": item.get("source_rank"),
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
                    "baidu_called": baidu_called,
                    "baidu_url_count": baidu_url_count,
                    "baidu_downloaded_count": baidu_downloaded_count,
                    "baidu_valid_count": baidu_valid_count,
                    "google_called": google_called,
                    "google_status": google_status,
                    "google_url_count": google_url_count,
                    "google_downloaded_count": google_downloaded_count,
                    "google_valid_count": google_valid_count,
                    "local_valid_count": local_valid_count,
                    "low_diversity_pool": low_diversity_pool,
                    "results": saved_results,
                },
            )
            completed += 1
            source_tags = {
                g: saved_results[g]["source"] for g in ("A", "B", "C")
            }
            print(
                f"    完成：A={Path(saved_results['A']['candidate_path']).name}[{source_tags['A']}] | "
                f"B={Path(saved_results['B']['candidate_path']).name}[{source_tags['B']}] | "
                f"C={Path(saved_results['C']['candidate_path']).name}[{source_tags['C']}]"
            )
            if low_diversity_pool:
                print("    [提醒] 本地候选正好3张，A/B/C分组空间较小，请人工复核")
        except Exception as error:
            failed += 1
            task_store.update(
                source_hash,
                {
                    "sequence": sequence,
                    "source_name": sample_path.name,
                    "source_path": str(sample_path),
                    "status": "insufficient_candidates" if "无法输出3张" in str(error) or "没有可用候选" in str(error) else "retryable_error",
                    "error": str(error),
                    "baidu_called": baidu_called,
                    "baidu_url_count": baidu_url_count,
                    "baidu_downloaded_count": baidu_downloaded_count,
                    "baidu_valid_count": baidu_valid_count,
                    "google_called": google_called,
                    "google_status": google_status,
                    "google_url_count": google_url_count,
                    "google_downloaded_count": google_downloaded_count,
                    "google_valid_count": google_valid_count,
                    "local_valid_count": local_valid_count,
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
