r"""
验证降级触发逻辑：本地有效候选少于3张时是否调用百度。
用法: .venv\Scripts\python.exe scripts\test_provider_fallback.py --input_dir <样图目录>
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.hybrid_searcher import HybridSearcher
from src.providers.local_faiss_provider import LocalFAISSProvider
from src.providers.baidu_image_provider import BaiduImageProvider
from src.duplicate_detector import DuplicateDetector
from src.image_classifier import ImageCategoryClassifier, categories_compatible
from src.candidate_registry import GlobalCandidateRegistry


def filter_local_candidates(sample_path, raw_candidates, detector, classifier,
                            sample_category, registry, clip, vector_store,
                            absolute_min=0.50, relative_ratio=0.70):
    """模拟批量脚本中的本地候选过滤逻辑。"""
    candidates = []
    for cand in raw_candidates:
        cand_path = Path(cand.local_path)
        if not cand_path.is_file() or registry.is_used(cand_path):
            continue
        check = detector.compare(sample_path, cand_path)
        if check["is_duplicate"]:
            continue
        try:
            candidate_vector = vector_store.index.reconstruct(cand.internal_id)
        except Exception:
            candidate_vector = clip.encode_image(cand_path)
        candidate_prediction = classifier.classify_embedding(candidate_vector)
        candidate_category = candidate_prediction["category"]
        if not categories_compatible(sample_category, candidate_category):
            continue
        candidates.append({
            "path": str(cand_path),
            "semantic_score": cand.source_score or 0,
            "phash_distance": check["phash_distance"],
            "category": candidate_category,
            "source": "local",
        })

    if not candidates:
        return []

    best_score = max(c["semantic_score"] for c in candidates)
    effective_min = max(absolute_min, best_score * relative_ratio)
    return [c for c in candidates if c["semantic_score"] >= effective_min]


def main():
    parser = argparse.ArgumentParser(description="测试提供器降级逻辑")
    parser.add_argument("--input_dir", required=True, help="样图目录")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--test_local_limit", type=int, default=0,
                        help="测试用：强制限制本地候选数量（0表示不限制）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    provider_cfg = config.get("search_provider", {})
    trigger_count = provider_cfg.get("fallback_trigger_count", 3)
    local_top_k = provider_cfg.get("local_top_k", 98)

    print("=" * 60)
    print("方案D 降级触发逻辑测试")
    print(f"降级触发阈值: 本地有效候选 < {trigger_count}")
    print(f"本地Top-K: {local_top_k}")
    print("=" * 60)

    # 初始化
    searcher = HybridSearcher(config_path=args.config)
    local_provider = LocalFAISSProvider(searcher.clip, searcher.vector_store)
    baidu_provider = BaiduImageProvider(config.get("baidu_image_search", {}))
    detector = DuplicateDetector(phash_threshold=6)
    classifier = ImageCategoryClassifier(searcher.clip)
    registry = GlobalCandidateRegistry(str(PROJECT_ROOT / "runtime/tasks/used_candidates.json"))

    # 取第一张样图测试
    samples = sorted([p for p in Path(args.input_dir).iterdir()
                      if p.is_file() and p.suffix.lower() in
                      {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])
    if not samples:
        print("[错误] 样图目录为空")
        return 1

    sample_path = str(samples[0])
    print(f"\n测试样图: {samples[0].name}")

    # 1. 本地检索
    print("\n[步骤1] 本地FAISS检索...")
    raw_local = local_provider.search(sample_path, top_k=local_top_k)
    print(f"  FAISS原始返回: {len(raw_local)} 条")

    # 2. 过滤
    print("[步骤2] 执行过滤...")
    query_vector = searcher.clip.encode_image(sample_path)
    sample_pred = classifier.classify_embedding(query_vector)
    filtered = filter_local_candidates(
        sample_path, raw_local, detector, classifier,
        sample_pred["category"], registry, searcher.clip, searcher.vector_store
    )
    print(f"  过滤后有效候选: {len(filtered)} 条")

    # 测试限制
    if args.test_local_limit > 0:
        filtered = filtered[:args.test_local_limit]
        print(f"  [测试限制] 强制限制为: {len(filtered)} 条")

    # 3. 判断是否触发百度
    print(f"\n[步骤3] 降级判断: {len(filtered)} < {trigger_count} ?")
    if len(filtered) < trigger_count:
        print("  -> 是，触发百度识图")
        if not baidu_provider.is_available:
            print("  [警告] 百度提供器不可用（熔断器可能已打开）")
            return 1
        baidu_candidates = baidu_provider.search(sample_path, top_k=10)
        print(f"  百度返回: {len(baidu_candidates)} 个URL")
        if baidu_candidates:
            print("  [成功] 降级触发正常")
        else:
            print("  [警告] 百度返回为空（可能网络问题或反爬）")
    else:
        print("  -> 否，不触发百度，直接使用本地候选")
        print("  [成功] 本地充足时不调用百度")

    print("\n" + "=" * 60)
    print("测试完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
