"""
演示脚本：展示混合检索系统的完整使用流程

用法：
  python scripts/demo.py --query /path/to/query.jpg --config config.yaml
  python scripts/demo.py --text "一只猫在沙发上" --config config.yaml

功能：
  - 以图搜图（CLIP + Vision 混合）
  - 以文搜图（纯 CLIP）
  - 打印详细结果和来源标注
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_searcher import HybridSearcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("demo")


def print_results(result: dict):
    """格式化打印检索结果。"""
    print("\n" + "=" * 70)
    print(f"查询: {result['query']}")
    print(f"模式: {result['mode']}")
    print(f"返回 {len(result['results'])} 条结果")
    print("-" * 70)

    source_labels = {
        "both": "混合(CLIP+Vision)",
        "clip_image": "CLIP图像",
        "clip_text": "CLIP文本",
        "vision": "Vision全网",
    }

    for i, item in enumerate(result["results"], 1):
        source = source_labels.get(item["source"], item["source"])
        print(f"  #{i:2d}  分数: {item['score']:.4f}  来源: {source}")
        print(f"       {item['id']}")

    # Vision 详细信息
    if result.get("vision_result"):
        vr = result["vision_result"]
        print("\n--- Vision 全网溯源详情 ---")
        print(f"  完全匹配: {len(vr['full_matches'])} 条")
        for m in vr["full_matches"][:3]:
            print(f"    {m['url']}")
        print(f"  部分匹配: {len(vr['partial_matches'])} 条")
        print(f"  视觉相似: {len(vr['similar_images'])} 条")
        print(f"  包含网页: {len(vr['pages_with_matches'])} 个")
        if vr["best_guess_labels"]:
            labels = [l["label"] for l in vr["best_guess_labels"]]
            print(f"  最佳标签: {', '.join(labels)}")
        if vr["web_entities"]:
            entities = [e["description"] for e in vr["web_entities"][:5]]
            print(f"  相关实体: {', '.join(entities)}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="混合图片检索演示")
    parser.add_argument("--query", type=str, help="查询图片路径（以图搜图）")
    parser.add_argument("--text", type=str, help="查询文本（以文搜图）")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--top_k", type=int, default=10, help="返回结果数量")
    parser.add_argument(
        "--no_vision",
        action="store_true",
        help="禁用 Vision（纯 CLIP 模式）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出结果",
    )
    args = parser.parse_args()

    if not args.query and not args.text:
        parser.error("请指定 --query (图片) 或 --text (文本)")

    searcher = HybridSearcher(config_path=args.config)
    logger.info("索引总量: %d 条", len(searcher.vector_store))

    if args.query:
        if not Path(args.query).exists():
            logger.error("查询图片不存在: %s", args.query)
            sys.exit(1)
        result = searcher.search_by_image(
            image_path=args.query,
            top_k=args.top_k,
            use_vision=not args.no_vision,
        )
    else:
        result = searcher.search_by_text(text=args.text, top_k=args.top_k)

    if args.json:
        # JSON 输出时去掉 vision_result 中的复杂对象
        output = {k: v for k, v in result.items() if k != "vision_result"}
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_results(result)


if __name__ == "__main__":
    main()
