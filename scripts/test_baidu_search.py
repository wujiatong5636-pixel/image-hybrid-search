r"""
单独测试百度识图是否可用。
用法: .venv\Scripts\python.exe scripts\test_baidu_search.py --image <样图路径>
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.providers.baidu_image_provider import BaiduImageProvider


def main():
    parser = argparse.ArgumentParser(description="测试百度识图")
    parser.add_argument("--image", required=True, help="样图路径")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    baidu_cfg = config.get("baidu_image_search", {})
    if not baidu_cfg.get("enabled", False):
        print("[错误] 百度识图未在配置中启用")
        return 1

    print(f"测试图片: {args.image}")
    print(f"模式: {baidu_cfg.get('mode', 'auto')}")
    print("-" * 50)

    provider = BaiduImageProvider(baidu_cfg)
    candidates = provider.search(args.image, top_k=10)

    if not candidates:
        print("[结果] 百度识图未返回任何结果")
        print("可能原因：")
        print("  1. 网络不通或被百度反爬")
        print("  2. HTTP模式解析失败，且未安装playwright")
        print("  3. 熔断器已打开（连续失败过多）")
        return 1

    print(f"[结果] 百度返回 {len(candidates)} 个URL:")
    for c in candidates:
        print(f"  #{c.source_rank}: {c.source_url[:80]}...")

    print("\n[成功] 百度识图可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
