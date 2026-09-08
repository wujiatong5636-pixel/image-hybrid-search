"""
批量构建向量索引脚本

用法：
  python scripts/build_index.py --image_dir /path/to/images --config config.yaml

功能：
  - 递归扫描指定目录下的所有图片
  - 批量 CLIP 编码
  - IVF 模式自动训练索引
  - 持久化索引和 ID 映射
  - 支持断点续传（已存在路径的图片跳过编码）
"""

import argparse
import logging
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_searcher import HybridSearcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_index")


def main():
    parser = argparse.ArgumentParser(description="批量构建图片向量索引")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="图片目录路径（递归扫描所有子目录）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="CLIP 编码批大小",
    )
    args = parser.parse_args()

    if not Path(args.image_dir).exists():
        logger.error("图片目录不存在: %s", args.image_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("开始构建索引")
    logger.info("图片目录: %s", args.image_dir)
    logger.info("配置文件: %s", args.config)
    logger.info("批大小: %d", args.batch_size)
    logger.info("=" * 60)

    searcher = HybridSearcher(config_path=args.config)
    searcher.build_index(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
    )

    usage = searcher.rate_limiter.get_usage()
    logger.info("索引构建完成！总量: %d 条", len(searcher.vector_store))
    logger.info("Vision 用量: %d/%d (%.1f%%)",
                usage["used"], usage["max"], usage["usage_rate"] * 100)


if __name__ == "__main__":
    main()
