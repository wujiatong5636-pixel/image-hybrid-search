"""
批量图片匹配入口。

第一阶段第七步功能：
1. 接收样图目录、候选目录和输出目录；
2. 检查目录是否合法；
3. 检查样图数量是否不少于10张；
4. 显示本次任务参数；
5. 为后续批量检索提供统一入口。
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List


# 当前支持的图片格式
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

def natural_sort_key(path: Path):
    """
    按照文件名中的数字进行自然排序。

    示例：
    1 (1).png
    1 (2).png
    1 (10).png
    """
    parts = re.split(r"(\d+)", path.name)

    return [
        (0, int(part)) if part.isdigit()
        else (1, part.casefold())
        for part in parts
    ]
def collect_images(directory: Path) -> List[Path]:
    """读取目录中的图片文件，不递归读取子目录。"""
    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def check_directory(directory: Path, description: str) -> None:
    """检查目录是否存在。"""
    if not directory.exists():
        raise FileNotFoundError(
            f"{description}不存在：{directory.resolve()}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"{description}不是文件夹：{directory.resolve()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量图片匹配程序"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="需要检索的样图目录",
    )

    parser.add_argument(
        "--candidate_dir",
        required=True,
        help="候选图片库目录",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="最终结果输出目录",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="项目配置文件路径",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="每张样图初步召回的候选数量",
    )

    parser.add_argument(
        "--min_input_count",
        type=int,
        default=10,
        help="批量任务要求的最少样图数量",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有断点继续任务",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()

    try:
        check_directory(input_dir, "样图目录")
        check_directory(candidate_dir, "候选图库")

        if input_dir == candidate_dir:
            raise ValueError(
                "样图目录不能和候选图库使用同一个目录"
            )

        if not config_path.exists():
            raise FileNotFoundError(
                f"配置文件不存在：{config_path}"
            )

        if args.top_k <= 0:
            raise ValueError("top_k必须大于0")

        if args.min_input_count <= 0:
            raise ValueError("min_input_count必须大于0")

        sample_images = collect_images(input_dir)
        candidate_images = collect_images(candidate_dir)

        if len(sample_images) < args.min_input_count:
            raise ValueError(
                f"当前只有{len(sample_images)}张样图，"
                f"批量任务至少需要{args.min_input_count}张"
            )

        if not candidate_images:
            raise ValueError("候选图库中没有可用图片")

        output_dir.mkdir(parents=True, exist_ok=True)

    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        PermissionError,
    ) as error:
        print(f"\n[错误] {error}")
        return 1

    print("\n" + "=" * 60)
    print("批量图片匹配任务检查通过")
    print("=" * 60)
    print(f"样图目录：       {input_dir}")
    print(f"样图数量：       {len(sample_images)}")
    print(f"候选图库：       {candidate_dir}")
    print(f"候选图片数量：   {len(candidate_images)}")
    print(f"输出目录：       {output_dir}")
    print(f"配置文件：       {config_path}")
    print(f"每图候选数量：   {args.top_k}")
    print(f"断点恢复：       {'开启' if args.resume else '关闭'}")
    print("检索方式：       本地CLIP + FAISS")
    print("-" * 60)

    print("当前识别到的样图：")

    for index, image_path in enumerate(
        sorted(sample_images, key=natural_sort_key),
        start=1,
    ):
        print(f"{index:04d}  {image_path.name}")

    print("-" * 60)
    print("第七步检查完成，本步骤暂不执行模型检索。")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
