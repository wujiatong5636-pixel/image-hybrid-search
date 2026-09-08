"""诊断样图和当前A/B/C结果的自动分类。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_searcher import HybridSearcher
from src.image_classifier import ImageCategoryClassifier


def main():
    parser = argparse.ArgumentParser(description="图片类别诊断")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--task_file", default=str(PROJECT_ROOT / "runtime/tasks/batch_task.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runtime/reports/category_diagnosis.json"))
    args = parser.parse_args()

    task_file = Path(args.task_file).resolve()
    if not task_file.is_file():
        print(f"[错误] 任务文件不存在：{task_file}")
        return 1

    print("正在加载CLIP模型并建立类别文本向量，请稍候……")
    searcher = HybridSearcher(config_path=str(Path(args.config).resolve()))
    classifier = ImageCategoryClassifier(searcher.clip)
    data = json.loads(task_file.read_text(encoding="utf-8"))
    tasks = sorted(data["tasks"].values(), key=lambda item: item["sequence"])
    rows = []

    for task in tasks:
        sample_result = classifier.classify_image(task["source_path"])
        row = {
            "sequence": task["sequence"],
            "sample_name": task["source_name"],
            "sample": sample_result,
            "results": {},
        }
        print(f"\n{task['sequence']:04d} {task['source_name']} → {sample_result['category']} ({sample_result['score']:.4f})")
        for group in "ABC":
            result = task["results"][group]
            predicted = classifier.classify_image(result["candidate_path"])
            row["results"][group] = {
                "candidate_name": Path(result["candidate_path"]).name,
                "prediction": predicted,
                "same_as_sample": predicted["category"] == sample_result["category"],
            }
            top_text = ", ".join(f"{x['category']}={x['score']:.3f}" for x in predicted["top"])
            print(f"  {group} {Path(result['candidate_path']).name} → {top_text}")
        rows.append(row)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    same = sum(item["same_as_sample"] for row in rows for item in row["results"].values())
    print(f"\n严格同类别：{same}/30")
    print(f"诊断文件：{output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
