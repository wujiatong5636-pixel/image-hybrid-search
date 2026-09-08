"""第十八步：自动审核正式批量检索结果。"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.result_auditor import ResultAuditor


def main():
    parser = argparse.ArgumentParser(description="自动审核批量图片匹配结果")
    parser.add_argument(
        "--task_file",
        default=str(PROJECT_ROOT / "runtime/tasks/batch_task.json"),
    )
    parser.add_argument(
        "--report_file",
        default=str(PROJECT_ROOT / "runtime/reports/audit_report.json"),
    )
    parser.add_argument("--min_samples", type=int, default=10)
    args = parser.parse_args()

    task_file = Path(args.task_file).resolve()
    report_file = Path(args.report_file).resolve()
    if not task_file.is_file():
        print(f"[错误] 任务文件不存在：{task_file}")
        return 1

    try:
        data = json.loads(task_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"[错误] 无法读取任务文件：{error}")
        return 1

    tasks = list(data.get("tasks", {}).values())
    result = ResultAuditor(min_samples=args.min_samples).audit(tasks)
    result["audited_at"] = datetime.now().isoformat(timespec="seconds")
    result["task_file"] = str(task_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print("批量检索自动审核结果")
    print("=" * 64)
    print(f"审核结论：{'通过' if result['passed'] else '不通过'}")
    print(f"样图任务：{result['sample_count']}")
    print(f"完成任务：{result['completed_count']}")
    print(f"输出结果：{result['result_count']}")
    print(f"唯一候选：{result['unique_candidate_count']}")
    for group, values in result["group_stats"].items():
        print(
            f"{group}组：CLIP均值={values['semantic_mean']:.4f} | "
            f"构图均值={values['composition_mean']:.4f} | "
            f"多维均值={values['multi_feature_mean']:.4f}"
        )
    for message in result["errors"]:
        print(f"[不通过] {message}")
    for message in result["warnings"]:
        print(f"[提醒] {message}")
    print(f"分数说明：{result['score_note']}")
    print(f"审核报告：{report_file}")
    print("=" * 64)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
