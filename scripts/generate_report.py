"""根据正式批量任务生成报告和四宫格预览图。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.result_exporter import create_contact_sheet, export_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="生成批量匹配报告")
    parser.add_argument(
        "--task_file",
        default=str(PROJECT_ROOT / "runtime/tasks/batch_task.json"),
    )
    parser.add_argument(
        "--report_dir",
        default=str(PROJECT_ROOT / "runtime/reports"),
    )
    args = parser.parse_args()

    task_file = Path(args.task_file).resolve()
    report_dir = Path(args.report_dir).resolve()
    if not task_file.is_file():
        print(f"[错误] 正式任务文件不存在：{task_file}")
        return 1

    data = json.loads(task_file.read_text(encoding="utf-8"))
    tasks = sorted(data.get("tasks", {}).values(), key=lambda item: item["sequence"])
    completed = [task for task in tasks if task.get("status") == "completed"]
    if not completed:
        print("[错误] 没有已完成任务")
        return 1

    for task in completed:
        if set(task.get("results", {})) != set("ABC"):
            print(f"[错误] {task.get('source_name')} 缺少A/B/C结果")
            return 1
        preview_path = report_dir / "previews" / f"{task['sequence']:04d}_preview.jpg"
        create_contact_sheet(task, preview_path)
        print(f"预览图已生成：{preview_path.name}")

    result = export_reports(completed, report_dir)
    print("\n" + "=" * 60)
    print("报告生成完成")
    print(f"完成任务：{len(completed)}")
    print(f"明细行数：{result['rows']}")
    print(f"CSV：{result['csv']}")
    print(f"Excel：{result['excel']}")
    print(f"JSON：{result['json']}")
    print(f"预览图：{report_dir / 'previews'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
