"""批量任务断点状态存储。"""

import json
import os
from datetime import datetime
from pathlib import Path


class TaskStore:
    """以样图内容哈希为键保存任务状态。"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.tasks = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.tasks = {}
            return
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        self.tasks = data.get("tasks", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "task_count": len(self.tasks),
            "tasks": self.tasks,
        }
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.path)

    def get(self, source_hash: str):
        value = self.tasks.get(source_hash)
        return dict(value) if value else None

    def update(self, source_hash: str, record: dict) -> None:
        value = dict(record)
        value["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.tasks[source_hash] = value
        self.save()

    @staticmethod
    def outputs_exist(record: dict) -> bool:
        results = record.get("results", {})
        if record.get("status") != "completed" or len(results) != 3:
            return False
        return all(Path(item["output_path"]).is_file() for item in results.values())
