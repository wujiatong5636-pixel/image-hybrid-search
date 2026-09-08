"""批量任务断点状态测试。"""

from pathlib import Path

from src.task_store import TaskStore


def test_task_store_save_and_load(tmp_path):
    path = tmp_path / "task.json"
    store = TaskStore(str(path))
    store.update("hash1", {"status": "processing", "results": {}})
    restored = TaskStore(str(path))
    assert restored.get("hash1")["status"] == "processing"


def test_completed_requires_three_existing_outputs(tmp_path):
    outputs = {}
    for group in "ABC":
        path = tmp_path / f"{group}.jpg"
        path.write_bytes(group.encode())
        outputs[group] = {"output_path": str(path)}
    record = {"status": "completed", "results": outputs}
    assert TaskStore.outputs_exist(record) is True


def test_missing_output_is_not_completed(tmp_path):
    record = {
        "status": "completed",
        "results": {
            "A": {"output_path": str(tmp_path / "A.jpg")},
            "B": {"output_path": str(tmp_path / "B.jpg")},
            "C": {"output_path": str(tmp_path / "C.jpg")},
        },
    }
    assert TaskStore.outputs_exist(record) is False
