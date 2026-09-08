"""全局候选去重测试。"""

import shutil
from pathlib import Path

from src.candidate_registry import GlobalCandidateRegistry


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_DIR = PROJECT_ROOT / "candidate_library"
RUNTIME_DIR = PROJECT_ROOT / "runtime" / "tasks"
TEST_REGISTRY = RUNTIME_DIR / "step11_test_registry.json"


def prepare_registry():
    """创建干净的测试登记表。"""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_REGISTRY.exists():
        TEST_REGISTRY.unlink()

    temp_file = TEST_REGISTRY.with_suffix(
        TEST_REGISTRY.suffix + ".tmp"
    )

    if temp_file.exists():
        temp_file.unlink()

    return GlobalCandidateRegistry(
        str(TEST_REGISTRY)
    )


def get_test_candidates():
    """取得两张真实候选图片。"""
    candidates = sorted(
        [
            path
            for path in CANDIDATE_DIR.iterdir()
            if path.is_file()
        ],
        key=lambda path: path.name,
    )

    assert len(candidates) >= 2
    return candidates[0], candidates[1]


def test_same_candidate_cannot_be_reserved_twice():
    """同一路径不能重复占用。"""
    registry = prepare_registry()
    candidate_a, _ = get_test_candidates()

    first_result = registry.reserve(
        image_path=candidate_a,
        sample_sequence=1,
        sample_name="1 (1).png",
        group="A",
    )

    second_result = registry.reserve(
        image_path=candidate_a,
        sample_sequence=2,
        sample_name="1 (2).png",
        group="B",
    )

    assert first_result is True
    assert second_result is False
    assert registry.count() == 1


def test_same_content_with_different_name_is_duplicate():
    """相同内容换文件名后仍然不能重复占用。"""
    registry = prepare_registry()
    candidate_a, _ = get_test_candidates()

    copied_candidate = (
        RUNTIME_DIR / "same_content_different_name.png"
    )

    shutil.copy2(candidate_a, copied_candidate)

    try:
        first_result = registry.reserve(
            image_path=candidate_a,
            sample_sequence=1,
            sample_name="1 (1).png",
            group="A",
        )

        second_result = registry.reserve(
            image_path=copied_candidate,
            sample_sequence=2,
            sample_name="1 (2).png",
            group="B",
        )

        assert first_result is True
        assert second_result is False
        assert registry.count() == 1
    finally:
        if copied_candidate.exists():
            copied_candidate.unlink()


def test_different_candidates_can_be_reserved():
    """不同内容的候选可以分别占用。"""
    registry = prepare_registry()
    candidate_a, candidate_b = get_test_candidates()

    assert registry.reserve(
        candidate_a,
        sample_sequence=1,
        sample_name="1 (1).png",
        group="A",
    ) is True

    assert registry.reserve(
        candidate_b,
        sample_sequence=1,
        sample_name="1 (1).png",
        group="B",
    ) is True

    assert registry.count() == 2


def test_registry_can_be_loaded_again():
    """重新创建对象后仍能恢复占用记录。"""
    registry = prepare_registry()
    candidate_a, _ = get_test_candidates()

    registry.reserve(
        candidate_a,
        sample_sequence=3,
        sample_name="1 (3).png",
        group="C",
    )

    restored_registry = GlobalCandidateRegistry(
        str(TEST_REGISTRY)
    )

    assert restored_registry.count() == 1
    assert restored_registry.is_used(candidate_a) is True

    record = restored_registry.get_record(candidate_a)

    assert record is not None
    assert record["sample_sequence"] == 3
    assert record["sample_name"] == "1 (3).png"
    assert record["group"] == "C"