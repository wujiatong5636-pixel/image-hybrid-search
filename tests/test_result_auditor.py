"""批量结果自动审核测试。"""

from PIL import Image

from src.result_auditor import ResultAuditor


def make_task(tmp_path, sequence, duplicate_path=False):
    results = {}
    for group in "ABC":
        path = tmp_path / ("same.png" if duplicate_path else f"{sequence}_{group}.png")
        Image.new("RGB", (8, 8), "white").save(path)
        results[group] = {
            "candidate_path": str(path),
            "output_path": str(path),
            "semantic_score": 0.8,
            "phash_distance": 20,
            "category": "living_room",
            "composition_similarity": 0.8 if group == "A" else 0.4,
            "shape_similarity": 0.7,
            "color_similarity": 0.6,
            "texture_similarity": 0.5,
            "multi_feature_score": 0.7,
        }
    return {
        "sequence": sequence,
        "status": "completed",
        "sample_category": "living_room",
        "effective_min_score": 0.5,
        "results": results,
    }


def test_valid_tasks_pass(tmp_path):
    tasks = [make_task(tmp_path, index) for index in range(1, 11)]
    result = ResultAuditor().audit(tasks)
    assert result["passed"] is True
    assert result["result_count"] == 30
    assert result["unique_candidate_count"] == 30


def test_duplicate_candidate_fails(tmp_path):
    task = make_task(tmp_path, 1, duplicate_path=True)
    result = ResultAuditor(min_samples=1).audit([task])
    assert result["passed"] is False
    assert any("重复候选路径" in message for message in result["errors"])
