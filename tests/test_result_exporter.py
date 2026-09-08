"""结果报告导出测试。"""

from src.result_exporter import build_rows


def test_build_rows_keeps_sample_and_group_order():
    def task(sequence):
        return {
            "sequence": sequence,
            "source_name": f"{sequence}.png",
            "source_path": f"input/{sequence}.png",
            "status": "completed",
            "results": {
                group: {
                    "candidate_path": f"candidate/{sequence}_{group}.png",
                    "output_path": f"output/{sequence}_{group}.png",
                    "semantic_score": 0.8,
                    "phash_distance": 20,
                }
                for group in "ABC"
            },
        }

    rows = build_rows([task(2), task(1)])
    assert [(row["样图序号"], row["分组"]) for row in rows] == [
        (1, "A"), (1, "B"), (1, "C"),
        (2, "A"), (2, "B"), (2, "C"),
    ]
