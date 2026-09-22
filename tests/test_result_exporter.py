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


def test_build_rows_records_google_fallback_fields():
    task = {
        "sequence": 1,
        "source_name": "1.png",
        "source_path": "input/1.png",
        "status": "completed",
        "google_called": True,
        "google_status": "success",
        "google_url_count": 8,
        "google_downloaded_count": 5,
        "google_valid_count": 2,
        "results": {
            group: {
                "candidate_path": f"candidate/{group}.jpg",
                "output_path": f"output/{group}.jpg",
                "source": "google" if group == "C" else "local",
                "source_url": "https://example.com/c.jpg" if group == "C" else None,
                "source_rank": 1 if group == "C" else None,
                "semantic_score": 0.8,
                "phash_distance": 20,
            }
            for group in "ABC"
        },
    }

    rows = build_rows([task])

    assert all(row["是否触发Google"] == "是" for row in rows)
    assert all(row["Google状态"] == "success" for row in rows)
    assert rows[2]["候选来源"] == "google"
    assert rows[2]["网络原始URL"] == "https://example.com/c.jpg"
