"""批量检索结果自动验收。"""

from pathlib import Path

from src.image_classifier import categories_compatible


REQUIRED_GROUPS = {"A", "B", "C"}
REQUIRED_FEATURES = {
    "composition_similarity",
    "shape_similarity",
    "color_similarity",
    "texture_similarity",
    "multi_feature_score",
}


class ResultAuditor:
    """检查任务完整性、重复、类别和分组统计。"""

    def __init__(self, min_samples=10, duplicate_phash_threshold=6):
        self.min_samples = int(min_samples)
        self.duplicate_phash_threshold = int(duplicate_phash_threshold)

    def audit(self, tasks):
        errors = []
        warnings = []
        completed = [item for item in tasks if item.get("status") == "completed"]
        if len(tasks) < self.min_samples:
            errors.append(f"样图任务不足{self.min_samples}个：当前{len(tasks)}个")
        if len(completed) != len(tasks):
            errors.append(f"存在未完成任务：{len(tasks) - len(completed)}个")

        rows = []
        for task in completed:
            sequence = task.get("sequence", "?")
            results = task.get("results", {})
            if set(results) != REQUIRED_GROUPS:
                errors.append(f"第{sequence}个任务缺少A/B/C结果")
                continue
            for group, item in results.items():
                row = dict(item)
                row["group"] = group
                row["sequence"] = sequence
                rows.append(row)
                if not Path(item.get("output_path", "")).is_file():
                    errors.append(f"第{sequence}个任务{group}组输出文件不存在")
                if not REQUIRED_FEATURES.issubset(item):
                    errors.append(f"第{sequence}个任务{group}组缺少多维评分")
                if int(item.get("phash_distance", 0)) <= self.duplicate_phash_threshold:
                    errors.append(f"第{sequence}个任务{group}组疑似与样图重复")
                sample_category = task.get("sample_category")
                candidate_category = item.get("category")
                if not sample_category or not candidate_category:
                    errors.append(f"第{sequence}个任务{group}组缺少类别信息")
                elif not categories_compatible(sample_category, candidate_category):
                    errors.append(f"第{sequence}个任务{group}组类别不兼容")
                if float(item.get("semantic_score", 0.0)) < float(task.get("effective_min_score", 0.0)):
                    errors.append(f"第{sequence}个任务{group}组低于语义质量门槛")

        paths = [str(Path(item.get("candidate_path", "")).resolve()) for item in rows]
        if len(paths) != len(set(paths)):
            errors.append("不同样图之间出现重复候选路径")

        expected_results = len(completed) * 3
        if len(rows) != expected_results:
            errors.append(f"结果数量异常：应有{expected_results}个，实际{len(rows)}个")

        group_stats = {}
        for group in "ABC":
            values = [item for item in rows if item["group"] == group]
            group_stats[group] = {
                "count": len(values),
                "semantic_mean": self._mean(values, "semantic_score"),
                "composition_mean": self._mean(values, "composition_similarity"),
                "shape_mean": self._mean(values, "shape_similarity"),
                "color_mean": self._mean(values, "color_similarity"),
                "texture_mean": self._mean(values, "texture_similarity"),
                "multi_feature_mean": self._mean(values, "multi_feature_score"),
            }

        if rows and group_stats["A"]["composition_mean"] <= group_stats["B"]["composition_mean"]:
            warnings.append("A组平均构图相似度未高于B组，请人工复核分组效果")
        if rows and group_stats["C"]["semantic_mean"] < 0.50:
            warnings.append("C组平均语义分低于0.50，请扩大高质量候选图库")

        return {
            "passed": not errors,
            "sample_count": len(tasks),
            "completed_count": len(completed),
            "result_count": len(rows),
            "unique_candidate_count": len(set(paths)),
            "errors": errors,
            "warnings": warnings,
            "group_stats": group_stats,
            "score_calibrated": False,
            "score_note": "当前CLIP与多维分数用于排序，尚不能解释为80%-95%的真实概率。",
        }

    @staticmethod
    def _mean(items, key):
        values = [float(item[key]) for item in items if key in item]
        return round(sum(values) / len(values), 6) if values else 0.0
