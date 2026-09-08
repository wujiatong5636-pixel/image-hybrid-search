"""批量结果报告与四宫格预览导出。"""

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _load_font(size: int):
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def create_contact_sheet(task: dict, destination: Path) -> None:
    """制作原图、A、B、C四宫格。"""
    canvas = Image.new("RGB", (1600, 1240), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(34)
    label_font = _load_font(25)
    title = f"{task['sequence']:04d}  {task['source_name']}"
    draw.text((30, 18), title, fill="#111111", font=title_font)

    items = [("原始样图", Path(task["source_output"]), None)]
    for group in "ABC":
        result = task["results"][group]
        label = (
            f"{group}  CLIP={result['semantic_score']:.4f}  "
            f"MULTI={result.get('multi_feature_score', 0):.4f}  {result.get('category', '')}"
        )
        items.append((label, Path(result["output_path"]), group))

    positions = [(20, 80), (810, 80), (20, 660), (810, 660)]
    for (label, path, group), (x, y) in zip(items, positions):
        draw.rounded_rectangle(
            (x, y, x + 770, y + 550), radius=16, fill="white", outline="#cccccc", width=2
        )
        fitted = _fit_image(path, (730, 470))
        image_x = x + (770 - fitted.width) // 2
        image_y = y + 15 + (470 - fitted.height) // 2
        canvas.paste(fitted, (image_x, image_y))
        color = {"A": "#c0392b", "B": "#2471a3", "C": "#1e8449"}.get(group, "#222222")
        draw.text((x + 20, y + 505), label, fill=color, font=label_font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=92)


def build_rows(tasks: list[dict]) -> list[dict]:
    rows = []
    for task in sorted(tasks, key=lambda item: item["sequence"]):
        for group in "ABC":
            result = task["results"][group]
            rows.append(
                {
                    "样图序号": task["sequence"],
                    "样图文件": task["source_name"],
                    "样图路径": task["source_path"],
                    "分组": group,
                    "候选原文件": Path(result["candidate_path"]).name,
                    "候选原路径": result["candidate_path"],
                    "输出文件": Path(result["output_path"]).name,
                    "输出路径": result["output_path"],
                    "CLIP原始分": result["semantic_score"],
                    "pHash距离": result["phash_distance"],
                    "样图类别": task.get("sample_category"),
                    "候选类别": result.get("category"),
                    "候选类别分": result.get("category_score"),
                    "构图相似": result.get("composition_similarity"),
                    "造型相似": result.get("shape_similarity"),
                    "色彩相似": result.get("color_similarity"),
                    "纹理相似": result.get("texture_similarity"),
                    "多维分组分": result.get("multi_feature_score"),
                    "任务状态": task["status"],
                    "本组最低质量门槛": task.get("effective_min_score"),
                    "分数已校准": "否",
                }
            )
    return rows


def export_reports(tasks: list[dict], report_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(tasks)
    frame = pd.DataFrame(rows)
    csv_path = report_dir / "result.csv"
    excel_path = report_dir / "result.xlsx"
    json_path = report_dir / "result.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    frame.to_excel(excel_path, index=False, sheet_name="匹配结果")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rows": len(rows), "csv": csv_path, "excel": excel_path, "json": json_path}
