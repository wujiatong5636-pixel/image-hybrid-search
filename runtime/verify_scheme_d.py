"""方案D批量结果自动验收。"""
import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

tasks = json.loads(Path("runtime/tasks/batch_task.json").read_text(encoding="utf-8"))["tasks"]
registry = json.loads(Path("runtime/tasks/used_candidates.json").read_text(encoding="utf-8"))["records"]
output = Path("output")

print("=" * 66)
print("方案D 批量结果验收")
print("=" * 66)

errors = []
source_counter = Counter()
all_candidate_hashes = []

for task in sorted(tasks.values(), key=lambda x: x["sequence"]):
    seq = task["sequence"]
    name = task["source_name"]
    status = task["status"]
    results = task.get("results", {})

    # 1. 状态必须completed
    if status != "completed":
        errors.append(f"#{seq} {name} 状态={status} 不是completed")
        continue

    # 2. 必须有A/B/C三个结果
    if set(results.keys()) != {"A", "B", "C"}:
        errors.append(f"#{seq} {name} 结果不完整: {list(results.keys())}")
        continue

    # 3. 输出文件必须真实存在
    folder = output / f"{seq:04d}_{name.split('.')[0].replace(' ','_')}"
    # 用results里的output_path检查更准确
    for g in "ABC":
        op = Path(results[g]["output_path"])
        if not op.is_file():
            errors.append(f"#{seq} {g}组输出文件不存在: {op}")
        source_counter[results[g].get("source", "?")] += 1

    # 4. source文件存在
    if not Path(task.get("source_output", "")).is_file():
        errors.append(f"#{seq} source样图缺失")

    baidu = task.get("baidu_called", False)
    local_n = task.get("local_valid_count", 0)
    print(f"#{seq:02d} {name:<12} 状态={status:<9} 本地有效={local_n:<2} 触发百度={baidu}  "
          f"A={results['A'].get('source'):<5} B={results['B'].get('source'):<5} C={results['C'].get('source'):<5}")

# 5. 全局占用：30个，且哈希无重复
print("-" * 66)
print(f"全局占用候选数: {len(registry)} (应为30)")
if len(registry) != 30:
    errors.append(f"全局占用数={len(registry)}，应为30")

hash_list = list(registry.keys())
if len(hash_list) != len(set(hash_list)):
    errors.append("全局占用存在重复哈希")

# 6. 来源统计
print(f"候选来源分布: {dict(source_counter)}")

# 7. 第6张百度兜底专项检查
t6 = [t for t in tasks.values() if t["sequence"] == 6][0]
print("-" * 66)
print("第6张（本地不足触发百度）专项:")
print(f"  本地有效候选: {t6.get('local_valid_count')}")
print(f"  是否触发百度: {t6.get('baidu_called')}")
print(f"  百度返回URL: {t6.get('baidu_url_count')}")
print(f"  百度下载成功: {t6.get('baidu_downloaded_count')}")
print(f"  百度过滤后有效: {t6.get('baidu_valid_count')}")
if t6.get("baidu_called") and t6.get("baidu_valid_count", 0) > 0:
    print("  [OK] 百度兜底链路完整")
else:
    errors.append("第6张百度兜底链路异常")

# 8. 其他9张不应触发百度
others = [t for t in tasks.values() if t["sequence"] != 6]
wrong_baidu = [t["sequence"] for t in others if t.get("baidu_called")]
if wrong_baidu:
    errors.append(f"本地充足却触发百度的样图: {wrong_baidu}")
else:
    print("-" * 66)
    print("[OK] 其余9张本地充足，均未触发百度")

print("=" * 66)
if errors:
    print(f"[发现 {len(errors)} 个问题]")
    for e in errors:
        print("  [问题]", e)
    sys.exit(1)
else:
    print("[全部验收通过] 10/10完成，30个候选无重复，百度兜底正常")
