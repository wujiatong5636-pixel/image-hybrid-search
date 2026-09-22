# 混合图片检索系统 (Image Hybrid Search)

基于 **CLIP + FAISS** 的本地主检索系统。正式批处理采用三级降级链路：本地有效候选不足 3 张时调用百度识图；百度不可用、无结果或过滤后仍不足 3 张时，才调用 **Google Vision Web Detection**。所有网络候选都会下载到本地并重新执行 CLIP、类别和重复度过滤。

> 当前稳定版本方向：`local → baidu → google`。Google 只作为最终补充，不参与本地候选充足时的检索。

## 架构概览

```
查询图片 ──┬──> CLIP 编码 ──> FAISS 向量检索 ──> 本地库 Top-K (clip_image)
           │
           ├──> Google Vision Web Detection ──> 全网匹配 URL (vision)
           │
           └──> 融合引擎 (weighted / RRF) ──> 最终排序结果
```

## 特性

- **双 CLIP 后端**：ONNX Runtime（CPU 轻量）/ OpenCLIP（GPU 开发）
- **双融合策略**：加权求和（分数稳定场景）/ RRF 倒数排名融合（异构结果场景）
- **向量库选型**：FAISS Flat（<5万）/ FAISS IVF（5万+），可扩展 Qdrant/Milvus
- **自动限流降级**：Vision 免费额度用到 80% 时自动切换纯 CLIP 模式
- **全网溯源**：Google Vision 返回完全匹配、部分匹配、视觉相似、包含网页、实体标签
- **配置化**：所有参数通过 YAML 配置，无需改代码

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖（CLIP + FAISS）
pip install numpy Pillow PyYAML faiss-cpu open-clip-torch onnxruntime modern-onnx-clip tqdm

# Google Vision（可选，不需要全网溯源可跳过）
pip install google-cloud-vision

# 开发测试
pip install pytest
```

### 2. 配置

编辑 `config.yaml`：

```yaml
clip:
  backend: "onnx"           # onnx | openclip
  device: "cpu"             # cpu | cuda

vector_store:
  index_type: "flat"        # flat (<5万) | ivf (5万+)
  index_path: "data/faiss_index.bin"

google_vision:
  enabled: true
  credentials_path: "credentials/google_vision_key.json"
  proxy:                    # 国内访问需要配置代理
    http: "http://127.0.0.1:7890"
    https: "http://127.0.0.1:7890"

fusion:
  strategy: "weighted"      # weighted | rrf
  weights:
    vision: 0.3
    clip_image: 0.5
    clip_text: 0.2
```

### 3. 构建索引

```bash
python scripts/build_index.py --image_dir /path/to/your/images
```

### 4. 检索

```bash
# 以图搜图（混合模式）
python scripts/demo.py --query /path/to/query.jpg

# 以图搜图（纯 CLIP，不调用 Vision）
python scripts/demo.py --query /path/to/query.jpg --no_vision

# 以文搜图
python scripts/demo.py --text "一只橘猫在窗台上晒太阳"

# JSON 输出
python scripts/demo.py --query query.jpg --json
```

## Python API 用法

```python
from src.hybrid_searcher import HybridSearcher

# 初始化
searcher = HybridSearcher(config_path="config.yaml")

# 构建索引
searcher.build_index(image_dir="/path/to/images")

# 以图搜图
result = searcher.search_by_image("query.jpg", top_k=10)
for item in result["results"]:
    print(f"{item['score']:.4f}  [{item['source']}]  {item['id']}")

# 以文搜图
result = searcher.search_by_text("夕阳下的海边", top_k=10)

# 增量添加
searcher.add_image("new_image.jpg")

# 查看 Vision 用量
usage = searcher.rate_limiter.get_usage()
print(f"已用 {usage['used']}/{usage['max']} 次")
```

## 模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| CLIP 封装 | `src/clip_searcher.py` | 图像/文本编码，ONNX/OpenCLIP 双后端 |
| 向量存储 | `src/vector_store.py` | FAISS 索引构建、查询、持久化 |
| Vision 溯源 | `src/vision_searcher.py` | Google Vision Web Detection 调用与解析 |
| 融合引擎 | `src/fusion.py` | 加权求和 / RRF 融合排序 |
| 限流降级 | `src/rate_limiter.py` | Vision 调用量监控、月度配额、自动降级 |
| 主入口 | `src/hybrid_searcher.py` | 整合所有模块，提供统一检索接口 |

## 成本说明

| 维度 | 说明 |
|------|------|
| Google Vision 免费额度 | 每个功能每月前 1000 次调用免费 |
| 超量费用 | $3.50 / 千次（1001–5,000,000 次区间） |
| 新用户赠金 | $300（Google Cloud 新注册用户） |
| 日均 30 张 | 月均 900 次，完全免费 |
| 日均 100 张 | 月均 3000 次，约 $7/月（~50 元） |
| CLIP + FAISS | 完全免费，本地运行 |

## 向量库选型建议

| 数据规模 | 推荐方案 | 理由 |
|----------|----------|------|
| < 5 万张 | FAISS Flat | 部署极简，查询最快 |
| 5 万–50 万 | FAISS IVF / Chroma / LanceDB | 轻量级，零配置，持久化 |
| > 50 万张 | Qdrant / Milvus | 分布式扩展，水平扩容 |
| 已有 PostgreSQL | pgvector | 与现有栈集成 |

## 融合策略选择

- **加权求和 (weighted)**：两路结果质量稳定、分数可比、需要精细调权时使用
- **RRF (rrf)**：两路结果数量差异大、Vision 无分值、不想调权时使用，更鲁棒

## 风险与应对

| 风险 | 应对 |
|------|------|
| Google Vision 国内访问 | 配置代理或 Cloud VPN |
| CLIP 冷启动慢 | 分批构建 + 进度断点续传 |
| 免费额度超量 | 硬限流：80% 时自动降级纯 CLIP |
| 模型更新导致向量变化 | 锁定模型版本号，更新时重建索引 |
| 图片格式兼容 | CLIP 自动转 RGB，PNG 透明通道自动处理 |

## 运行测试

```bash
python -m pytest tests/ -v
```

测试覆盖融合引擎、限流器和向量存储（不依赖外部模型和 API）。

## 项目结构

```
image-hybrid-search/
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖清单
├── README.md               # 本文档
├── src/
│   ├── __init__.py
│   ├── clip_searcher.py     # CLIP 封装
│   ├── vector_store.py      # FAISS 向量存储
│   ├── vision_searcher.py   # Google Vision
│   ├── fusion.py            # 融合排序
│   ├── rate_limiter.py      # 限流降级
│   └── hybrid_searcher.py   # 主入口
├── scripts/
│   ├── build_index.py       # 批量建索引
│   └── demo.py              # 演示脚本
├── tests/
│   └── test_hybrid.py       # 单元测试
└── data/                    # 索引和数据文件（自动生成）
```

## License

MIT
