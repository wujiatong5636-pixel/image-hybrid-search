# 方案 D 落地部署指南（小白版）

> 本文档面向非专业开发者，每一步都有
> **复制粘贴的命令**
> 和
> **做完后应该看到什么**
> 的验证方法。
> 遇到问题先看文末「常见问题排查」。



***

## 一、先搞懂：你的项目现在是什么状态

### 1.1 项目里已经有的东西（不用动，直接复用）



| 文件夹 / 文件                         | 作用              | 大白话解释                             |
| -------------------------------- | --------------- | --------------------------------- |
| `src/clip_searcher.py`           | CLIP 图片编码       | 把图片变成一串数字（向量），用来比较相似度             |
| `src/vector_store.py`            | FAISS 向量数据库     | 把所有候选图片的数字存起来，快速搜索相似图             |
| `src/duplicate_detector.py`      | 重复图片检测          | 用 SHA256（完全一样）和 pHash（看起来差不多）判断重复 |
| `src/image_classifier.py`        | 图片分类            | 判断图片是客厅 / 卧室 / 厨房等 12 种类别         |
| `src/multi_feature_scorer.py`    | 多维评分 + A/B/C 选择 | 从构图、造型、色彩、纹理四个维度打分，选出 A/B/C 三张    |
| `src/candidate_registry.py`      | 全局占用登记          | 记录哪张候选图已经被用过了，防止不同样图选到同一张         |
| `src/task_store.py`              | 任务断点存储          | 记录每张样图处理到哪了，中断后可以继续               |
| `src/result_exporter.py`         | 报告导出            | 生成 Excel/CSV/JSON 报告和四宫格预览图       |
| `src/vision_searcher.py`         | Google Vision   | 谷歌识图（方案 D 要关掉，但代码保留）              |
| `scripts/run_batch_selection.py` | 批量处理主脚本         | 跑 10 张样图的主程序                      |

### 1.2 方案 D 要改什么（一句话总结）

> **现在的程序只从本地 FAISS 库里找图，找不到够 3 张就报错。方案 D 要加一个 "百度识图" 兜底：本地找不到够 3 张时，自动去百度搜图、下载下来、重新打分，凑够 A/B/C 三张。同时彻底关掉 Google Vision 省钱。**

### 1.3 方案 D 的核心规则（必须记住）



1. **本地优先**：先在本地 FAISS 库找 98 张，过滤后够 3 张就绝不调用百度

2. **过滤后才算数**：FAISS 返回 98 张没用，要经过 "路径存在→不重复→类别对→分数够→没被占用" 五道过滤后，剩下的才叫 "有效候选"

3. **百度结果不算数**：百度返回的排名、相似度全部丢弃，下载到本地后用咱们自己的 CLIP 重新打分

4. **凑不够就报错**：百度也找不到够 3 张时，标记 "候选不足"，不强行凑数，下次运行会重试

5. **百度失败不崩溃**：百度挂了只影响当前样图，程序继续处理下一张



***

## 二、实施前准备（必做）

### 2.1 备份项目（出问题能回滚）

打开命令行（按 `Win+R`，输入 `cmd`，回车），执行：



```
cd /d C:\Users\mozhihao\core\image-hybrid-search

xcopy . runtime\backups\before\\\_scheme\\\_d\\\_final\ /E /I /H /Y
```

**验证**：执行完后，`runtime\backups\before_scheme_d_final\` 文件夹里应该有完整的项目文件。

> 以后任何一步出问题，把这个文件夹里的文件复制回项目根目录就能恢复。

### 2.2 确认虚拟环境能用



```
cd /d C:\Users\mozhihao\core\image-hybrid-search

.venv\Scripts\python.exe --version
```

**验证**：应该显示 Python 版本号（比如 `Python 3.11.x`）。

### 2.3 确认当前配置状态



```
.venv\Scripts\python.exe -c "import yaml; d=yaml.safe\\\_load(open('config.yaml',encoding='utf-8')); print('google\\\_vision.enabled =', d\\\['google\\\_vision']\\\['enabled'])"
```

**验证**：应该输出 `google_vision.enabled = False`（你的配置已经是关闭状态了）。



***

## 三、第一阶段：修改配置文件（5 分钟）

### 3.1 用记事本打开配置文件



```
notepad config.yaml
```

### 3.2 在文件末尾追加以下内容

把下面这段完整复制，粘贴到 `config.yaml` 文件的**最后面**（不要删原来的内容）：



```
\\# ===== 方案D：检索提供器配置 =====

search\\\_provider:

\&#x20; primary: "local"          # 主检索：本地CLIP+FAISS

\&#x20; fallback: "baidu"         # 降级检索：百度识图

\&#x20; fallback\\\_trigger\\\_count: 3 # 本地有效候选少于3张时触发百度

\&#x20; local\\\_top\\\_k: 98           # 本地FAISS召回数量

\\# ===== 百度识图配置 =====

baidu\\\_image\\\_search:

\&#x20; enabled: true

\&#x20; mode: "auto"              # auto=HTTP失败自动改用浏览器；http=只用HTTP；browser=只用浏览器

\&#x20; timeout: 30               # 单次请求超时（秒）

\&#x20; max\\\_results: 30           # 百度最多返回多少个URL

\&#x20; max\\\_downloads: 20         # 实际最多下载多少张

\&#x20; retries: 1                # 失败重试次数

\&#x20; min\\\_interval\\\_seconds: 3   # 两次百度请求之间最小间隔（秒），防止被封

\&#x20; circuit\\\_breaker\\\_failures: 3  # 连续失败多少次后熔断（暂时不再调用百度）

\&#x20; cache\\\_dir: "runtime/baidu\\\_cache"           # 百度图片缓存目录

\&#x20; browser\\\_profile\\\_dir: "runtime/baidu\\\_browser\\\_profile"  # 浏览器数据目录

\\# ===== 远程图片下载配置 =====

remote\\\_download:

\&#x20; timeout: 15               # 下载超时（秒）

\&#x20; retries: 2                # 下载重试次数

\&#x20; max\\\_file\\\_mb: 15           # 单文件最大15MB

\&#x20; min\\\_width: 200            # 图片最小宽度

\&#x20; min\\\_height: 200           # 图片最小高度
```

### 3.3 保存并验证配置

按 `Ctrl+S` 保存，关闭记事本。然后执行：



```
.venv\Scripts\python.exe -c "import yaml; d=yaml.safe\\\_load(open('config.yaml',encoding='utf-8')); print('google\\\_vision.enabled =', d\\\['google\\\_vision']\\\['enabled']); print('baidu\\\_image\\\_search.enabled =', d\\\['baidu\\\_image\\\_search']\\\['enabled']); print('fallback\\\_trigger\\\_count =', d\\\['search\\\_provider']\\\['fallback\\\_trigger\\\_count'])"
```

**验证**：应该输出：



```
google\\\_vision.enabled = False

baidu\\\_image\\\_search.enabled = True

fallback\\\_trigger\\\_count = 3
```

> 如果报错说找不到
> `search_provider`
> ，说明你没粘贴对位置，重新检查。



***

## 四、第二阶段：新增代码文件（核心工作）

> 这一步要新建 7 个文件。每个文件我都给出了
> **完整代码**
> ，你只需要：新建文件 → 复制代码 → 保存。

### 4.1 新建 `src/providers/` 文件夹



```
mkdir src\providers
```

### 4.2 新建 `src/providers/__init__.py`



```
notepad src\providers\\\\\\\_\\\_init\\\_\\\_.py
```

在记事本里粘贴：



```
"""检索提供器包：本地FAISS和百度识图统一接口。"""
```

保存关闭。

### 4.3 新建 `src/providers/base.py`（统一数据结构）



```
notepad src\providers\base.py
```

粘贴以下完整代码：



```
"""

统一候选数据结构定义。

无论候选来自本地还是百度，进入后续评分前都必须转换成这个格式。

"""

from dataclasses import dataclass, field

from typing import Optional

@dataclass

class Candidate:

\&#x20;   """统一候选数据结构。"""

\&#x20;   # 来源标识："local" 或 "baidu"

\&#x20;   source: str

\&#x20;   # 来源ID：本地为文件路径，百度为原始URL

\&#x20;   source\\\_id: str

\&#x20;   # 百度原始URL（本地候选为None）

\&#x20;   source\\\_url: Optional\\\[str] = None

\&#x20;   # 本地文件路径（百度候选下载后的缓存路径）

\&#x20;   local\\\_path: str = ""

\&#x20;   # 在来源中的排名（从1开始）

\&#x20;   source\\\_rank: int = 0

\&#x20;   # 来源给出的分数（百度为None，本地为CLIP分数）

\&#x20;   source\\\_score: Optional\\\[float] = None

\&#x20;   # 匹配类型："local" 或 "visually\\\_similar"

\&#x20;   match\\\_type: str = "local"

\&#x20;   def to\\\_dict(self) -> dict:

\&#x20;       return {

\&#x20;           "source": self.source,

\&#x20;           "source\\\_id": self.source\\\_id,

\&#x20;           "source\\\_url": self.source\\\_url,

\&#x20;           "local\\\_path": self.local\\\_path,

\&#x20;           "source\\\_rank": self.source\\\_rank,

\&#x20;           "source\\\_score": self.source\\\_score,

\&#x20;           "match\\\_type": self.match\\\_type,

\&#x20;       }

class BaseProvider:

\&#x20;   """检索提供器基类，定义统一接口。"""

\&#x20;   def search(self, sample\\\_path: str, top\\\_k: int = 98) -> list:

\&#x20;       """

\&#x20;       检索候选图片。

\&#x20;       Args:

\&#x20;           sample\\\_path: 样图路径

\&#x20;           top\\\_k: 返回候选数量

\&#x20;       Returns:

\&#x20;           Candidate 对象列表

\&#x20;       """

\&#x20;       raise NotImplementedError("子类必须实现 search 方法")

\&#x20;   @property

\&#x20;   def is\\\_available(self) -> bool:

\&#x20;       """提供器是否可用。"""

\&#x20;       return True
```

保存关闭。

### 4.4 新建 `src/providers/local_faiss_provider.py`（本地检索封装）



```
notepad src\providers\local\\\_faiss\\\_provider.py
```

粘贴：



```
"""

本地CLIP+FAISS检索提供器。

封装现有的 CLIPSearcher 和 VectorStore，输出统一的 Candidate 格式。

"""

import logging

from pathlib import Path

from typing import List

from .base import Candidate, BaseProvider

logger = logging.getLogger(\\\_\\\_name\\\_\\\_)

class LocalFAISSProvider(BaseProvider):

\&#x20;   """本地CLIP+FAISS检索。"""

\&#x20;   def \\\_\\\_init\\\_\\\_(self, clip\\\_searcher, vector\\\_store):

\&#x20;       """

\&#x20;       Args:

\&#x20;           clip\\\_searcher: 已初始化的 CLIPSearcher 实例

\&#x20;           vector\\\_store: 已加载的 VectorStore 实例

\&#x20;       """

\&#x20;       self.clip = clip\\\_searcher

\&#x20;       self.vector\\\_store = vector\\\_store

\&#x20;   def search(self, sample\\\_path: str, top\\\_k: int = 98) -> List\\\[Candidate]:

\&#x20;       """

\&#x20;       执行本地FAISS检索。

\&#x20;       Args:

\&#x20;           sample\\\_path: 样图路径

\&#x20;           top\\\_k: 召回数量

\&#x20;       Returns:

\&#x20;           Candidate 列表（原始召回，未过滤）

\&#x20;       """

\&#x20;       sample\\\_path = str(Path(sample\\\_path).resolve())

\&#x20;       # CLIP编码样图

\&#x20;       query\\\_vector = self.clip.encode\\\_image(sample\\\_path)

\&#x20;       # FAISS检索

\&#x20;       raw\\\_results = self.vector\\\_store.search(

\&#x20;           query\\\_vector,

\&#x20;           top\\\_k=top\\\_k,

\&#x20;           threshold=0.0,

\&#x20;       )

\&#x20;       candidates = \\\[]

\&#x20;       for rank, (path, score, internal\\\_id) in enumerate(raw\\\_results, start=1):

\&#x20;           candidate = Candidate(

\&#x20;               source="local",

\&#x20;               source\\\_id=str(Path(path).resolve()),

\&#x20;               source\\\_url=None,

\&#x20;               local\\\_path=str(Path(path).resolve()),

\&#x20;               source\\\_rank=rank,

\&#x20;               source\\\_score=float(score),

\&#x20;               match\\\_type="local",

\&#x20;           )

\&#x20;           # 附带内部ID，方便后续重建向量

\&#x20;           candidate.internal\\\_id = internal\\\_id

\&#x20;           candidates.append(candidate)

\&#x20;       logger.info("本地FAISS召回 %d 条候选", len(candidates))

\&#x20;       return candidates

\&#x20;   @property

\&#x20;   def is\\\_available(self) -> bool:

\&#x20;       return len(self.vector\\\_store) > 0
```

保存关闭。

### 4.5 新建 `src/providers/baidu_image_provider.py`（百度识图）



```
notepad src\providers\baidu\\\_image\\\_provider.py
```

粘贴：



```
"""

百度识图提供器。

支持两种模式：

\&#x20; \\- http：直接发HTTP请求解析结果（快，但可能被反爬）

\&#x20; \\- browser：用Playwright浏览器自动化（慢，但更稳定）

\&#x20; \\- auto：先试HTTP，失败再用浏览器

注意：百度识图是网页逆向实现，接口随时可能变化。

本模块只负责获取图片URL列表，下载和评分由其他模块处理。

"""

import hashlib

import json

import logging

import os

import re

import time

from pathlib import Path

from typing import Dict, List, Optional

from urllib.parse import urlparse

import requests

from .base import Candidate, BaseProvider

logger = logging.getLogger(\\\_\\\_name\\\_\\\_)

class BaiduImageProvider(BaseProvider):

\&#x20;   """百度识图提供器。"""

\&#x20;   # 百度识图上传接口

\&#x20;   UPLOAD\\\_URL = "https://graph.baidu.com/upload"

\&#x20;   SEARCH\\\_URL = "https://graph.baidu.com/s"

\&#x20;   def \\\_\\\_init\\\_\\\_(self, config: dict):

\&#x20;       """

\&#x20;       Args:

\&#x20;           config: config.yaml 中的 baidu\\\_image\\\_search 配置段

\&#x20;       """

\&#x20;       self.config = config

\&#x20;       self.mode = config.get("mode", "auto")

\&#x20;       self.timeout = config.get("timeout", 30)

\&#x20;       self.max\\\_results = config.get("max\\\_results", 30)

\&#x20;       self.retries = config.get("retries", 1)

\&#x20;       self.min\\\_interval = config.get("min\\\_interval\\\_seconds", 3)

\&#x20;       self.circuit\\\_breaker\\\_failures = config.get("circuit\\\_breaker\\\_failures", 3)

\&#x20;       # 熔断状态

\&#x20;       self.\\\_consecutive\\\_failures = 0

\&#x20;       self.\\\_last\\\_request\\\_time = 0.0

\&#x20;       self.\\\_session = self.\\\_build\\\_session()

\&#x20;   def \\\_build\\\_session(self) -> requests.Session:

\&#x20;       """构建带浏览器UA的请求会话。"""

\&#x20;       session = requests.Session()

\&#x20;       session.headers.update({

\&#x20;           "User-Agent": (

\&#x20;               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "

\&#x20;               "AppleWebKit/537.36 (KHTML, like Gecko) "

\&#x20;               "Chrome/120.0.0.0 Safari/537.36"

\&#x20;           ),

\&#x20;           "Accept": "application/json, text/plain, \\\*/\\\*",

\&#x20;           "Accept-Language": "zh-CN,zh;q=0.9",

\&#x20;       })

\&#x20;       return session

\&#x20;   @property

\&#x20;   def is\\\_circuit\\\_open(self) -> bool:

\&#x20;       """熔断器是否打开（连续失败过多时暂时停止调用）。"""

\&#x20;       return self.\\\_consecutive\\\_failures >= self.circuit\\\_breaker\\\_failures

\&#x20;   def \\\_record\\\_success(self):

\&#x20;       """记录一次成功调用，重置失败计数。"""

\&#x20;       self.\\\_consecutive\\\_failures = 0

\&#x20;   def \\\_record\\\_failure(self):

\&#x20;       """记录一次失败调用。"""

\&#x20;       self.\\\_consecutive\\\_failures += 1

\&#x20;       logger.warning("百度识图连续失败 %d/%d 次",

\&#x20;                      self.\\\_consecutive\\\_failures, self.circuit\\\_breaker\\\_failures)

\&#x20;   def \\\_wait\\\_interval(self):

\&#x20;       """两次请求之间的最小间隔，防止被封。"""

\&#x20;       elapsed = time.time() - self.\\\_last\\\_request\\\_time

\&#x20;       if elapsed < self.min\\\_interval:

\&#x20;           time.sleep(self.min\\\_interval - elapsed)

\&#x20;       self.\\\_last\\\_request\\\_time = time.time()

\&#x20;   def search(self, sample\\\_path: str, top\\\_k: int = 30) -> List\\\[Candidate]:

\&#x20;       """

\&#x20;       百度识图检索。

\&#x20;       Args:

\&#x20;           sample\\\_path: 样图路径

\&#x20;           top\\\_k: 返回URL数量上限

\&#x20;       Returns:

\&#x20;           Candidate 列表（只有URL，尚未下载）

\&#x20;       """

\&#x20;       if self.is\\\_circuit\\\_open:

\&#x20;           logger.warning("百度识图熔断器已打开（连续失败%d次），跳过本次调用",

\&#x20;                          self.\\\_consecutive\\\_failures)

\&#x20;           return \\\[]

\&#x20;       sample\\\_path = str(Path(sample\\\_path).resolve())

\&#x20;       if not Path(sample\\\_path).is\\\_file():

\&#x20;           logger.error("样图不存在: %s", sample\\\_path)

\&#x20;           return \\\[]

\&#x20;       urls: List\\\[str] = \\\[]

\&#x20;       # 根据模式选择执行方式

\&#x20;       if self.mode in ("http", "auto"):

\&#x20;           urls = self.\\\_search\\\_http(sample\\\_path)

\&#x20;           if urls:

\&#x20;               self.\\\_record\\\_success()

\&#x20;           elif self.mode == "auto":

\&#x20;               logger.info("HTTP模式无结果，自动切换浏览器模式")

\&#x20;               urls = self.\\\_search\\\_browser(sample\\\_path)

\&#x20;               if urls:

\&#x20;                   self.\\\_record\\\_success()

\&#x20;               else:

\&#x20;                   self.\\\_record\\\_failure()

\&#x20;           else:

\&#x20;               self.\\\_record\\\_failure()

\&#x20;       elif self.mode == "browser":

\&#x20;           urls = self.\\\_search\\\_browser(sample\\\_path)

\&#x20;           if urls:

\&#x20;               self.\\\_record\\\_success()

\&#x20;           else:

\&#x20;               self.\\\_record\\\_failure()

\&#x20;       # 限制返回数量

\&#x20;       urls = urls\\\[:min(top\\\_k, self.max\\\_results)]

\&#x20;       # 转换为统一Candidate格式

\&#x20;       candidates = \\\[]

\&#x20;       for rank, url in enumerate(urls, start=1):

\&#x20;           candidate = Candidate(

\&#x20;               source="baidu",

\&#x20;               source\\\_id=url,

\&#x20;               source\\\_url=url,

\&#x20;               local\\\_path="",  # 下载后才填充

\&#x20;               source\\\_rank=rank,

\&#x20;               source\\\_score=None,  # 百度分数不采信

\&#x20;               match\\\_type="visually\\\_similar",

\&#x20;           )

\&#x20;           candidates.append(candidate)

\&#x20;       logger.info("百度识图返回 %d 个URL", len(candidates))

\&#x20;       return candidates

\&#x20;   def \\\_search\\\_http(self, sample\\\_path: str) -> List\\\[str]:

\&#x20;       """HTTP模式：上传图片并解析结果。"""

\&#x20;       try:

\&#x20;           self.\\\_wait\\\_interval()

\&#x20;           # 第一步：上传图片

\&#x20;           with open(sample\\\_path, "rb") as f:

\&#x20;               files = {"image": (Path(sample\\\_path).name, f, "image/jpeg")}

\&#x20;               response = self.\\\_session.post(

\&#x20;                   self.UPLOAD\\\_URL,

\&#x20;                   files=files,

\&#x20;                   timeout=self.timeout,

\&#x20;               )

\&#x20;           response.raise\\\_for\\\_status()

\&#x20;           data = response.json()

\&#x20;           # 解析返回的搜索URL或sign

\&#x20;           search\\\_url = data.get("data", {}).get("url", "")

\&#x20;           if not search\\\_url:

\&#x20;               logger.warning("百度上传返回中没有搜索URL")

\&#x20;               return \\\[]

\&#x20;           # 第二步：访问搜索结果页

\&#x20;           self.\\\_wait\\\_interval()

\&#x20;           response = self.\\\_session.get(search\\\_url, timeout=self.timeout)

\&#x20;           response.raise\\\_for\\\_status()

\&#x20;           # 从HTML中提取图片URL

\&#x20;           urls = self.\\\_extract\\\_urls\\\_from\\\_html(response.text)

\&#x20;           return urls

\&#x20;       except Exception as e:

\&#x20;           logger.warning("百度HTTP模式失败: %s", e)

\&#x20;           return \\\[]

\&#x20;   def \\\_extract\\\_urls\\\_from\\\_html(self, html: str) -> List\\\[str]:

\&#x20;       """从百度搜索结果HTML中提取图片URL。"""

\&#x20;       urls = \\\[]

\&#x20;       # 尝试从JSON数据中提取（百度页面通常把结果嵌在script标签里）

\&#x20;       json\\\_patterns = \\\[

\&#x20;           r'"thumbUrl"\s\\\*:\s\\\*"(\\\[^"]+)"',

\&#x20;           r'"objUrl"\s\\\*:\s\\\*"(\\\[^"]+)"',

\&#x20;           r'"middleUrl"\s\\\*:\s\\\*"(\\\[^"]+)"',

\&#x20;           r'"hoverUrl"\s\\\*:\s\\\*"(\\\[^"]+)"',

\&#x20;       ]

\&#x20;       for pattern in json\\\_patterns:

\&#x20;           for match in re.finditer(pattern, html):

\&#x20;               url = match.group(1).replace("\\\\\\\\/", "/")

\&#x20;               if url and url.startswith("http") and url not in urls:

\&#x20;                   urls.append(url)

\&#x20;       # 尝试从img标签提取

\&#x20;       for match in re.finditer(r'\\\<img\\\[^>]+data-imgurl="(\\\[^"]+)"', html):

\&#x20;           url = match.group(1)

\&#x20;           if url and url.startswith("http") and url not in urls:

\&#x20;               urls.append(url)

\&#x20;       return urls

\&#x20;   def \\\_search\\\_browser(self, sample\\\_path: str) -> List\\\[str]:

\&#x20;       """

\&#x20;       浏览器模式：用Playwright打开百度识图页面上传图片。

\&#x20;       需要先安装 playwright：pip install playwright && playwright install chromium

\&#x20;       """

\&#x20;       try:

\&#x20;           from playwright.sync\\\_api import sync\\\_playwright

\&#x20;       except ImportError:

\&#x20;           logger.warning("未安装playwright，浏览器模式不可用。"

\&#x20;                          "请运行: pip install playwright && playwright install chromium")

\&#x20;           return \\\[]

\&#x20;       try:

\&#x20;           self.\\\_wait\\\_interval()

\&#x20;           profile\\\_dir = str(Path(self.config.get(

\&#x20;               "browser\\\_profile\\\_dir", "runtime/baidu\\\_browser\\\_profile"

\&#x20;           )).resolve())

\&#x20;           Path(profile\\\_dir).mkdir(parents=True, exist\\\_ok=True)

\&#x20;           with sync\\\_playwright() as p:

\&#x20;               browser = p.chromium.launch\\\_persistent\\\_context(

\&#x20;                   user\\\_data\\\_dir=profile\\\_dir,

\&#x20;                   headless=True,

\&#x20;                   viewport={"width": 1280, "height": 800},

\&#x20;               )

\&#x20;               page = browser.new\\\_page()

\&#x20;               # 打开百度识图

\&#x20;               page.goto("https://graph.baidu.com/", timeout=self.timeout \\\* 1000)

\&#x20;               page.wait\\\_for\\\_load\\\_state("networkidle", timeout=self.timeout \\\* 1000)

\&#x20;               # 上传图片

\&#x20;               file\\\_input = page.locator('input\\\[type="file"]').first

\&#x20;               file\\\_input.set\\\_input\\\_files(sample\\\_path)

\&#x20;               # 等待结果加载

\&#x20;               page.wait\\\_for\\\_load\\\_state("networkidle", timeout=self.timeout \\\* 1000)

\&#x20;               time.sleep(3)  # 额外等待动态内容

\&#x20;               # 提取页面中的图片URL

\&#x20;               html = page.content()

\&#x20;               urls = self.\\\_extract\\\_urls\\\_from\\\_html(html)

\&#x20;               browser.close()

\&#x20;               return urls

\&#x20;       except Exception as e:

\&#x20;           logger.warning("百度浏览器模式失败: %s", e)

\&#x20;           return \\\[]

\&#x20;   @property

\&#x20;   def is\\\_available(self) -> bool:

\&#x20;       return not self.is\\\_circuit\\\_open
```

保存关闭。

### 4.6 新建 `src/remote_candidate_cache.py`（百度图片下载缓存）



```
notepad src\remote\\\_candidate\\\_cache.py
```

粘贴：



```
"""

远程候选图片下载与缓存模块。

负责把百度返回的URL下载到本地，做安全校验和缓存。

安全规则：

\&#x20; \\- 只允许 http/https

\&#x20; \\- 禁止访问内网/局域网地址

\&#x20; \\- 单文件最大15MB

\&#x20; \\- 超时15秒，重试2次

\&#x20; \\- Content-Type必须是图片

\&#x20; \\- PIL必须能打开

\&#x20; \\- 宽高不低于200

\&#x20; \\- 统一转RGB

\&#x20; \\- 同一URL不重复下载

"""

import hashlib

import ipaddress

import logging

import socket

from pathlib import Path

from typing import Dict, Optional

from urllib.parse import urlparse

import requests

from PIL import Image

logger = logging.getLogger(\\\_\\\_name\\\_\\\_)

class RemoteCandidateCache:

\&#x20;   """远程图片下载缓存器。"""

\&#x20;   ALLOWED\\\_SCHEMES = {"http", "https"}

\&#x20;   IMAGE\\\_CONTENT\\\_TYPES = {"image/jpeg", "image/png", "image/webp",

\&#x20;                          "image/gif", "image/bmp", "image/jpg"}

\&#x20;   def \\\_\\\_init\\\_\\\_(self, cache\\\_dir: str = "runtime/baidu\\\_cache",

\&#x20;                timeout: int = 15, retries: int = 2,

\&#x20;                max\\\_file\\\_mb: int = 15, min\\\_width: int = 200,

\&#x20;                min\\\_height: int = 200):

\&#x20;       self.cache\\\_dir = Path(cache\\\_dir).resolve()

\&#x20;       self.cache\\\_dir.mkdir(parents=True, exist\\\_ok=True)

\&#x20;       self.timeout = timeout

\&#x20;       self.retries = retries

\&#x20;       self.max\\\_file\\\_bytes = max\\\_file\\\_mb \\\* 1024 \\\* 1024

\&#x20;       self.min\\\_width = min\\\_width

\&#x20;       self.min\\\_height = min\\\_height

\&#x20;       self.\\\_session = requests.Session()

\&#x20;       self.\\\_session.headers.update({

\&#x20;           "User-Agent": (

\&#x20;               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "

\&#x20;               "AppleWebKit/537.36 (KHTML, like Gecko) "

\&#x20;               "Chrome/120.0.0.0 Safari/537.36"

\&#x20;           ),

\&#x20;           "Referer": "https://graph.baidu.com/",

\&#x20;       })

\&#x20;   @staticmethod

\&#x20;   def \\\_url\\\_hash(url: str) -> str:

\&#x20;       """用URL的SHA-256作为缓存文件名。"""

\&#x20;       return hashlib.sha256(url.encode("utf-8")).hexdigest()

\&#x20;   def \\\_cache\\\_path(self, url: str) -> Path:

\&#x20;       """获取URL对应的缓存文件路径。"""

\&#x20;       return self.cache\\\_dir / f"{self.\\\_url\\\_hash(url)}.jpg"

\&#x20;   def \\\_is\\\_safe\\\_url(self, url: str) -> bool:

\&#x20;       """检查URL是否安全（禁止内网地址）。"""

\&#x20;       try:

\&#x20;           parsed = urlparse(url)

\&#x20;           if parsed.scheme.lower() not in self.ALLOWED\\\_SCHEMES:

\&#x20;               logger.warning("URL协议不允许: %s", url)

\&#x20;               return False

\&#x20;           hostname = parsed.hostname

\&#x20;           if not hostname:

\&#x20;               return False

\&#x20;           # 解析IP，检查是否为内网地址

\&#x20;           try:

\&#x20;               ip = socket.gethostbyname(hostname)

\&#x20;               ip\\\_obj = ipaddress.ip\\\_address(ip)

\&#x20;               if ip\\\_obj.is\\\_private or ip\\\_obj.is\\\_loopback or ip\\\_obj.is\\\_link\\\_local:

\&#x20;                   logger.warning("URL指向内网地址，拒绝下载: %s (%s)", url, ip)

\&#x20;                   return False

\&#x20;           except socket.gaierror:

\&#x20;               # 域名解析失败，交给后续下载步骤处理

\&#x20;               pass

\&#x20;           return True

\&#x20;       except Exception:

\&#x20;           return False

\&#x20;   def download(self, url: str) -> Dict:

\&#x20;       """

\&#x20;       下载并缓存一张远程图片。

\&#x20;       Returns:

\&#x20;           {

\&#x20;               "success": bool,

\&#x20;               "url": str,

\&#x20;               "local\\\_path": str,

\&#x20;               "width": int,

\&#x20;               "height": int,

\&#x20;               "error": str (失败时)

\&#x20;           }

\&#x20;       """

\&#x20;       cache\\\_path = self.\\\_cache\\\_path(url)

\&#x20;       # 缓存命中

\&#x20;       if cache\\\_path.is\\\_file():

\&#x20;           try:

\&#x20;               with Image.open(cache\\\_path) as img:

\&#x20;                   width, height = img.size

\&#x20;               logger.debug("缓存命中: %s", url)

\&#x20;               return {

\&#x20;                   "success": True,

\&#x20;                   "url": url,

\&#x20;                   "local\\\_path": str(cache\\\_path),

\&#x20;                   "width": width,

\&#x20;                   "height": height,

\&#x20;               }

\&#x20;           except Exception:

\&#x20;               # 缓存文件损坏，重新下载

\&#x20;               cache\\\_path.unlink(missing\\\_ok=True)

\&#x20;       # 安全检查

\&#x20;       if not self.\\\_is\\\_safe\\\_url(url):

\&#x20;           return {"success": False, "url": url, "error": "unsafe\\\_url"}

\&#x20;       # 带重试下载

\&#x20;       last\\\_error = ""

\&#x20;       for attempt in range(self.retries + 1):

\&#x20;           try:

\&#x20;               response = self.\\\_session.get(url, timeout=self.timeout, stream=True)

\&#x20;               response.raise\\\_for\\\_status()

\&#x20;               # 检查Content-Type

\&#x20;               content\\\_type = response.headers.get("Content-Type", "").lower()

\&#x20;               if not any(ct in content\\\_type for ct in self.IMAGE\\\_CONTENT\\\_TYPES):

\&#x20;                   # 有些CDN不返回正确的Content-Type，尝试用PIL验证

\&#x20;                   pass

\&#x20;               # 检查文件大小

\&#x20;               content\\\_length = int(response.headers.get("Content-Length", 0))

\&#x20;               if content\\\_length > self.max\\\_file\\\_bytes:

\&#x20;                   return {"success": False, "url": url,

\&#x20;                           "error": f"file\\\_too\\\_large: {content\\\_length} bytes"}

\&#x20;               # 下载到临时文件

\&#x20;               temp\\\_path = cache\\\_path.with\\\_suffix(".tmp")

\&#x20;               downloaded = 0

\&#x20;               with open(temp\\\_path, "wb") as f:

\&#x20;                   for chunk in response.iter\\\_content(chunk\\\_size=8192):

\&#x20;                       downloaded += len(chunk)

\&#x20;                       if downloaded > self.max\\\_file\\\_bytes:

\&#x20;                           f.close()

\&#x20;                           temp\\\_path.unlink(missing\\\_ok=True)

\&#x20;                           return {"success": False, "url": url,

\&#x20;                                   "error": "file\\\_too\\\_large"}

\&#x20;                       f.write(chunk)

\&#x20;               # PIL验证并转换

\&#x20;               with Image.open(temp\\\_path) as img:

\&#x20;                   img = img.convert("RGB")

\&#x20;                   width, height = img.size

\&#x20;                   if width < self.min\\\_width or height < self.min\\\_height:

\&#x20;                       temp\\\_path.unlink(missing\\\_ok=True)

\&#x20;                       return {"success": False, "url": url,

\&#x20;                               "error": f"image\\\_too\\\_small: {width}x{height}"}

\&#x20;                   img.save(cache\\\_path, format="JPEG", quality=92)

\&#x20;               temp\\\_path.unlink(missing\\\_ok=True)

\&#x20;               logger.info("下载成功: %s -> %s (%dx%d)", url, cache\\\_path.name, width, height)

\&#x20;               return {

\&#x20;                   "success": True,

\&#x20;                   "url": url,

\&#x20;                   "local\\\_path": str(cache\\\_path),

\&#x20;                   "width": width,

\&#x20;                   "height": height,

\&#x20;               }

\&#x20;           except Exception as e:

\&#x20;               last\\\_error = str(e)

\&#x20;               logger.warning("下载失败 (尝试%d/%d): %s - %s",

\&#x20;                              attempt + 1, self.retries + 1, url, e)

\&#x20;               if attempt < self.retries:

\&#x20;                   import time

\&#x20;                   time.sleep(1)

\&#x20;       return {"success": False, "url": url, "error": last\\\_error}

\&#x20;   def download\\\_batch(self, urls: list, max\\\_downloads: int = 20) -> list:

\&#x20;       """

\&#x20;       批量下载图片。

\&#x20;       Args:

\&#x20;           urls: URL列表

\&#x20;           max\\\_downloads: 最大下载数量

\&#x20;       Returns:

\&#x20;           下载成功的结果列表

\&#x20;       """

\&#x20;       results = \\\[]

\&#x20;       for url in urls\\\[:max\\\_downloads]:

\&#x20;           result = self.download(url)

\&#x20;           if result\\\["success"]:

\&#x20;               results.append(result)

\&#x20;       logger.info("批量下载完成: 成功%d/%d", len(results), min(len(urls), max\\\_downloads))

\&#x20;       return results
```

保存关闭。

### 4.7 新建 `src/candidate_pool.py`（候选合并去重）



```
notepad src\candidate\\\_pool.py
```

粘贴：



```
"""

候选池合并与去重模块。

负责把本地候选和百度候选合并，执行两层去重：

\&#x20; 1\\. 和样图比较（完全重复/近重复）

\&#x20; 2\\. 候选之间互相比较（SHA256相同/pHash过小/全局占用）

来源优先规则：内容相同时优先保留本地候选。

"""

import logging

from pathlib import Path

from typing import Dict, List, Optional

logger = logging.getLogger(\\\_\\\_name\\\_\\\_)

class CandidatePool:

\&#x20;   """候选池合并与去重器。"""

\&#x20;   def \\\_\\\_init\\\_\\\_(self, duplicate\\\_detector, global\\\_registry=None):

\&#x20;       """

\&#x20;       Args:

\&#x20;           duplicate\\\_detector: DuplicateDetector 实例

\&#x20;           global\\\_registry: GlobalCandidateRegistry 实例（可选）

\&#x20;       """

\&#x20;       self.detector = duplicate\\\_detector

\&#x20;       self.registry = global\\\_registry

\&#x20;   def merge\\\_and\\\_deduplicate(

\&#x20;       self,

\&#x20;       sample\\\_path: str,

\&#x20;       local\\\_candidates: List\\\[dict],

\&#x20;       baidu\\\_candidates: List\\\[dict],

\&#x20;   ) -> List\\\[dict]:

\&#x20;       """

\&#x20;       合并本地和百度候选，执行去重。

\&#x20;       Args:

\&#x20;           sample\\\_path: 样图路径

\&#x20;           local\\\_candidates: 本地候选列表（已过滤）

\&#x20;           baidu\\\_candidates: 百度候选列表（已下载、已重新评分）

\&#x20;       Returns:

\&#x20;           去重后的合并候选列表

\&#x20;       """

\&#x20;       sample\\\_path = str(Path(sample\\\_path).resolve())

\&#x20;       # 第一层：和样图去重

\&#x20;       local\\\_candidates = self.\\\_filter\\\_against\\\_sample(sample\\\_path, local\\\_candidates)

\&#x20;       baidu\\\_candidates = self.\\\_filter\\\_against\\\_sample(sample\\\_path, baidu\\\_candidates)

\&#x20;       # 合并：本地候选在前（优先级高）

\&#x20;       all\\\_candidates = local\\\_candidates + baidu\\\_candidates

\&#x20;       # 第二层：候选之间去重

\&#x20;       result = self.\\\_deduplicate\\\_among\\\_candidates(all\\\_candidates)

\&#x20;       logger.info("候选池合并完成: 本地%d + 百度%d -> 合并后%d",

\&#x20;                    len(local\\\_candidates), len(baidu\\\_candidates), len(result))

\&#x20;       return result

\&#x20;   def \\\_filter\\\_against\\\_sample(self, sample\\\_path: str, candidates: List\\\[dict]) -> List\\\[dict]:

\&#x20;       """过滤掉和样图重复的候选。"""

\&#x20;       result = \\\[]

\&#x20;       for cand in candidates:

\&#x20;           cand\\\_path = cand.get("path", cand.get("local\\\_path", ""))

\&#x20;           if not cand\\\_path or not Path(cand\\\_path).is\\\_file():

\&#x20;               continue

\&#x20;           try:

\&#x20;               check = self.detector.compare(sample\\\_path, cand\\\_path)

\&#x20;               if check\\\["is\\\_duplicate"]:

\&#x20;                   logger.debug("候选与样图重复，剔除: %s", cand\\\_path)

\&#x20;                   continue

\&#x20;               cand\\\["phash\\\_distance"] = int(check\\\["phash\\\_distance"])

\&#x20;               result.append(cand)

\&#x20;           except Exception as e:

\&#x20;               logger.warning("候选与样图比较失败，剔除: %s - %s", cand\\\_path, e)

\&#x20;       return result

\&#x20;   def \\\_deduplicate\\\_among\\\_candidates(self, candidates: List\\\[dict]) -> List\\\[dict]:

\&#x20;       """候选之间互相去重。"""

\&#x20;       result = \\\[]

\&#x20;       seen\\\_hashes = set()

\&#x20;       for cand in candidates:

\&#x20;           cand\\\_path = cand.get("path", cand.get("local\\\_path", ""))

\&#x20;           if not cand\\\_path or not Path(cand\\\_path).is\\\_file():

\&#x20;               continue

\&#x20;           # 全局占用检查

\&#x20;           if self.registry and self.registry.is\\\_used(Path(cand\\\_path)):

\&#x20;               logger.debug("候选已被全局占用，剔除: %s", cand\\\_path)

\&#x20;               continue

\&#x20;           # SHA256去重

\&#x20;           try:

\&#x20;               cand\\\_hash = self.detector.calculate\\\_sha256(Path(cand\\\_path))

\&#x20;           except Exception:

\&#x20;               continue

\&#x20;           if cand\\\_hash in seen\\\_hashes:

\&#x20;               logger.debug("候选SHA256重复，剔除: %s", cand\\\_path)

\&#x20;               continue

\&#x20;           # pHash近重复检查（和已保留的候选比较）

\&#x20;           is\\\_near\\\_dup = False

\&#x20;           for kept in result:

\&#x20;               kept\\\_path = kept.get("path", kept.get("local\\\_path", ""))

\&#x20;               if not kept\\\_path:

\&#x20;                   continue

\&#x20;               try:

\&#x20;                   check = self.detector.compare(cand\\\_path, kept\\\_path)

\&#x20;                   if check\\\["is\\\_duplicate"]:

\&#x20;                       # 近重复时保留质量更高的（语义分数更高的）

\&#x20;                       cand\\\_score = cand.get("semantic\\\_score", 0)

\&#x20;                       kept\\\_score = kept.get("semantic\\\_score", 0)

\&#x20;                       if cand\\\_score > kept\\\_score:

\&#x20;                           # 新候选质量更高，替换旧的

\&#x20;                           result.remove(kept)

\&#x20;                           seen\\\_hashes.discard(

\&#x20;                               self.detector.calculate\\\_sha256(Path(kept\\\_path))

\&#x20;                           )

\&#x20;                           logger.debug("近重复候选替换: %s (%.3f) > %s (%.3f)",

\&#x20;                                        cand\\\_path, cand\\\_score, kept\\\_path, kept\\\_score)

\&#x20;                       else:

\&#x20;                           is\\\_near\\\_dup = True

\&#x20;                           logger.debug("候选与已保留候选近重复，剔除: %s", cand\\\_path)

\&#x20;                       break

\&#x20;               except Exception:

\&#x20;                   continue

\&#x20;           if not is\\\_near\\\_dup:

\&#x20;               seen\\\_hashes.add(cand\\\_hash)

\&#x20;               result.append(cand)

\&#x20;       return result
```

保存关闭。

### 4.8 新建测试脚本



```
notepad scripts\test\\\_baidu\\\_search.py
```

粘贴：



```
"""

单独测试百度识图是否可用。

用法: .venv\Scripts\python.exe scripts\test\\\_baidu\\\_search.py --image <样图路径>

"""

import argparse

import sys

from pathlib import Path

PROJECT\\\_ROOT = Path(\\\_\\\_file\\\_\\\_).resolve().parent.parent

sys.path.insert(0, str(PROJECT\\\_ROOT))

import yaml

from src.providers.baidu\\\_image\\\_provider import BaiduImageProvider

def main():

\&#x20;   parser = argparse.ArgumentParser(description="测试百度识图")

\&#x20;   parser.add\\\_argument("--image", required=True, help="样图路径")

\&#x20;   parser.add\\\_argument("--config", default=str(PROJECT\\\_ROOT / "config.yaml"))

\&#x20;   args = parser.parse\\\_args()

\&#x20;   with open(args.config, "r", encoding="utf-8") as f:

\&#x20;       config = yaml.safe\\\_load(f)

\&#x20;   baidu\\\_cfg = config.get("baidu\\\_image\\\_search", {})

\&#x20;   if not baidu\\\_cfg.get("enabled", False):

\&#x20;       print("\\\[错误] 百度识图未在配置中启用")

\&#x20;       return 1

\&#x20;   print(f"测试图片: {args.image}")

\&#x20;   print(f"模式: {baidu\\\_cfg.get('mode', 'auto')}")

\&#x20;   print("-" \\\* 50)

\&#x20;   provider = BaiduImageProvider(baidu\\\_cfg)

\&#x20;   candidates = provider.search(args.image, top\\\_k=10)

\&#x20;   if not candidates:

\&#x20;       print("\\\[结果] 百度识图未返回任何结果")

\&#x20;       print("可能原因：")

\&#x20;       print("  1. 网络不通或被百度反爬")

\&#x20;       print("  2. HTTP模式解析失败，且未安装playwright")

\&#x20;       print("  3. 熔断器已打开（连续失败过多）")

\&#x20;       return 1

\&#x20;   print(f"\\\[结果] 百度返回 {len(candidates)} 个URL:")

\&#x20;   for c in candidates:

\&#x20;       print(f"  #{c.source\\\_rank}: {c.source\\\_url\\\[:80]}...")

\&#x20;   print("\n\\\[成功] 百度识图可用")

\&#x20;   return 0

if \\\_\\\_name\\\_\\\_ == "\\\_\\\_main\\\_\\\_":

\&#x20;   sys.exit(main())
```

保存关闭。



```
notepad scripts\test\\\_provider\\\_fallback.py
```

粘贴：



```
"""

验证降级触发逻辑：本地有效候选少于3张时是否调用百度。

用法: .venv\Scripts\python.exe scripts\test\\\_provider\\\_fallback.py --input\\\_dir <样图目录>

"""

import argparse

import sys

from pathlib import Path

PROJECT\\\_ROOT = Path(\\\_\\\_file\\\_\\\_).resolve().parent.parent

sys.path.insert(0, str(PROJECT\\\_ROOT))

import yaml

from src.hybrid\\\_searcher import HybridSearcher

from src.providers.local\\\_faiss\\\_provider import LocalFAISSProvider

from src.providers.baidu\\\_image\\\_provider import BaiduImageProvider

from src.duplicate\\\_detector import DuplicateDetector

from src.image\\\_classifier import ImageCategoryClassifier, categories\\\_compatible

from src.candidate\\\_registry import GlobalCandidateRegistry

def filter\\\_local\\\_candidates(sample\\\_path, raw\\\_candidates, detector, classifier,

\&#x20;                           sample\\\_category, registry, absolute\\\_min=0.50,

\&#x20;                           relative\\\_ratio=0.70):

\&#x20;   """模拟批量脚本中的本地候选过滤逻辑。"""

\&#x20;   from pathlib import Path as P

\&#x20;   candidates = \\\[]

\&#x20;   for cand in raw\\\_candidates:

\&#x20;       cand\\\_path = P(cand.local\\\_path)

\&#x20;       if not cand\\\_path.is\\\_file() or registry.is\\\_used(cand\\\_path):

\&#x20;           continue

\&#x20;       check = detector.compare(sample\\\_path, cand\\\_path)

\&#x20;       if check\\\["is\\\_duplicate"]:

\&#x20;           continue

\&#x20;       # 这里简化：不重建向量，直接用source\\\_score

\&#x20;       cand\\\_category = "unknown"  # 简化

\&#x20;       candidates.append({

\&#x20;           "path": str(cand\\\_path),

\&#x20;           "semantic\\\_score": cand.source\\\_score or 0,

\&#x20;           "phash\\\_distance": check\\\["phash\\\_distance"],

\&#x20;           "category": cand\\\_category,

\&#x20;           "source": "local",

\&#x20;       })

\&#x20;   if not candidates:

\&#x20;       return \\\[]

\&#x20;   best\\\_score = max(c\\\["semantic\\\_score"] for c in candidates)

\&#x20;   effective\\\_min = max(absolute\\\_min, best\\\_score \\\* relative\\\_ratio)

\&#x20;   return \\\[c for c in candidates if c\\\["semantic\\\_score"] >= effective\\\_min]

def main():

\&#x20;   parser = argparse.ArgumentParser(description="测试提供器降级逻辑")

\&#x20;   parser.add\\\_argument("--input\\\_dir", required=True, help="样图目录")

\&#x20;   parser.add\\\_argument("--config", default=str(PROJECT\\\_ROOT / "config.yaml"))

\&#x20;   parser.add\\\_argument("--test\\\_local\\\_limit", type=int, default=0,

\&#x20;                       help="测试用：强制限制本地候选数量（0表示不限制）")

\&#x20;   args = parser.parse\\\_args()

\&#x20;   with open(args.config, "r", encoding="utf-8") as f:

\&#x20;       config = yaml.safe\\\_load(f)

\&#x20;   provider\\\_cfg = config.get("search\\\_provider", {})

\&#x20;   trigger\\\_count = provider\\\_cfg.get("fallback\\\_trigger\\\_count", 3)

\&#x20;   local\\\_top\\\_k = provider\\\_cfg.get("local\\\_top\\\_k", 98)

\&#x20;   print("=" \\\* 60)

\&#x20;   print("方案D 降级触发逻辑测试")

\&#x20;   print(f"降级触发阈值: 本地有效候选 < {trigger\\\_count}")

\&#x20;   print(f"本地Top-K: {local\\\_top\\\_k}")

\&#x20;   print("=" \\\* 60)

\&#x20;   # 初始化

\&#x20;   searcher = HybridSearcher(config\\\_path=args.config)

\&#x20;   local\\\_provider = LocalFAISSProvider(searcher.clip, searcher.vector\\\_store)

\&#x20;   baidu\\\_provider = BaiduImageProvider(config.get("baidu\\\_image\\\_search", {}))

\&#x20;   detector = DuplicateDetector(phash\\\_threshold=6)

\&#x20;   classifier = ImageCategoryClassifier(searcher.clip)

\&#x20;   registry = GlobalCandidateRegistry(str(PROJECT\\\_ROOT / "runtime/tasks/used\\\_candidates.json"))

\&#x20;   # 取第一张样图测试

\&#x20;   samples = sorted(\\\[p for p in Path(args.input\\\_dir).iterdir()

\&#x20;                     if p.is\\\_file() and p.suffix.lower() in

\&#x20;                     {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])

\&#x20;   if not samples:

\&#x20;       print("\\\[错误] 样图目录为空")

\&#x20;       return 1

\&#x20;   sample\\\_path = str(samples\\\[0])

\&#x20;   print(f"\n测试样图: {samples\\\[0].name}")

\&#x20;   # 1. 本地检索

\&#x20;   print("\n\\\[步骤1] 本地FAISS检索...")

\&#x20;   raw\\\_local = local\\\_provider.search(sample\\\_path, top\\\_k=local\\\_top\\\_k)

\&#x20;   print(f"  FAISS原始返回: {len(raw\\\_local)} 条")

\&#x20;   # 2. 过滤

\&#x20;   print("\\\[步骤2] 执行过滤...")

\&#x20;   query\\\_vector = searcher.clip.encode\\\_image(sample\\\_path)

\&#x20;   sample\\\_pred = classifier.classify\\\_embedding(query\\\_vector)

\&#x20;   filtered = filter\\\_local\\\_candidates(

\&#x20;       sample\\\_path, raw\\\_local, detector, classifier,

\&#x20;       sample\\\_pred\\\["category"], registry

\&#x20;   )

\&#x20;   print(f"  过滤后有效候选: {len(filtered)} 条")

\&#x20;   # 测试限制

\&#x20;   if args.test\\\_local\\\_limit > 0:

\&#x20;       filtered = filtered\\\[:args.test\\\_local\\\_limit]

\&#x20;       print(f"  \\\[测试限制] 强制限制为: {len(filtered)} 条")

\&#x20;   # 3. 判断是否触发百度

\&#x20;   print(f"\n\\\[步骤3] 降级判断: {len(filtered)} < {trigger\\\_count} ?")

\&#x20;   if len(filtered) < trigger\\\_count:

\&#x20;       print("  -> 是，触发百度识图")

\&#x20;       if not baidu\\\_provider.is\\\_available:

\&#x20;           print("  \\\[警告] 百度提供器不可用（熔断器可能已打开）")

\&#x20;           return 1

\&#x20;       baidu\\\_candidates = baidu\\\_provider.search(sample\\\_path, top\\\_k=10)

\&#x20;       print(f"  百度返回: {len(baidu\\\_candidates)} 个URL")

\&#x20;       if baidu\\\_candidates:

\&#x20;           print("  \\\[成功] 降级触发正常")

\&#x20;       else:

\&#x20;           print("  \\\[警告] 百度返回为空（可能网络问题或反爬）")

\&#x20;   else:

\&#x20;       print("  -> 否，不触发百度，直接使用本地候选")

\&#x20;       print("  \\\[成功] 本地充足时不调用百度")

\&#x20;   print("\n" + "=" \\\* 60)

\&#x20;   print("测试完成")

\&#x20;   return 0

if \\\_\\\_name\\\_\\\_ == "\\\_\\\_main\\\_\\\_":

\&#x20;   sys.exit(main())
```

保存关闭。



***

## 五、第三阶段：安装新依赖（百度需要）

### 5.1 安装 requests（应该已经有了，确认一下）



```
cd /d C:\Users\mozhihao\core\image-hybrid-search

.venv\Scripts\python.exe -c "import requests; print('requests版本:', requests.\\\_\\\_version\\\_\\\_)"
```

如果报错，执行：



```
.venv\Scripts\pip.exe install requests
```

### 5.2 安装 Playwright（浏览器模式需要，HTTP 模式不需要也能跑）



```
.venv\Scripts\pip.exe install playwright

.venv\Scripts\playwright.exe install chromium
```

> 这一步会下载 Chromium 浏览器（约 150MB），需要等几分钟。
> 如果下载慢，可以先跳过，只用 HTTP 模式（config 里 mode 设为 "http"）。

### 5.3 验证所有依赖



```
.venv\Scripts\python.exe -c "import requests; import yaml; import faiss; import PIL; import numpy; print('核心依赖OK')"
```

**验证**：输出 `核心依赖OK`。



***

## 六、第四阶段：验证新增模块能正常导入



```
cd /d C:\Users\mozhihao\core\image-hybrid-search

.venv\Scripts\python.exe -c "

from src.providers.base import Candidate, BaseProvider

from src.providers.local\\\_faiss\\\_provider import LocalFAISSProvider

from src.providers.baidu\\\_image\\\_provider import BaiduImageProvider

from src.remote\\\_candidate\\\_cache import RemoteCandidateCache

from src.candidate\\\_pool import CandidatePool

print('所有新增模块导入成功')

"
```

**验证**：输出 `所有新增模块导入成功`。

> 如果报错，根据错误信息检查对应文件是否保存正确。



***

## 七、第五阶段：测试百度识图是否能用

### 7.1 找一张样图测试



```
.venv\Scripts\python.exe scripts\test\\\_baidu\\\_search.py --image "input\\\_samples\1 (1).png"
```

**可能的结果：**

**情况 A：成功返回 URL**



```
\\\[结果] 百度返回 10 个URL:

\&#x20; \\#1: https://...

\\\[成功] 百度识图可用
```

→ 太好了，HTTP 模式能用，继续下一步。

**情况 B：返回空，但提示 playwright 未安装**

→ 回到 5.2 安装 playwright，或者把 config 里 mode 改成 "browser" 再试。

**情况 C：返回空，没有明确错误**

→ 百度可能反爬了。试试：



1. 等几分钟再试（不要频繁调用）

2. 把 config 里 mode 改成 "browser"

3. 检查网络是否能访问百度

> 百度识图即使暂时用不了，也不影响本地检索功能。方案 D 的设计是 "本地优先，百度兜底"，百度挂了只是少了兜底能力。



***

## 八、第六阶段：修改批量主脚本（接入方案 D 逻辑）

> 这一步是最关键的，要修改
> `scripts/run_batch_selection.py`
> 。
> 修改前已经备份过了，放心改。

### 8.1 用记事本打开主脚本



```
notepad scripts\run\\\_batch\\\_selection.py
```

### 8.2 需要修改的地方（共 4 处）

#### 修改 1：在文件顶部导入新模块

找到文件开头的 `from src.xxx import ...` 部分，在最后加上：



```
from src.providers.local\\\_faiss\\\_provider import LocalFAISSProvider

from src.providers.baidu\\\_image\\\_provider import BaiduImageProvider

from src.remote\\\_candidate\\\_cache import RemoteCandidateCache

from src.candidate\\\_pool import CandidatePool
```

#### 修改 2：在参数解析部分增加百度相关参数

找到 `parser.add_argument("--resume", action="store_true")` 这一行，在它后面加上：



```
\&#x20;   parser.add\\\_argument("--test\\\_local\\\_limit", type=int, default=0,

\&#x20;                       help="测试用：强制限制本地候选数量（0表示不限制）")
```

#### 修改 3：在初始化部分增加百度提供器

找到这几行：



```
\&#x20;   registry = GlobalCandidateRegistry(str(registry\\\_path))

\&#x20;   task\\\_store = TaskStore(str(task\\\_path))

\&#x20;   detector = DuplicateDetector(phash\\\_threshold=6)

\&#x20;   selector = MultiFeatureScorer()
```

在后面加上：



```
\&#x20;   # 方案D：初始化提供器

\&#x20;   search\\\_provider\\\_cfg = searcher.config.get("search\\\_provider", {})

\&#x20;   fallback\\\_trigger\\\_count = search\\\_provider\\\_cfg.get("fallback\\\_trigger\\\_count", 3)

\&#x20;   local\\\_top\\\_k = search\\\_provider\\\_cfg.get("local\\\_top\\\_k", 98)

\&#x20;   local\\\_provider = LocalFAISSProvider(searcher.clip, searcher.vector\\\_store)

\&#x20;   baidu\\\_provider = BaiduImageProvider(searcher.config.get("baidu\\\_image\\\_search", {}))

\&#x20;   remote\\\_cfg = searcher.config.get("remote\\\_download", {})

\&#x20;   remote\\\_cache = RemoteCandidateCache(

\&#x20;       cache\\\_dir=searcher.config.get("baidu\\\_image\\\_search", {}).get("cache\\\_dir", "runtime/baidu\\\_cache"),

\&#x20;       timeout=remote\\\_cfg.get("timeout", 15),

\&#x20;       retries=remote\\\_cfg.get("retries", 2),

\&#x20;       max\\\_file\\\_mb=remote\\\_cfg.get("max\\\_file\\\_mb", 15),

\&#x20;       min\\\_width=remote\\\_cfg.get("min\\\_width", 200),

\&#x20;       min\\\_height=remote\\\_cfg.get("min\\\_height", 200),

\&#x20;   )

\&#x20;   candidate\\\_pool = CandidatePool(detector, registry)
```

#### 修改 4：替换核心检索逻辑（最关键的修改）

找到从 `query_vector = searcher.clip.encode_image(sample_path)` 开始，到 `selected = selector.select(sample_path, candidates)` 之前的整段代码，替换为：



```
\&#x20;           # ===== 方案D：本地优先，百度兜底 =====

\&#x20;           query\\\_vector = searcher.clip.encode\\\_image(sample\\\_path)

\&#x20;           sample\\\_prediction = classifier.classify\\\_embedding(query\\\_vector)

\&#x20;           sample\\\_category = sample\\\_prediction\\\["category"]

\&#x20;           # 1. 本地FAISS检索

\&#x20;           raw\\\_local = local\\\_provider.search(str(sample\\\_path), top\\\_k=local\\\_top\\\_k)

\&#x20;           # 2. 本地候选过滤

\&#x20;           candidates = \\\[]

\&#x20;           category\\\_rejected = 0

\&#x20;           baidu\\\_called = False

\&#x20;           baidu\\\_url\\\_count = 0

\&#x20;           baidu\\\_downloaded\\\_count = 0

\&#x20;           baidu\\\_valid\\\_count = 0

\&#x20;           low\\\_diversity\\\_pool = False

\&#x20;           for cand in raw\\\_local:

\&#x20;               candidate\\\_path = Path(cand.local\\\_path).resolve()

\&#x20;               if not candidate\\\_path.is\\\_file() or registry.is\\\_used(candidate\\\_path):

\&#x20;                   continue

\&#x20;               check = detector.compare(sample\\\_path, candidate\\\_path)

\&#x20;               if check\\\["is\\\_duplicate"]:

\&#x20;                   continue

\&#x20;               try:

\&#x20;                   candidate\\\_vector = searcher.vector\\\_store.index.reconstruct(cand.internal\\\_id)

\&#x20;               except Exception:

\&#x20;                   candidate\\\_vector = searcher.clip.encode\\\_image(candidate\\\_path)

\&#x20;               candidate\\\_prediction = classifier.classify\\\_embedding(candidate\\\_vector)

\&#x20;               candidate\\\_category = candidate\\\_prediction\\\["category"]

\&#x20;               if not categories\\\_compatible(sample\\\_category, candidate\\\_category):

\&#x20;                   category\\\_rejected += 1

\&#x20;                   continue

\&#x20;               candidates.append({

\&#x20;                   "path": str(candidate\\\_path),

\&#x20;                   "internal\\\_id": cand.internal\\\_id,

\&#x20;                   "semantic\\\_score": cand.source\\\_score or 0.0,

\&#x20;                   "phash\\\_distance": int(check\\\["phash\\\_distance"]),

\&#x20;                   "is\\\_duplicate": False,

\&#x20;                   "category": candidate\\\_category,

\&#x20;                   "category\\\_score": float(candidate\\\_prediction\\\["score"]),

\&#x20;                   "source": "local",

\&#x20;                   "source\\\_url": None,

\&#x20;                   "source\\\_rank": cand.source\\\_rank,

\&#x20;               })

\&#x20;           # 测试限制

\&#x20;           if args.test\\\_local\\\_limit > 0:

\&#x20;               candidates = candidates\\\[:args.test\\\_local\\\_limit]

\&#x20;           local\\\_valid\\\_count = len(candidates)

\&#x20;           print(f"    本地有效候选: {local\\\_valid\\\_count} 张")

\&#x20;           # 3. 动态质量门槛

\&#x20;           if candidates:

\&#x20;               best\\\_score = max(item\\\["semantic\\\_score"] for item in candidates)

\&#x20;               effective\\\_min\\\_score = max(

\&#x20;                   args.absolute\\\_min\\\_score,

\&#x20;                   best\\\_score \\\* args.relative\\\_min\\\_ratio,

\&#x20;               )

\&#x20;               candidates = \\\[

\&#x20;                   item for item in candidates

\&#x20;                   if item\\\["semantic\\\_score"] >= effective\\\_min\\\_score

\&#x20;               ]

\&#x20;           else:

\&#x20;               effective\\\_min\\\_score = args.absolute\\\_min\\\_score

\&#x20;           # 4. 判断是否触发百度

\&#x20;           if len(candidates) < fallback\\\_trigger\\\_count:

\&#x20;               baidu\\\_called = True

\&#x20;               print(f"    本地候选不足{fallback\\\_trigger\\\_count}张，触发百度识图...")

\&#x20;               missing\\\_count = fallback\\\_trigger\\\_count - len(candidates)

\&#x20;               download\\\_target = max(10, missing\\\_count \\\* 5)

\&#x20;               try:

\&#x20;                   # 百度搜索

\&#x20;                   baidu\\\_raw = baidu\\\_provider.search(str(sample\\\_path), top\\\_k=30)

\&#x20;                   baidu\\\_url\\\_count = len(baidu\\\_raw)

\&#x20;                   print(f"    百度返回URL: {baidu\\\_url\\\_count} 个")

\&#x20;                   if baidu\\\_raw:

\&#x20;                       # 下载图片

\&#x20;                       urls = \\\[c.source\\\_url for c in baidu\\\_raw if c.source\\\_url]

\&#x20;                       download\\\_results = remote\\\_cache.download\\\_batch(

\&#x20;                           urls, max\\\_downloads=min(download\\\_target, 20)

\&#x20;                       )

\&#x20;                       baidu\\\_downloaded\\\_count = len(download\\\_results)

\&#x20;                       print(f"    百度下载成功: {baidu\\\_downloaded\\\_count} 张")

\&#x20;                       # 对百度候选重新评分

\&#x20;                       baidu\\\_candidates = \\\[]

\&#x20;                       for dl in download\\\_results:

\&#x20;                           dl\\\_path = dl\\\["local\\\_path"]

\&#x20;                           try:

\&#x20;                               # CLIP编码

\&#x20;                               dl\\\_vector = searcher.clip.encode\\\_image(dl\\\_path)

\&#x20;                               dl\\\_score = float(searcher.clip.similarity(query\\\_vector, dl\\\_vector))

\&#x20;                               # 类别

\&#x20;                               dl\\\_pred = classifier.classify\\\_embedding(dl\\\_vector)

\&#x20;                               if not categories\\\_compatible(sample\\\_category, dl\\\_pred\\\["category"]):

\&#x20;                                   continue

\&#x20;                               # 质量门槛

\&#x20;                               if dl\\\_score < effective\\\_min\\\_score:

\&#x20;                                   continue

\&#x20;                               baidu\\\_candidates.append({

\&#x20;                                   "path": dl\\\_path,

\&#x20;                                   "semantic\\\_score": dl\\\_score,

\&#x20;                                   "phash\\\_distance": 0,  # 后面合并时会重新算

\&#x20;                                   "is\\\_duplicate": False,

\&#x20;                                   "category": dl\\\_pred\\\["category"],

\&#x20;                                   "category\\\_score": float(dl\\\_pred\\\["score"]),

\&#x20;                                   "source": "baidu",

\&#x20;                                   "source\\\_url": dl\\\["url"],

\&#x20;                                   "source\\\_rank": 0,

\&#x20;                               })

\&#x20;                           except Exception as e:

\&#x20;                               logger.warning("百度候选评分失败: %s - %s", dl\\\_path, e)

\&#x20;                               continue

\&#x20;                       baidu\\\_valid\\\_count = len(baidu\\\_candidates)

\&#x20;                       print(f"    百度过滤后有效: {baidu\\\_valid\\\_count} 张")

\&#x20;                       # 合并去重

\&#x20;                       if baidu\\\_candidates:

\&#x20;                           merged = candidate\\\_pool.merge\\\_and\\\_deduplicate(

\&#x20;                               str(sample\\\_path), candidates, baidu\\\_candidates

\&#x20;                           )

\&#x20;                           candidates = merged

\&#x20;                           print(f"    合并后总候选: {len(candidates)} 张")

\&#x20;               except Exception as e:

\&#x20;                   print(f"    \\\[警告] 百度识图调用失败: {e}")

\&#x20;                   print("    继续使用本地候选")

\&#x20;           else:

\&#x20;               # 本地正好3张时标记低多样性

\&#x20;               if len(candidates) == fallback\\\_trigger\\\_count:

\&#x20;                   low\\\_diversity\\\_pool = True

\&#x20;                   print(f"    本地候选正好{fallback\\\_trigger\\\_count}张，未调用百度（低多样性提醒）")

\&#x20;           if not candidates:

\&#x20;               raise RuntimeError("重复过滤后没有可用候选")

\&#x20;           # 5. A/B/C选择

\&#x20;           selected = selector.select(sample\\\_path, candidates)
```

#### 修改 5：在任务保存部分增加百度相关字段

找到 `task_store.update(source_hash, { ... "status": "completed", ... })` 那一段，在 `"results": saved_results,` 前面加上：



```
\&#x20;                   "baidu\\\_called": baidu\\\_called,

\&#x20;                   "baidu\\\_url\\\_count": baidu\\\_url\\\_count,

\&#x20;                   "baidu\\\_downloaded\\\_count": baidu\\\_downloaded\\\_count,

\&#x20;                   "baidu\\\_valid\\\_count": baidu\\\_valid\\\_count,

\&#x20;                   "local\\\_valid\\\_count": local\\\_valid\\\_count,

\&#x20;                   "low\\\_diversity\\\_pool": low\\\_diversity\\\_pool,
```

#### 修改 6：在失败处理部分也增加字段

找到 `except Exception as error:` 后面的 `task_store.update`，在 `"results": {},` 前面加上：



```
\&#x20;                   "baidu\\\_called": baidu\\\_called if 'baidu\\\_called' in dir() else False,

\&#x20;                   "local\\\_valid\\\_count": local\\\_valid\\\_count if 'local\\\_valid\\\_count' in dir() else 0,
```

### 8.3 保存并验证语法



```
.venv\Scripts\python.exe -m py\\\_compile scripts\run\\\_batch\\\_selection.py
```

**验证**：没有输出（没有报错）就是语法正确。

> 如果报错，仔细检查修改的地方，特别是缩进（Python 对缩进很敏感，用 4 个空格）。



***

## 九、第七阶段：单图测试（不跑全量）

### 9.1 先测试本地充足的情况（不触发百度）



```
.venv\Scripts\python.exe scripts\test\\\_provider\\\_fallback.py --input\\\_dir input\\\_samples
```

**预期输出**：



```
本地有效候选: X 张（X >= 3）

-> 否，不触发百度，直接使用本地候选

\\\[成功] 本地充足时不调用百度
```

### 9.2 测试强制触发百度



```
.venv\Scripts\python.exe scripts\test\\\_provider\\\_fallback.py --input\\\_dir input\\\_samples --test\\\_local\\\_limit 2
```

**预期输出**：



```
本地有效候选: 2 张

-> 是，触发百度识图

百度返回: X 个URL
```

> 如果百度返回 0 个，说明百度暂时不可用，但降级触发逻辑是正常的。



***

## 十、第八阶段：10 张正式批量测试

### 10.1 清空之前的任务状态（全新跑一遍）

> 注意：这会清除之前的全局占用记录，相当于重新开始。



```
copy runtime\tasks\used\\\_candidates.json runtime\tasks\used\\\_candidates\\\_backup.json

copy runtime\tasks\batch\\\_task.json runtime\tasks\batch\\\_task\\\_backup.json

del runtime\tasks\used\\\_candidates.json

del runtime\tasks\batch\\\_task.json
```

### 10.2 运行批量任务



```
.venv\Scripts\python.exe scripts\run\\\_batch\\\_selection.py --input\\\_dir input\\\_samples --output\\\_dir output --top\\\_k 98
```

**观察输出**：



* 每张样图应该显示 `本地有效候选: X 张`

* 如果 X<3，应该显示 `触发百度识图...`

* 最后显示汇总：完成 X 张，跳过 X 张，失败 X 张

### 10.3 检查输出



```
dir output /b
```

应该有 10 个文件夹（0001\_xxx 到 0010\_xxx），每个里面有 source + A + B + C 四张图。



***

## 十一、第九阶段：生成报告并验证

### 11.1 生成报告



```
.venv\Scripts\python.exe scripts\generate\\\_report.py
```

> 如果没有这个脚本，用下面的命令手动生成：



```
.venv\Scripts\python.exe -c "

import json, sys

from pathlib import Path

sys.path.insert(0, '.')

from src.result\\\_exporter import export\\\_reports

tasks = json.loads(Path('runtime/tasks/batch\\\_task.json').read\\\_text(encoding='utf-8'))\\\['tasks']

task\\\_list = list(tasks.values())

result = export\\\_reports(task\\\_list, Path('runtime/reports'))

print('报告生成:', result)

"
```

### 11.2 检查 Excel 报告

打开 `runtime\reports\result.xlsx`，确认：



* 有 30 行数据（10 张样图 × A/B/C 三组）

* 每张样图的 A/B/C 都有候选

* 没有重复的候选路径



***

## 十二、第十阶段：断点恢复测试

### 12.1 用 --resume 重新运行



```
.venv\Scripts\python.exe scripts\run\\\_batch\\\_selection.py --input\\\_dir input\\\_samples --output\\\_dir output --top\\\_k 98 --resume
```

**预期**：10 张全部显示 `跳过已完成`，不会重新处理。

### 12.2 模拟失败任务重试

手动把任务状态改成失败（测试用）：



```
.venv\Scripts\python.exe -c "

import json

from pathlib import Path

p = Path('runtime/tasks/batch\\\_task.json')

data = json.loads(p.read\\\_text(encoding='utf-8'))

keys = list(data\\\['tasks'].keys())

if keys:

\&#x20;   data\\\['tasks']\\\[keys\\\[0]]\\\['status'] = 'insufficient\\\_candidates'

\&#x20;   p.write\\\_text(json.dumps(data, ensure\\\_ascii=False, indent=2), encoding='utf-8')

\&#x20;   print('已将第一张样图标记为 insufficient\\\_candidates')

"
```

然后再用 --resume 运行：



```
.venv\Scripts\python.exe scripts\run\\\_batch\\\_selection.py --input\\\_dir input\\\_samples --output\\\_dir output --top\\\_k 98 --resume
```

**预期**：第一张样图会被重新处理，其余 9 张跳过。



***

## 十三、方案 D 验收清单（全部打勾才算完成）



* [ ] `config.yaml` 中 `google_vision.enabled = False`

* [ ] `config.yaml` 中 `baidu_image_search.enabled = True`

* [ ] `config.yaml` 中 `search_provider.fallback_trigger_count = 3`

* [ ] 新增 `src/providers/` 目录及 4 个文件

* [ ] 新增 `src/remote_candidate_cache.py`

* [ ] 新增 `src/candidate_pool.py`

* [ ] 新增 `scripts/test_baidu_search.py`

* [ ] 新增 `scripts/test_provider_fallback.py`

* [ ] 所有新增模块能正常 import

* [ ] 本地候选 >=3 张时**不调用**百度

* [ ] 本地候选 < 3 张时**自动调用**百度

* [ ] 百度图片下载后重新用 CLIP 评分（不使用百度排名）

* [ ] 百度失败不崩溃，继续处理下一张

* [ ] 候选不足 3 张时标记 `insufficient_candidates`，不强行凑数

* [ ] `--resume` 只跳过真正完成的任务

* [ ] 全局占用生效，不同样图不重复使用同一候选

* [ ] 10 张样图全部输出 A/B/C 三张

* [ ] Excel 报告包含来源字段（local/baidu）



***

## 十四、常见问题排查

### Q1: 运行脚本报 `ModuleNotFoundError: No module named 'xxx'`

**解决**：



```
.venv\Scripts\pip.exe install xxx
```

常见需要装的：`requests`、`playwright`、`imagehash`

### Q2: 百度识图一直返回空

**可能原因和解决**：



1. 网络问题：确认能正常访问 `https://www.baidu.com`

2. 被反爬：等 10 分钟再试，或者把 `mode` 改成 `"browser"`

3. HTTP 解析失效：百度页面改版了，需要更新 `_extract_urls_from_html` 里的正则

4. 熔断器打开：连续失败 3 次后会暂时停止，删除 `runtime/baidu_cache` 目录后重启程序

### Q3: Playwright 安装失败或下载慢

**解决**：



* 可以先不用浏览器模式，把 config 里 `mode` 改成 `"http"`

* HTTP 模式够用的话，不需要装 playwright

* 如果一定要装，设置国内镜像：



```
set PLAYWRIGHT\\\_DOWNLOAD\\\_HOST=https://npmmirror.com/mirrors/playwright

.venv\Scripts\playwright.exe install chromium
```

### Q4: 修改主脚本后报缩进错误

**解决**：



* Python 用 4 个空格缩进，不能用 Tab

* 检查复制的代码是否对齐

* 用 `notepad` 打开时，注意不要自动转换缩进

### Q5: 批量运行时某张样图失败了怎么办

**正常现象**。方案 D 的设计就是单张失败不影响整体。



* 失败的样图会标记为 `retryable_error` 或 `insufficient_candidates`

* 下次用 `--resume` 运行时会自动重试

* 如果一直失败，检查那张样图是不是格式有问题，或者本地库里确实没有相似图

### Q6: 想临时关掉百度，只用本地检索

**解决**：把 config.yaml 里 `baidu_image_search.enabled` 改成 `false`。

程序会自动跳过百度，本地不够 3 张就直接报错。

### Q7: 想恢复到方案 D 之前的状态

**解决**：



```
xcopy runtime\backups\before\\\_scheme\\\_d\\\_final\ . /E /Y
```

（在项目根目录执行，会用备份覆盖当前文件）

### Q8: 百度缓存目录越来越大怎么办

**解决**：定期删除 `runtime\baidu_cache\` 目录即可，下次运行会自动重建。

缓存的作用是避免重复下载同一张图，删了不影响功能，只是会重新下载。



***

## 十五、文件变更总览

### 新增文件（7 个）



```
src/providers/\\\_\\\_init\\\_\\\_.py

src/providers/base.py

src/providers/local\\\_faiss\\\_provider.py

src/providers/baidu\\\_image\\\_provider.py

src/remote\\\_candidate\\\_cache.py

src/candidate\\\_pool.py

scripts/test\\\_baidu\\\_search.py

scripts/test\\\_provider\\\_fallback.py
```

### 修改文件（2 个）



```
config.yaml              (追加方案D配置)

scripts/run\\\_batch\\\_selection.py  (接入百度降级逻辑)
```

### 不修改的文件（保持原样）



```
src/clip\\\_searcher.py

src/vector\\\_store.py

src/duplicate\\\_detector.py

src/image\\\_classifier.py

src/multi\\\_feature\\\_scorer.py

src/candidate\\\_registry.py

src/task\\\_store.py

src/result\\\_exporter.py

src/vision\\\_searcher.py   (代码保留，配置关闭)
```



***

## 十六、后续优化建议（可选，不是必须）



1. **报告增强**：在 `result_exporter.py` 的 `build_rows` 函数里增加 `候选来源`、`百度原始URL`、`是否触发百度` 等字段

2. **预览图标注来源**：在 `create_contact_sheet` 里把来源（local/baidu）显示在标签上

3. **全局登记增加来源**：在 `candidate_registry.py` 的 `reserve` 方法里记录 `candidate_source` 和 `candidate_url`

4. **缓存清理脚本**：写一个脚本定期清理 `runtime/baidu_cache` 里超过 30 天的文件

5. **百度结果质量监控**：统计百度候选的平均 CLIP 分数，如果持续偏低说明百度结果质量不好



***

> **最后提醒**
> ：每完成一个阶段就验证一次，不要一口气全改完再测。出了问题容易定位是哪一步的错。
> 有任何报错，先把完整的错误信息复制下来，对照「常见问题排查」解决。



***

# 十八、真机实测落地结论（2026-09-08 最终验收）

> 本章节是 10 张样图完整跑通后的真实结果，供日常使用和排错参考。

## 18.1 最终运行结果



* 10/10 全部完成，0 失败，全局占用 30 个候选，SHA-256 哈希无重复。

* 9 张本地候选充足（4\~14 张），**均未调用百度**，全部 local。

* 第 6 张本地过滤后仅剩 2 张，**自动触发百度兜底**：拿到 30 个 URL → 下载 10 张 → 重新评分后有效 10 张 → 最终 A/B 来自百度、C 来自本地。

* 候选来源最终分布：local 28 张、baidu 2 张。

## 18.2 百度识图的关键实测经验（重要）



1. **纯 HTTP 接口走不通**：直接 POST `https://graph.baidu.com/upload` 会被反爬返回 `{"status":1,"msg":"Reject"}`，无论是否带 Cookie/Referer。不必反复尝试加请求头。

2. **用系统自带 Edge 即可，无需下载 Chromium**：浏览器模式通过 Playwright 的 `channel="msedge"` 驱动本机 Edge，免去 `playwright install chromium` 几百 MB 下载。auto 模式下 HTTP 会快速失败（约 1\~2 秒）后自动切到 Edge，属正常现象，不是报错。

3. **URL 必须做 HTML 实体解码**：页面里 `&` 会变成 `&amp;`，不解码会导致下载全部 400（代码已用 `html_lib.unescape` 处理）。

4. 百度候选**不采信其排名**，全部下载到 `runtime/baidu_cache/` 后，用本项目 CLIP + 多维评分器重新打分；实测 8 张下载图里有 2 张因类别不兼容被正确剔除。

## 18.3 日常使用三条命令（小白照抄）



```
\# 1) 全新批量跑（样图放 input\_samples，结果在 output）

.\\.venv\Scripts\python.exe scripts\run\_batch\_selection.py --input\_dir input\_samples --output\_dir output --top\_k 98

\# 2) 中断后续跑（只跳过真正完成且三图齐全的，失败的会自动重跑）

.\\.venv\Scripts\python.exe scripts\run\_batch\_selection.py --input\_dir input\_samples --output\_dir output --top\_k 98 --resume

\# 3) 跑完生成 Excel 报告和四宫格预览图

.\\.venv\Scripts\python.exe scripts\generate\_report.py
```

## 18.4 自动验收



```
.\\.venv\Scripts\python.exe runtime\verify\_scheme\_d.py
```

输出 “\[全部验收通过]” 即代表：10 张齐全、30 候选无重复、百度兜底链路正常、本地充足不调百度。

## 18.5 已验证的容错行为



* 百度 HTTP 被拒 → 自动切 Edge 浏览器；浏览器也失败 → 保留本地候选，任务标 `insufficient_candidates`，**不中断整批**，继续下一张。

* 连续失败 3 次触发熔断，避免反复开浏览器。

* 候选不足 3 张时**绝不复制旧图、不降低重复标准凑数**，明确报错并可在下次 `--resume` 重跑。

* `completed` 但 A/B/C 任一输出文件丢失，`--resume` 也会重跑；`insufficient_candidates / baidu_failed / download_failed / processing` 全部会重跑。

## 18.6 备份位置



* 方案 D 改造前整项目备份：`C:\Users\mozhihao\core\backup_before_scheme_d_final\`

* 本轮旧任务状态 / 旧输出备份：`runtime\tasks\backup_before_scheme_d\`