
# 第10章 llama.cpp 转换器维护工具 —— 模型清单的「自动哨兵」

## 学习目标

- 理解 convert_hf_to_gguf_update.py 脚本的设计动机：从 Hugging Face 自动同步预分词器哈希，避免人工维护
- 掌握 SHA-256 哈希作为「预分词器指纹」的工作原理与实现机制
- 熟悉 TOKENIZER_TYPE 枚举（SPM/BPE/WPM/UGM）四大分词器家族的差异
- 理解「增量同步」模式：如何通过正则解析 convert_hf_to_gguf.py 已存在的模型清单
- 掌握用正则表达式进行「外科手术式」源码原地替换（re.DOTALL + re.MULTILINE）
- 了解词表测试用例的设计思路与边界场景（空白、Emoji、连写、CJK、否定用例等）
- 熟悉 HF 鉴权机制：Bearer Token、环境变量 ~/.cache/huggingface/token 与命令行三种获取方式
- 能根据新增模型扩展脚本，将自定义分词器识别码纳入 llama.cpp 转换流水线

## 10.1 生活类比

想象 `convert_hf_to_gguf_update.py` 是一个**尽职尽责的仓库管理员**：**模型清单** = 仓库里的货架编号与商品目录；**Hugging Face Token** = 进入仓库的钥匙；`download_file_with_auth` = 快递员取件流程（凭证检查、签收、入库）；**CHK_TXT** = 一套精心设计的「口味测试套餐」（甜、咸、辣、异域风味全覆盖）；**SHA-256 哈希** = 每种预分词器的「唯一 DNA 鉴定报告」；`get_existing_models` = 仓库盘点员，扫描现有库存避免重复进货；`re.sub + Marker` = 仓库改造工人，按图纸（标记）精准替换某一区段的货架标签；**tests 列表** = 质检部门的抽样检测表，确保每批货都能经受住刁钻场景考验。就像仓库管理员需要保证 SKU 编号与商品一一匹配，`convert_hf_to_gguf_update.py` 也需要保证每个模型的预分词器哈希与 `convert_hf_to_gguf.py` 中的分支表严格对应。

## 10.2 源码地图

```text
./convert_hf_to_gguf_update.py
├── 导入与日志配置（1-9行）
│   ├── logging 模块               # DEBUG 级别详细诊断
│   ├── requests.Session()         # 全局 HTTP 会话（sess）
│   └── pathlib.Path               # 跨平台路径处理
├── HF Token 与目标文件读取（11-13行）
│   ├── hf_token_pth               # ~/.cache/huggingface/token
│   ├── convert_py_pth             # ./convert_hf_to_gguf.py
│   └── convert_py                 # 目标源码字符串
├── TOKENIZER_TYPE 枚举（16-20行）
│   ├── SPM                        # SentencePiece
│   ├── BPE                        # Byte-Pair Encoding
│   ├── WPM                        # WordPiece
│   └── UGM                        # Unigram
├── DOC_STRING 帮助文档（22-44行）
│   └── 脚本用途说明 + ref: PR 6920
├── argparse 参数定义（46-65行）
│   ├── --full                     # 下载全部模型
│   ├── --check-missing            # 仅检测缺失哈希
│   └── hf_token                   # 可选命令行 Token
├── CHK_TXT 探测文本（73-78行）
│   └── 覆盖空白/Unicode/Emoji/CJK/数字/标点等边界
├── models 主清单（80-152行）
│   ├── 每个条目: {name, tokt, repo}
│   └── 包含 llama-spm/llama-bpe/phi-3/qwen2/jina-v*/gemma/t5/... 等 60+ 模型
├── pre_computed_hashes 异常清单（154-180行）
│   ├── 上游损坏或需手工指定的哈希
│   ├── chatglm-bpe / glm4 / minerva-7b / hunyuan / falcon-h1 系列
│   └── kimi-k2 / qwen2 (Qwen3-Embedding) / grok-2 / jina-v2-de 变体
├── download_file_with_auth() 函数（196-202行）
│   └── Bearer Token 单文件下载器
├── download_model() 函数（205-244行）
│   ├── SPM → tokenizer.model
│   ├── UGM → spiece.model
│   └── 本地路径模式（shutil.copy2）与远程 URL 模式
├── get_existing_models() 函数（247-256行）
│   └── 正则匹配 if chkhsh == "...": res = "..." 模式
├── 主流程：增量过滤（258-275行）
│   └── 增量过滤（not args.full）   # 仅处理未注册模型
├── 主流程：哈希计算与诊断输出（278-318行）
│   ├── SPM/UGM 跳过哈希计算        # 有显式枚举值
│   ├── 哈希计算：tokenizer.encode(CHK_TXT) → SHA-256
│   └── 诊断输出：normalizer / pre_tokenizer / ignore_merges
├── 主流程：源码生成与替换（321-394行）
│   ├── src_ifs 函数体生成（含 chkhsh → name 映射链）
│   └── re.sub(Marker Start/End, lambda 替换, DOTALL|MULTILINE)
├── 落盘更新（395行）
│   └── convert_py_pth.write_text()
└── 词表测试用例生成（399-466行）
    ├── tests 测试用例列表（399-438行）
    │   ├── 空白/Unicode/Emoji/CJK/数字边界
    │   └── 否定用例 Cửa Việt（llama-bpe 已知名失败）
    ├── 写入 .gguf.inp 输入文件（__ggml_vocab_test__ 分隔）
    ├── 写入 .gguf.out 输出文件（空格分隔 token ID）
    └── 生成 convert_hf_to_gguf.py 调用命令（打印到日志）
```

## 10.3 脚本入口与配置加载 —— 转换器家族的「自动哨兵」

脚本的核心定位是**自动识别 Hugging Face 仓库中 BPE 预分词器的类型**，并据此生成 `convert_hf_to_gguf.py` 中 `get_vocab_base_pre` 函数的分支逻辑。这避免了人工维护庞大的模型分支表。启动时，脚本会读取 `convert_hf_to_gguf.py` 的完整源码以提取已存在模型列表；从 `~/.cache/huggingface/token` 或命令行参数读取 HF 鉴权 Token；通过 `argparse` 暴露 `--full`（下载全部）与 `--check-missing`（仅检测缺失哈希）两种运行模式；利用 `logging` 模块以 DEBUG 级别输出详细诊断信息；全局 `requests.Session()` 复用 HTTP 连接，提升下载效率。

### 10.3.1 导入与全局状态初始化

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 1-13 行）

```python
# 第 1-9 行：核心依赖导入
import argparse                 # 命令行参数解析
import json                     # JSON 序列化（诊断输出用）
import logging                  # 日志系统
import os                       # 文件系统操作
import re                       # 正则表达式（源码替换核心）
import shutil                   # 高级文件拷贝
from enum import IntEnum, auto  # 枚举定义
from hashlib import sha256      # SHA-256 哈希算法
from pathlib import Path        # 跨平台路径对象

import requests                 # HTTP 客户端
from transformers import AutoTokenizer  # HF 分词器加载器

# 第 9 行：全局日志配置，DEBUG 级别确保能看到详细诊断信息
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 第 11-13 行：HF Token 与目标文件路径初始化
hf_token_pth = Path("~/.cache/huggingface/token").expanduser()  # 默认 Token 存放位置
convert_py_pth = Path("convert_hf_to_gguf.py")                  # 目标转换脚本
convert_py = convert_py_pth.read_text(encoding="utf-8")         # 完整源码字符串
```

这段代码完成了脚本运行所需的**基础设施搭建**：导入依赖、配置日志、读取鉴权凭证、加载目标文件内容到内存。`convert_py` 字符串将在后续作为正则替换的操作对象。

### 10.3.2 命令行参数解析

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 46-65 行）

```python
# 第 46-65 行：argparse 参数定义
parser = argparse.ArgumentParser(
    description=DOC_STRING,  # 22-44 行定义的帮助文档
    formatter_class=argparse.RawTextHelpFormatter  # 保留换行格式
)

# 可选命令行 Token（优先级最高）
parser.add_argument(
    "hf_token", nargs="?", default=None,
    help="Hugging Face token (optional, can use HF_TOKEN env var or ~/.cache/huggingface/token)"
)

# --full：强制处理所有模型，忽略增量过滤
parser.add_argument(
    "--full", action="store_true",
    help="Process all models (default: only process models not in convert_hf_to_gguf.py)"
)

# --check-missing：仅检测缺失哈希，不下载、不更新
parser.add_argument(
    "--check-missing", action="store_true",
    help="Only check for missing hashes, don't download or update"
)

args = parser.parse_args()

# 第 67-72 行：Token 获取优先级链（命令行 > 环境变量 > 缓存文件）
hf_token = args.hf_token or os.getenv("HF_TOKEN")
if hf_token is None and hf_token_pth.exists():
    hf_token = hf_token_pth.read_text(encoding="utf-8").strip()
```

这段代码实现了**三级 Token 获取策略**：命令行参数 → 环境变量 `HF_TOKEN` → `~/.cache/huggingface/token` 文件。`--full` 与 `--check-missing` 互为互斥的运行模式开关，分别对应「全量重建」与「只读巡检」两种维护场景。

## 10.4 模型清单与「预分词器识别码」机制 —— 模型身份的「指纹字典」

llama.cpp 需要在 GGUF 文件中写入 `tokenizer.ggml.pre` 字段，以指示运行时使用哪种预分词器。为此，脚本维护了一份「模型名 → 仓库 URL → 期望分词器类型」的三元组清单 `models`。`TOKENIZER_TYPE` 枚举区分四大分词器家族：SPM（SentencePiece）、BPE（Byte-Pair Encoding）、WPM（WordPiece）、UGM（Unigram）。`pre_computed_hashes` 用于记录已知上游仓库被破坏、无法自动下载的特殊模型（如 glm4、falcon-h1 系列等）。通过在精心设计的测试字符串 `CHK_TXT` 上运行分词器并计算 SHA-256 来生成「预分词器指纹」。`models` 列表涵盖了 llama-spm/llama-bpe/phi-3/qwen2/jina-v*/gemma/t5 等 60+ 模型。

### 10.4.1 TOKENIZER_TYPE 枚举定义

源码路径：`./convert_hf_to_gguf_update.py` - `TOKENIZER_TYPE`（第 16-22 行）

```python
# 第 16-22 行：分词器类型枚举
class TOKENIZER_TYPE(IntEnum):
    SPM = auto()  # SentencePiece (如 LLaMA、T5)
    BPE = auto()  # Byte-Pair Encoding (如 GPT-2、Qwen)
    WPM = auto()  # WordPiece (如 BERT)
    UGM = auto()  # Unigram (如 mBART、XLNet)
```

`IntEnum` 让枚举成员可直接参与整数比较，`auto()` 自动分配 1、2、3、4。这是整个识别逻辑的**类型坐标系**：后续下载策略、哈希计算跳过逻辑、GGUF 写入枚举值均依赖此分类。

### 10.4.2 CHK_TXT 探测文本设计

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 73-78 行）

```python
# 第 73-78 行：精心设计的「口味测试套餐」
CHK_TXT = (
    " \t\n\r"                                    # 空白字符：空格、制表、换行、回车
    "¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿"              # Latin-1 Supplement 符号
    "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"              # 西里尔字母
    "αβγδεζηθικλμνξοπρςστυφχψω"                    # 希腊字母
    "你好世界"                                     # 中文（CJK）
    "こんにちは世界"                                # 日文（平假名/片假名）
    "안녕하세요"                                    # 韩文（谚文）
    "🌟🚀🎉💯🔥"                                     # Emoji（多码位、ZWJ 序列）
    "1234567890"                                   # 连续数字
    ".,!?;:()[]{}\"'<>"                            # 标点符号
)
```

`CHK_TXT` 是**哈希计算的输入基准**。它覆盖了分词器最容易产生差异的边界场景：空白处理策略、Unicode 规范化、Emoji 拆分、CJK 字符切分、数字连写、标点附着。同一个分词器在不同版本下若预分词规则变更，对此串的编码结果必然不同，从而导致哈希值变化——这正是「指纹」的敏感性所在。

### 10.4.3 models 主清单与 pre_computed_hashes 异常清单

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 80-180 行）

```python
# 第 80-152 行：models 主清单（节选，共 60+ 条目）
models = [
    {"name": "llama-spm",      "tokt": TOKENIZER_TYPE.SPM, "repo": "hf-internal-testing/llama-tokenizer"},
    {"name": "llama-bpe",      "tokt": TOKENIZER_TYPE.BPE, "repo": "hf-internal-testing/llama-bpe-tokenizer"},
    {"name": "phi-3",          "tokt": TOKENIZER_TYPE.BPE, "repo": "microsoft/Phi-3-mini-4k-instruct"},
    {"name": "qwen2",          "tokt": TOKENIZER_TYPE.BPE, "repo": "Qwen/Qwen2-7B-Instruct"},
    {"name": "jina-v2-base-en","tokt": TOKENIZER_TYPE.BPE, "repo": "jinaai/jina-embeddings-v2-base-en"},
    {"name": "gemma",          "tokt": TOKENIZER_TYPE.BPE, "repo": "google/gemma-7b"},
    {"name": "t5",             "tokt": TOKENIZER_TYPE.UGM, "repo": "google-t5/t5-base"},
    # ... 其余 50+ 条目
]

# 第 154-180 行：pre_computed_hashes 异常清单（节选）
pre_computed_hashes = [
    {"name": "chatglm-bpe",    "tokt": TOKENIZER_TYPE.BPE, "repo": "THUDM/chatglm3-6b",
     "chkhsh": "a1b2c3..."},  # 64 位十六进制
    {"name": "glm4",           "tokt": TOKENIZER_TYPE.BPE, "repo": "ZhipuAI/glm-4-9b-chat",
     "chkhsh": "d4e5f6..."},
    {"name": "falcon-h1",      "tokt": TOKENIZER_TYPE.BPE, "repo": "tiiuae/Falcon-H1-7B",
     "chkhsh": "h8i9j0..."},
    # ... 其余异常条目
]
```

以下是主清单与异常清单的对比说明：

| 维度 | models 主清单 | pre_computed_hashes 异常清单 |
|------|---------------|------------------------------|
| **用途** | 常规模型，脚本自动下载并计算哈希 | 上游仓库损坏/需鉴权/文件缺失，预置已知哈希 |
| **字段** | name, tokt, repo | name, tokt, repo, **chkhsh** |
| **典型场景** | llama-spm、qwen2、gemma 等标准模型 | chatglm-bpe、glm4、falcon-h1 等问题模型 |
| **处理逻辑** | 下载 → 加载 → 编码 CHK_TXT → 计算 SHA-256 | 直接使用预置 chkhsh，跳过下载与计算 |

## 10.5 下载器实现 —— 模型资产的「快递员」

脚本需要实际的分词器文件（`tokenizer.json` / `tokenizer.model` 等）才能本地计算哈希。`download_file_with_auth` 通过 `requests.Session` 携带 Bearer Token 下载单个文件；`download_model` 根据分词器类型选择性拉取所需文件清单；支持本地路径模式（直接 `shutil.copy2`）与远程 URL 模式（HTTP GET）；智能跳过已存在的目标文件，避免重复下载；SPM 类型额外下载 `tokenizer.model`，UGM 类型额外下载 `spiece.model`。

### 10.5.1 单文件下载器

源码路径：`./convert_hf_to_gguf_update.py` - `download_file_with_auth`（第 196-213 行）

```python
# 第 196-213 行：Bearer Token 单文件下载器
def download_file_with_auth(url, token, save_path):
    # ① 构造鉴权头（无 Token 则 headers=None，公开仓库可匿名访问）
    headers = {"Authorization": f"Bearer {token}"} if token else None
    # ② 发起 GET 请求（复用全局 sess 连接池）
    response = sess.get(url, headers=headers)
    # ③ 非 2xx 抛异常
    response.raise_for_status()
    # ④ 确保父目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # ⑤ 二进制写入
    with open(save_path, 'wb') as downloaded_file:
        downloaded_file.write(response.content)
    logger.info(f"File {save_path} downloaded successfully")
```

这段代码是**下载原语**：`requests.Session` 复用 TCP 连接与 TLS 会话，显著提升批量下载吞吐；`raise_for_status()` 让 HTTP 错误（401/404/500）立即抛出异常，由上层捕获记录。

### 10.5.2 模型级下载编排

源码路径：`./convert_hf_to_gguf_update.py` - `download_model`（第 215-236 行）

```python
# 第 215-236 行：模型级下载编排
def download_model(model):
    name = model["name"]
    repo = model["repo"]
    tokt = model["tokt"]

    # ① 目标目录
    os.makedirs(f"models/tokenizers/{name}", exist_ok=True)

    # ② 基础文件清单
    files = ["config.json", "tokenizer.json", "tokenizer_config.json"]

    # ③ 特殊模型：gpt-4o 无 config.json
    if name == "gpt-4o":
        files = ["tokenizer.json", "tokenizer_config.json"]

    # ④ 分词器类型特有文件
    if tokt == TOKENIZER_TYPE.SPM:
        files.append("tokenizer.model")      # SentencePiece 模型文件
    if tokt == TOKENIZER_TYPE.UGM:
        files.append("spiece.model")         # Unigram 模型文件

    # ⑤ 本地路径模式 vs 远程 URL 模式
    if os.path.isdir(repo):
        # 本地目录：逐文件 copy2（保留元数据）
        for file in files:
            src_path = os.path.join(repo, file)
            dst_path = f"models/tokenizers/{name}/{file}"
            if os.path.isfile(dst_path):
                logger.info(f"{name}: File {dst_path} already exists - skipping")
                continue
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"{name}: Copied {src_path} to {dst_path}")
            else:
                logger.warning(f"{name}: Source file {src_path} does not exist")
    else:
        # 远程 URL：逐文件下载
        for file in files:
            save_path = f"models/tokenizers/{name}/{file}"
            if os.path.isfile(save_path):
                logger.info(f"{name}: File {save_path} already exists - skipping")
                continue
            download_file_with_auth(f"{repo}/resolve/main/{file}", hf_token, save_path)
```

这段代码实现了**按需下载策略**：基础三件套（config、tokenizer.json、tokenizer_config.json）全类型通用；SPM/UGM 追加二进制模型文件；本地路径走 `shutil.copy2`（保留修改时间等元数据），远程走 Bearer Token 认证下载；文件级幂等检查避免重复网络开销。

## 10.6 现有模型识别 —— 「只处理新增」的增量思维

转换脚本中的 `if chkhsh == "...": res = "..."` 分支是「模型清单的事实来源」。`get_existing_models` 通过正则匹配哈希 → 名称的映射，避免重复处理已知模型。默认情况下（非 `--full` 模式）只下载并处理 `convert_hf_to_gguf.py` 中尚未注册的条目。当 `args.check_missing` 为真时，仅检查是否有缺失的预分词器哈希而不执行下载。这是典型的「增量同步」模式，保持主转换脚本与上游 Hugging Face 仓库同步。

### 10.6.1 正则解析已注册模型

源码路径：`./convert_hf_to_gguf_update.py` - `get_existing_models`（第 247-256 行）

```python
# 第 247-256 行：正则提取已注册模型
def get_existing_models(convert_py):
    # 匹配模式：if chkhsh == "64位十六进制":\n        res = "模型名"
    pattern = r'if chkhsh == "([a-f0-9]{64})":\s*\n\s*.*\s*res = "([^"]+)"'
    matches = re.findall(pattern, convert_py)
    output = {}
    for chkhsh, res in matches:
        output[res] = chkhsh  # {模型名: 哈希值}
    return output
```

正则 `r'if chkhsh == "([a-f0-9]{64})":\s*\n\s*.*\s*res = "([^"]+)"'` 精准捕获**生成代码的固定模板结构**：哈希值固定 64 位十六进制，`res` 赋值在紧随其后的缩进块中。`re.findall` 返回所有匹配的元组列表，构建 `{name: chkhsh}` 字典供后续去重使用。

### 10.6.2 增量过滤逻辑

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 258-275 行）

```python
# 第 258-275 行：增量过滤主逻辑
existing_models = {}
all_models = models.copy()

if not args.full:
    # ① 非 --full 模式：提取已注册模型
    existing_models = get_existing_models(convert_py)
    all_models = models.copy()
    # ② 仅保留未注册的模型
    models = [model for model in all_models if model["name"] not in existing_models]

if not args.check_missing:
    # ③ 非 --check-missing：执行下载
    logging.info(f"Downloading {len(models)} models...")
    for model in models:
        try:
            download_model(model)
        except Exception as e:
            logger.error(f"Failed to download model {model['name']}. Error: {e}")
```

这是**增量同步的核心控制流**：
1.  `--full` 关闭 → 读取现有清单 → 过滤掉已注册模型 → `models` 仅含新增项
2.  `--check-missing` 关闭 → 对 `models` 逐个下载，异常隔离不中断整体流程
3.  两者结合：`--full --check-missing` 可用于「全量巡检，不修改」场景

## 10.7 哈希计算与诊断输出 —— 「唯一指纹」的诞生

核心算法：用一段精心设计的「探测文本」喂给分词器，将产出的 token 列表序列化后取 SHA-256。SPM/UGM 类型分词器跳过哈希计算（这些有显式的 `tokenizer.ggml.pre` 枚举值）。`tokenizer.encode(CHK_TXT)` 生成 token ID 序列。`sha256(str(chktok).encode()).hexdigest()` 计算 64 位十六进制摘要。同步打印 `normalizer` / `pre_tokenizer` / `ignore_merges` 三个 JSON 字段，便于贡献者人工核对。`t5` 模型使用 `use_fast=False` 的特殊加载路径。

### 10.7.1 哈希计算主循环

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 278-318 行）

```python
# 第 278-318 行：哈希计算与诊断输出（节选关键片段）
src_ifs = ""
for model in [*pre_computed_hashes, *all_models]:
    name = model["name"]
    tokt = model["tokt"]
    chkhsh = model.get("chkhsh")

    # ① SPM/UGM 直接跳过（有显式 GGML 枚举值，无需哈希识别）
    if tokt == TOKENIZER_TYPE.SPM or tokt == TOKENIZER_TYPE.UGM:
        continue

    # ② 三级哈希来源判定
    if chkhsh is not None:
        # 预置哈希（pre_computed_hashes）
        logger.info(f"Using pre-computed hash for model {name}: {chkhsh}")
    elif name in existing_models:
        # 已注册模型：复用旧哈希
        chkhsh = existing_models[name]
    else:
        # 新增模型：实时计算
        # 校验 tokenizer_config.json 存在性
        if not os.path.isfile(f"models/tokenizers/{name}/tokenizer_config.json"):
            raise OSError(f"Config for tokenizer {name} not found...")

        # 加载分词器（t5 特殊处理）
        try:
            logger.info(f"Loading tokenizer from {f'models/tokenizers/{name}'}...")
            if name == "t5":
                tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}", use_fast=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}")
        except Exception as e:
            raise OSError(f"Error loading tokenizer for model {name}.") from e

        # 核心：编码 CHK_TXT → 列表 → 字符串 → SHA-256
        chktok = tokenizer.encode(CHK_TXT)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        # ③ 详细诊断日志
        logger.info(f"model: {name}")
        logger.info(f"tokt: {tokt}")
        logger.info(f"repo: {model['repo']}")
        logger.info(f"chktok: {chktok}")
        logger.info(f"chkhsh: {chkhsh}")

        # ④ 解析 tokenizer.json 打印关键配置
        with open(f"models/tokenizers/{name}/tokenizer.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            normalizer = cfg["normalizer"]
            logger.info("normalizer: " + json.dumps(normalizer, indent=4))
            pre_tokenizer = cfg["pre_tokenizer"]
            logger.info("pre_tokenizer: " + json.dumps(pre_tokenizer, indent=4))
            if "ignore_merges" in cfg["model"]:
                logger.info("ignore_merges: " + json.dumps(cfg["model"]["ignore_merges"], indent=4))

        logger.info("")
```

**为什么用 `str(chktok)` 而不是 `bytes(chktok)`？** `tokenizer.encode` 返回 `List[int]`，`str([1, 2, 3])` 产生 `'[1, 2, 3]'` 这种**可读、跨平台稳定**的字符串表示；若用 `bytes(chktok)` 会因整数溢出（>255 抛异常）或字节序歧义导致不可移植。

**三段诊断输出的含义：**
- `normalizer`：文本规范化管线（NFD、Lowercase、Strip 等），影响字符层面预处理
- `pre_tokenizer`：预分词器核心配置（ByteLevel、BertPreTokenizer、Metaspace 等），直接决定分词边界
- `ignore_merges`：BPE 合并规则中的例外列表，部分模型（如 GPT-2）用于保护特殊 token 不被合并

## 10.8 代码生成与原地替换 —— 「外科手术式」的源码更新

这是脚本的终极目标：把识别出的哈希直接写回 `convert_hf_to_gguf.py` 的 `get_vocab_base_pre` 函数。通过 `# Marker: Start get_vocab_base_pre` / `# Marker: End get_vocab_base_pre` 标记定位插入点。用 `re.sub` + `re.DOTALL | re.MULTILINE` 进行多行匹配替换。生成的新代码块包含完整的 if-elif 链，附带 `# ref: <repo URL>` 注释便于溯源。使用 `lambda m: m.group(1) + src_func + m.group(3)` 保留头部与尾部结构。最终调用 `convert_py_pth.write_text()` 落盘更新。

### 10.8.1 源码生成：构建 if-elif 链

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 321-394 行，节选生成逻辑）

```python
# 第 321-394 行：源码生成（节选）
src_ifs = ""
for model in [*pre_computed_hashes, *all_models]:
    name = model["name"]
    tokt = model["tokt"]
    chkhsh = model.get("chkhsh")

    if tokt == TOKENIZER_TYPE.SPM or tokt == TOKENIZER_TYPE.UGM:
        continue

    # ... 哈希获取逻辑（同上）...

    # ① 累加 if-elif 分支
    src_ifs += f"        if chkhsh == \"{chkhsh}\":\n"
    src_ifs += f"            # ref: {model['repo']}\n"
    src_ifs += f"            res = \"{name}\"\n"

# ② 模板字符串构建完整函数体
src_func = f"""
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = {repr(CHK_TXT)}

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {{chktok}}")
        logger.debug(f"chkhsh: {{chkhsh}}")

        res = None

        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
{src_ifs}
        if res is None:
            logger.warning("\\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert_hf_to_gguf_update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggml-org/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {{chkhsh}}")
            logger.warning("**************************************************************************************")
            logger.warning("\\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {{repr(res)}}")
        logger.debug(f"chkhsh: {{chkhsh}}")

        return res
"""
```

`src_ifs` 累加生成**顺序 if-elif 链**，顺序即优先级。每个分支包含 `# ref:` 注释，指向 Hugging Face 仓库，便于溯源审计。`repr(CHK_TXT)` 确保测试文本中的转义字符（`\t\n\r`）在生成的源码中保持字面量正确。

### 10.8.2 正则原地替换

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 392-395 行）

```python
# 第 392-395 行：re.sub 原地替换
convert_py = re.sub(
    r"(# Marker: Start get_vocab_base_pre)(.+?)( +# Marker: End get_vocab_base_pre)",
    lambda m: m.group(1) + src_func + m.group(3),
    convert_py,
    flags=re.DOTALL | re.MULTILINE,
)

# 第 395 行：落盘
convert_py_pth.write_text(convert_py, encoding="utf-8")
```

**正则分组解析：**
- `m.group(1)`：`# Marker: Start get_vocab_base_pre`（保留）
- `m.group(2)`：旧函数体（被丢弃）
- `m.group(3)`：`# Marker: End get_vocab_base_pre` 及其前缀缩进（保留）

**标志位作用：**
- `re.DOTALL`：让 `.` 匹配换行符，跨越多行函数体
- `re.MULTILINE`：让 `^` `$` 匹配行首行尾（此处虽未显式用到，但作为防御性编程保留）

**为什么用 lambda 而不是字符串替换？** lambda 能访问捕获组（`m.group(1)`、`m.group(3)`），精准保留 Marker 注释与缩进结构；字符串替换会丢失这两段「锚点」，导致下次运行无法定位。

**如果缺失 Marker 注释会发生什么？** `re.sub` 匹配失败，`convert_py` 保持原样，脚本静默结束且不报错——这是一种**故障安全**设计：宁可不更新，也不破坏源码结构。

## 10.9 词表测试用例生成 —— 「质量门卫」的诞生

为什么需要测试用例？让 llama.cpp 的 C++ 端分词器与 HuggingFace Python 分词器逐条对比，发现实现偏差。`tests` 列表覆盖空白、Unicode、Emoji、CJK、连续数字、标点等棘手边界。包括一条「llama-bpe 已知名失败」的 `Cửa Việt` 案例作为负面回归测试。输入文件以 `__ggml_vocab_test__` 分隔每条测试用例。输出文件以空格分隔 token ID，与 llama.cpp 的 `test-tokenizer-*` 程序约定一致。自动打印生成 vocab 文件的命令行调用。

### 10.9.1 tests 测试用例列表

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 399-438 行）

```python
# 第 399-438 行：tests 列表（节选）
tests = [
    " ",                    # 单空格
    "  ",                   # 双空格
    "   ",                  # 三空格
    "\t",                   # 制表符
    "\n",                   # 换行符
    "\r",                   # 回车符
    "a",                    # 单字母
    "á",                    # 带重音字母
    "🌟",                    # 单 Emoji
    "🚀🎉",                  # 多 Emoji 连写
    "你好",                  # 中文双字
    "世界",                  # 中文双字
    "こんにちは",              # 日文平假名
    "안녕하세요",              # 韩文谚文
    "123",                  # 连续数字
    "123456",               # 长数字
    "Cửa Việt",             # 越南语地名（llama-bpe 已知名失败，负向回归）
    "🇺🇸",                   # 国旗 Emoji（地区指示符序列）
    "👨‍👩‍👧‍👦",               # 家庭 Emoji（ZWJ 序列）
    "🏳️‍🌈",                 # 彩虹旗（ZWJ + 变体选择器）
]
```

**负向回归测试的意义**：`Cửa Việt` 在 llama-bpe 下会产生错误分词（如将 `Cửa` 拆分为 `C` `ủ` `a`），记录该用例可确保后续优化不引入新的退化。`__ggml_vocab_test__` 分隔符是 llama.cpp C++ 测试程序约定的**记录分界符**，解析时按此分隔逐条对比。

### 10.9.2 测试文件写入与命令生成

源码路径：`./convert_hf_to_gguf_update.py` - `*`（第 440-466 行）

```python
# 第 440-466 行：测试文件写入（节选）
for model in models:
    name = model["name"]
    tokt = model["tokt"]

    # 跳过无 tokenizer 目录的模型
    if not os.path.exists(f"models/tokenizers/{name}"):
        logger.warning(f"Directory for tokenizer {name} not found. Skipping...")
        continue

    # 重新加载分词器（复用前面逻辑）
    try:
        if name == "t5":
            tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}", use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}")
    except (OSError, TypeError) as e:
        logger.error(f"Failed to load tokenizer for model {name}. Error: {e}")
        continue

    # 跳过无 GGUF 文件的模型（仅生成测试用例不转换）
    if not os.path.exists(f"models/ggml-vocab-{name}.gguf"):
        logger.info(f"Skip vocab files for model {name}, no GGUF file found")
        continue

    # 写入 .gguf.inp：原文 + 分隔符
    with open(f"models/ggml-vocab-{name}.gguf.inp", "w", encoding="utf-8") as f:
        for text in tests:
            f.write(f"{text}")
            f.write("\n__ggml_vocab_test__\n")

    # 写入 .gguf.out：token ID 空格分隔，每用例一行
    with open(f"models/ggml-vocab-{name}.gguf.out", "w") as f:
        for text in tests:
            res = tokenizer.encode(text, add_special_tokens=False)
            for r in res:
                f.write(f" {r}")
            f.write("\n")

    logger.info(f"Tests for {name} written in ./models/ggml-vocab-{name}.gguf.*")
```

`.gguf.inp` 格式：
```
原文1
__ggml_vocab_test__
原文2
__ggml_vocab_test__
...
```

`.gguf.out` 格式：
```
 token1 token2 token3
 token4 token5
 ...
```

llama.cpp 的 `test-tokenizer-*` 程序会读取 `.inp` 逐段分词，与 `.out` 逐行对比 token ID 序列，**任何不匹配即判定为回归**。这种「黄金样本」机制保证了 Python 参考实现与 C++ 生产实现的一致性。

## 10.10 设计中的取舍

**为什么不用简单的字符串替换，而要用 `re.sub + Marker + lambda`？**

简单字符串替换（如 `str.replace`）无法处理多行、无法保留缩进结构、无法验证替换区域边界。Marker + 正则 + lambda 组合提供了**边界确认**（Start/End 必须成对存在）、**结构保留**（保留 Marker 与缩进）、**原子性替换**（整个函数体一次性换入）三重保障。这是典型的「外科手术式」源码修改模式，相比整体重写，最大程度减少了 Git diff 噪音，便于代码审查。

**为什么 SPM/UGM 不计算哈希？**

SPM（SentencePiece）与 UGM（Unigram）在 GGML 中有**显式的枚举值**（`LLAMA_TOKENIZER_SPM`、`LLAMA_TOKENIZER_UGM`），运行时直接通过枚举分发，无需通过哈希识别。BPE/WPM 族类型繁多且预分词规则差异大（ByteLevel vs Metaspace vs BertPreTokenizer 等），必须用哈希指纹区分。

**`pre_computed_hashes` 存在的意义是什么？**

部分上游仓库（如 glm4、falcon-h1 系列）存在**文件缺失、损坏、鉴权变更**等问题，导致自动下载失败。预置哈希允许维护者在本地手工验证后「钦定」哈希值，保证转换流水线不阻塞。这是对上游不可控因素的**工程容错**。

## 10.11 动手练习

1.  **阅读脚本主流程**  
    阅读 `./convert_hf_to_gguf_update.py` 全文（1-466 行），理解脚本的工作流程：
    1. 参数解析与 HF Token 获取（1-73 行）
    2. CHK_TXT 探测文本的设计意图（73-78 行）
    3. models 清单与 pre_computed_hashes 异常清单（80-180 行）
    4. 下载器实现与主循环（196-318 行）  
    回答问题：
    - 为什么 SPM 和 UGM 类型不需要计算哈希？
    - pre_computed_hashes 中 falcon-h1 为何有 4 条不同记录？
    - 脚本如何避免对已注册模型进行重复下载？

2.  **哈希计算与诊断输出**  
    聚焦 `./convert_hf_to_gguf_update.py` 第 278-318 行的主循环逻辑，理解哈希计算机制：
    1. `tokenizer.encode(CHK_TXT)` 调用细节
    2. `sha256(str(chktok).encode()).hexdigest()` 序列化与哈希过程
    3. normalizer / pre_tokenizer / ignore_merges 三段 JSON 诊断输出  
    回答问题：
    - 为什么用 `str(chktok)` 而不是 `bytes(chktok)` 作为哈希输入？
    - pre_tokenizer 与 normalizer 的区别是什么？
    - ignore_merges 字段何时会出现？

3.  **源码原地替换机制**  
    深入研究 `./convert_hf_to_gguf_update.py` 第 321-394 行的代码生成与替换逻辑：
    1. `src_ifs` 字符串的拼接规则
    2. `# Marker: Start get_vocab_base_pre` 标记的作用
    3. `re.sub` 中 `lambda m: m.group(1) + src_func + m.group(3)` 的妙用
    4. `re.DOTALL | re.MULTILINE` 标志位的含义  
    回答问题：
    - 为什么采用 lambda 而不是简单的字符串替换？
    - 如果 convert_hf_to_gguf.py 缺失 Marker 注释会发生什么？
    - 这种「外科手术式」替换相比整体重写有什么优势？

4.  **词表测试用例设计**  
    分析 `./convert_hf_to_gguf_update.py` 第 399-438 行的 `tests` 列表与第 440-466 行的文件写入逻辑：
    1. 为什么包含 `'Cửa Việt'` 这条测试用例？
    2. `__ggml_vocab_test__` 分隔符的设计目的
    3. `.gguf.inp` 与 `.gguf.out` 文件格式与 llama.cpp test-tokenizer-* 程序的约定  
    回答问题：
    - 如何为新增模型扩展测试用例列表？
    - 负向回归测试的意义是什么？
    - 测试用例应如何覆盖中文/日文/韩文等 CJK 边界？

5.  **动手扩展：添加自定义模型**  
    假设你需要为一个新的 Hugging Face 模型（假设名为 `my-custom-model`，分词器类型为 BPE，仓库 URL 为 `https://huggingface.co/my-org/my-custom-model`）扩展该脚本：
    1. 在 `models` 列表中正确添加条目
    2. 决定是否需要添加到 `pre_computed_hashes`
    3. 运行脚本并观察诊断日志输出  
    回答问题：
    - 如果上游仓库需要鉴权，如何通过命令行传入 Token？
    - 脚本运行后，convert_hf_to_gguf.py 中会生成怎样的 if-elif 分支？
    - 如何为该模型补充 `.gguf.inp` / `.gguf.out` 测试文件？

## 10.12 本章小结

这一章中我们学习了 llama.cpp 转换器家族的维护哨兵——`convert_hf_to_gguf_update.py`。首先，我们了解了脚本的核心定位：自动从 Hugging Face 同步预分词器哈希，生成 `get_vocab_base_pre` 分支表，避免人工维护。其次，剖析了 `TOKENIZER_TYPE` 四大分词器家族枚举与 `CHK_TXT` 探测文本的边界覆盖设计。接着，详细解读了 `models` 主清单与 `pre_computed_hashes` 异常清单的协作机制。然后，深入下载器实现，理解 Bearer Token 认证、本地/远程双模式、按类型选择性拉取文件的策略。之后，揭示了增量同步的核心——正则解析已注册模型、过滤新增模型的去重逻辑。随后，见证了「唯一指纹」的诞生：`tokenizer.encode(CHK_TXT)` → `str(list)` → SHA-256，以及三段诊断日志的溯源价值。紧接着，体验了「外科手术式」源码更新：Marker 定位 + `re.sub(DOTALL|MULTILINE)` + lambda 保留结构的精准替换。最后，理解了词表测试用例生成的黄金样本机制，确保 Python 与 C++ 分词实现的一致性。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| 脚本入口 | argparse + HF Token 三种获取方式（环境变量/文件/命令行） |
| TOKENIZER_TYPE 枚举 | SPM/BPE/WPM/UGM 四大分词器家族区分 |
| CHK_TXT 探测文本 | 覆盖空白/Unicode/Emoji/CJK 等棘手边界的「口味测试套餐」 |
| models 主清单 | 60+ 模型的「模型名 → 仓库 URL → 分词器类型」三元组 |
| pre_computed_hashes | 上游损坏或特殊的预计算哈希（glm4/falcon-h1/kimi-k2 等） |
| download_file_with_auth | Bearer Token 单文件下载器，requests.Session 全局复用 |
| download_model | 根据 tokt 智能拉取文件（SPM→tokenizer.model, UGM→spiece.model） |
| get_existing_models | 正则扫描 convert_hf_to_gguf.py 已注册模型，避免重复处理 |
| 增量同步 | 默认仅处理未注册模型（--full 覆盖），保持与上游同步 |
| SHA-256 哈希 | 用探测文本喂给分词器，将 token ID 序列取哈希作为预分词器指纹 |
| SPM/UGM 跳过 | 这两种有显式 GGML 枚举值，无需计算哈希 |
| 诊断日志输出 | 打印 normalizer/pre_tokenizer/ignore_merges 便于人工核对 |
| Marker 标记替换 | 用 # Marker: Start/End 注释定位 + re.sub(DOTALL\|MULTILINE) 原地替换源码 |
| src_func 代码生成 | 自动生成完整的 get_vocab_base_pre 函数体，含 chkhsh → name 映射链 |
| tests 词表测试用例 | 覆盖空白/Emoji/CJK/数字/标点的边界场景，含否定用例 Cửa Việt |
| .gguf.inp/.gguf.out | 用 __ggml_vocab_test__ 分隔的输入/输出文件，供 llama.cpp test-tokenizer-* 程序对比 |

下一章中，我们将学习 llama.cpp 概览 —— 认识这座"大模型推理引擎"。
