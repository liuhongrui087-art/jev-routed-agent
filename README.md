# LangChain v1 多步骤推理机器人

基于 **LangChain v1** + **Flask** + **Ollama** + **jev**的多步骤推理机器人。

用户提问 → 路由层判断意图 → 走对应的处理链（知识库检索 / 数学计算 / 天气查询 / 闲聊）
→ **逐字流式**返回有依据的中文答案。

**三个主要特点**：

| 特点 | 说明 |
|---|---|
| **Jev 决策路由** | 用专门的**决策模型**做意图判断，把"该用哪个工具"从生成模型里剥离出来。实测路由准确率 **6/6、置信度 1.0**，让这个环节从"概率性"变成"确定性"。详见下文《核心特性：Jev 决策路由》 |
| **流式输出** | SSE 逐字返回，等待期间有实时计时器；检索内容单独成块展示，过程可见 |
| **确定性优先** | 代码能确定完成的（检索）就不交给模型判断；模型只做它擅长的（理解语义、抽取参数） |


---

## 技术栈

| 层 | 技术 |
|---|---|
| Agent | LangChain v1 —— `create_agent`（原生 tool calling） |
| 底层运行时 | LangGraph（`create_agent` 返回的是一张已编译的图） |
| 对话模型 | Ollama `qwen2.5:3b` |
| 嵌入模型 | Ollama `bge-m3`（1024 维，中文友好） |
| 向量库 | ChromaDB，持久化到 `.chroma/` |
| 路由模型 | Jev（TypeSafe System One）—— 可选，不可用时自动降级为本地规则 |
| Web | Flask 3 + 蓝图，SSE 流式 |

---

## 架构

依赖单向：`web → services → core → config.py`，`core/` 内**禁止出现任何 HTTP 相关代码**。

```
浏览器
  │  POST /query/stream   (SSE)
  ▼
web/routes.py               接口层：只管 HTTP（收请求、校验、序列化）
  ▼
services/qa_service.py      编排层：路由分发 + 超时 + 降级
  ▼
services/router.py          路由层：Jev 优先 → 本地规则兜底
  │
  ├─ knowledge  ─▶ 代码直接检索 ─▶ 裸 LLM + RAG 模板      （1 次 LLM，确定性）
  ├─ calculator ─▶ 单工具 Agent（只带 calculator）        （2 次 LLM）
  ├─ weather    ─▶ 单工具 Agent（只带 get_weather）       （2 次 LLM）
  └─ chat       ─▶ 裸 LLM，不带任何工具                    （1 次 LLM）
  ▼
core/                       能力层
  ├── agent_builder.py       按路线组装并缓存 Agent
  ├── tools.py               三个工具：calculator / get_weather / search_knowledge
  ├── rag.py                 知识库索引与检索
  ├── llm.py                 模型工厂
  └── prompts.py             四套专用提示词 + 兜底提示词
  ▼
Ollama（qwen2.5:3b 对话 / bge-m3 嵌入）
```

**为什么要分四条路**：本地 3b 模型在"三个工具里该选哪个"这件事上成功率只有约 1/3。
把路由决策交给专门模型后，每条路**只带一个工具、只配一套提示词** —— 模型不需要"选"，
只需要"填参数"。

---

## 核心特性：Jev 决策路由

> 这是本项目相对一般 Agent / RAG 项目最主要的差异点。

### 1. 为什么要引入它

本地 `qwen2.5:3b` 需要在一个 prompt 里同时完成四件事：

```
判断「该不该用工具」 → 选择「用哪个」 → 构造「参数怎么写」 → 生成「最终答案」
        └──────── 这四步里，前三步都是决策，只有最后一步是生成 ────────┘
```

实测结果是：**同一句「什么是过拟合」，时对时错**，成功率约 **1/3**。
原因不是配置写错，而是**模型的先验分布问题** ——

- 通用对话模型的训练语料里，"问概念 → 直接解释"的样本**海量存在**，
  而"先调工具再回答"的样本占比**极低**
- 系统提示词能提供一层条件化偏置，把倾向往"调工具"推，
  但 3b 的容量不足以让这个推力**稳定压过先验**
- 于是每次生成都在"推力 vs 先验"之间角力，结果**概率性地成功或失败**

更麻烦的是：**这个失败是静默的**。模型不会报错，它会若无其事地凭记忆编一个答案。

### 2. Jev 是什么

**Jev = TypeSafe AI 的 System One 决策模型**（2026 年 9 月发布）。

它的定位与普通 LLM 完全不同：

> **面向软件的概率化决策模型。输入一段 state，输出带概率的类型化答案 —— 不生成自由文本。**

| | 普通 LLM 做决策 | **Jev** |
|---|---|---|
| 输出形态 | 生成一段文本 / 结构化 JSON | **类型化字段 + 概率** |
| 需要解析吗 | 需要，而且还要处理格式跑偏 | **不需要** |
| 有没有置信度 | 没有 | **有 `confidence` 和完整概率分布** |
| 会不会幻觉 | 会 | **不生成文本，没有幻觉的空间** |
| 多个判断 | 塞进一个 prompt，互相污染 | **并行评估，互不干扰** |
| 适用场景 | 通用对话与生成 | **只做判断** |

它有三种问题类型：`Choice`（选择）/ `Noul`（是-否概率）/ `Score`（有序打分）。
本项目只用到 `Choice`。

### 3. 请求与响应

**请求**（`services/router.py` 里的 `_QUESTIONS`）：

```json
{
  "model": "jev-latest",
  "state": "什么是过拟合",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "这句话应该由哪种方式处理？",
      "criteria": {
        "knowledge":  "机器学习的概念、术语、原理、方法相关的问题",
        "calculator": "需要做数学计算的问题",
        "weather":    "查询某个地方的天气",
        "chat":       "闲聊、问候，或与上述都无关的内容"
      }
    }
  }
}
```

> `criteria` 是**你自己定义的候选集合** —— 它只会在里面选，**选不出第五个**。
> 这比"让模型自由发挥再解析"稳得多。

**响应**（实测，模型版本 `jev-1.13.0`）：

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "knowledge",
      "probabilities": {"knowledge": 1.0, "chat": 0.0, "weather": 0.0, "calculator": 0.0},
      "confidence": 1.0
    }
  },
  "usage": {"input_tokens": 395, "output_tokens": 45}
}
```

**整个响应里没有一个字是"生成"的文本** —— 那 45 个 output token 就是标签本身。

| 字段 | 用途 |
|---|---|
| `choice` | 选中的分支，**直接可用** |
| `probabilities` | 每个候选的概率，可用来判断是否在犹豫 |
| `confidence` | 分布的集中程度（1.0 = 完全确定） |

### 4. 集成方式：两级路由

```
用户提问
   │
   ▼
┌──────────────────────────────────────┐
│ services/router.py                   │
│                                      │
│  ① Jev（首选）                        │
│     confidence ≥ 0.6 → 采用           │
│          │                           │
│          └─ 失败/低置信度 ──┐          │
│                            ▼         │
│  ② 本地关键词规则（兜底）              │
│     天气词 → weather                  │
│     计算动词 + 数字 → calculator       │
│     问候语 → chat                     │
│     其余 → knowledge                  │
└──────────────────────────────────────┘
   │
   ▼  intent
四条处理路线（knowledge / calculator / weather / chat）
```

```python
def route(question):
    intent, confidence = _route_by_jev(question)        # ① Jev 优先
    if intent and confidence >= JEV_MIN_CONFIDENCE:
        return intent, confidence

    intent, confidence = _route_by_rules(question)      # ② 本地规则兜底
    return intent, confidence
```

**三条设计原则**：

| 原则 | 说明 |
|---|---|
| **低耦合** | Jev **只输出一个标签**，不碰提示词、不碰工具、不碰生成。换主模型、加新工具都不影响它 |
| **不盲信** | `confidence < 0.6` 时**不采信**，改用规则 —— 宁可不路由，也不错路由 |
| **降级要降到确定的那一档** | Jev 挂了**绝不退回"三工具 Agent 自主决策"**（成功率仅 1/3），而是退回**永远稳定**的本地规则 |

### 5. 效果

#### ① 路由准确率：6 / 6，置信度全为 1.0

`check_jev.py` 用 6 个中文问题实测：

| 问题 | 期望 | 预测 | 置信度 | 耗时 |
|---|---|---|---|---|
| 什么是过拟合 | knowledge | **knowledge** | 1.0 | 1.1s |
| 正则化有什么作用 | knowledge | **knowledge** | 1.0 | 1.1s |
| 请计算 sqrt(16) + 2**8 | calculator | **calculator** | 1.0 | 1.1s |
| 北京今天天气怎么样 | weather | **weather** | 1.0 | 2.3s |
| 你好啊 | chat | **chat** | 1.0 | 1.0s |
| 模型选择是什么意思 | knowledge | **knowledge** | 1.0 | 1.0s |

结果：**6 / 6 正确**。

**注意那列置信度 —— 全是 1.0，概率分布里非选项全是 0.0。**
它不是"猜对了"，是**完全确定**。这和本地 3b 在两个选项间摇摆
（两边概率接近、argmax 随机翻转）是两回事。

#### ② 决策这一环：从「概率性」变成「确定性」

| 能力 | 加 Jev 前（三工具 Agent 自主） | 加 Jev 后 |
|---|---|---|
| 走哪条处理链 | 模型临场判断，实测**约 1/3** 正确 | Jev 判断，**6/6** 正确 |
| 同一问题的结果 | **时对时错**（`temperature=0` 也拦不住） | **每次一致** |
| 把握程度可见吗 | 不可见 | **`confidence` + 完整概率分布** |
| 失败时的表现 | **静默选错**，用户完全看不出 | 返回低 confidence，**可被拦截** |
| 决策耗时 | 混在生成里，无法单独衡量 | 独立 **1.0 ~ 2.3 秒** |
| 成本 | 0 | 约 **440 token / 次** |
| 外部依赖 | 无 | **有**（境外服务，需代理） |

#### ③ 支撑四条路线，直接改善 RAG 可用性

Jev 的决策是"四条路各带一个工具"这个设计的前提 —— 有了它，模型才不需要"三选一"。

| 指标 | 引入前 | 引入后 |
|---|---|---|
| knowledge 路线检索率 | **0 / 4** | **100%** |
| 概念题单次耗时 | ~33 ~ 37s | **19.7s** |
| 前端「检索知识库」区块 | 时有时无（取决于模型当次是否调工具） | **稳定出现** |
| 每类问题的提示词 | 一套提示词兼顾所有情况 | **一套只管一件事** |
| 工具选择出错的可能 | 三选一，约 1/3 出错 | **一对一，选择难度≈0** |

> 说明：knowledge 检索率从 0/4 到 100%，是 **「Jev 路由」+「knowledge 改预检索」两步共同的结果** ——
> Jev 负责把问题分对路，预检索负责让检索必然发生。两者缺一不可。

#### ④ 端到端实测（`check_e2e.py`，24 / 24 通过）

| 问题 | 命中路线 | 工具内容 | 首字延迟 | 服务端耗时 |
|---|---|---|---|---|
| 什么是过拟合 | knowledge | 检索 **605 字** | 14.1s | **19.7s** |
| 请计算 sqrt(16) + 2**8 | calculator | `260.0` | 5.9s | 7.2s |
| 北京今天天气怎么样 | weather | 实时天气 | 5.7s | 7.4s |
| 你好啊 | chat | 无工具 | — | ~1s |

### 6. 启用方式

**不配置也能跑** —— Jev 不可用时会自动降级为本地关键词规则，功能不受影响。

想启用 Jev 路由：

```powershell
[Environment]::SetEnvironmentVariable("TYPESAFE_API_KEY", "你的key", "User")
```

API key 从 https://console.typesafe.ai 获取（有试用额度）。

> ⚠️ 两个已知注意点：
> 1. **设完环境变量必须重启 IDE / 终端** —— 进程只继承启动那一刻的环境快照。
>    否则 `app.py` 会读不到 key，日志里出现 `[router] 未配置 TYPESAFE_API_KEY`。
> 2. **Jev 是境外服务，需要代理**。若在某个进程里报 `SSLEOFError`，多半是该进程
>    没拿到代理变量（`requests` 只认进程环境变量）。此时会自动降级为本地规则，
>    服务照常工作 —— 这正是不把它放在关键路径上的意义。

---

## 快速开始

### 1. 准备模型

```bash
ollama pull qwen2.5:3b
ollama pull bge-m3
```

### 2. 安装依赖

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 3. 准备知识库

仓库**不含** PDF（体积原因）。请自行准备一份文本结构良好的 PDF 放入 `knowledge/`。

推荐演示用文档：**《动手学深度学习》** —— https://zh-v2.d2l.ai/d2l-zh.pdf
（配套脚本 `scripts/split_pdf.py` 可按章节拆分大 PDF）

然后构建索引（耗时操作，会对每一段调用一次嵌入模型）：

```bash
venv/Scripts/python.exe scripts/build_index.py
```

> 用 `RETRIEVE_K` / `page_content` 截断长度这类**检索期**参数改完立即生效；
> 改 `chunk_size` / `separators` 等**建索引期**参数必须删掉 `.chroma/` 重建。

### 4. 配置 Jev 路由（可选，但推荐）

**不配置也能跑** —— Jev 不可用时会自动降级为本地关键词规则，功能不受影响。

想启用 Jev 路由：

```powershell
[Environment]::SetEnvironmentVariable("TYPESAFE_API_KEY", "你的key", "User")
```

API key 从 https://console.typesafe.ai 获取（有试用额度）。

> 设完**必须重启 PyCharm / 终端** —— 进程只继承启动那一刻的环境快照。
> 原理、效果数据与两个已知注意点，见下文《核心特性：Jev 决策路由》。

### 5. 启动服务

```bash
venv/Scripts/python.exe app.py
```

浏览器打开 http://127.0.0.1:5023/

### 6. 跑测试

```bash
# 冒烟测试（不碰 AI 与网络）
venv/Scripts/python.exe -B -m pytest tests/ -v -p no:cacheprovider

# 端到端测试（真机 HTTP + SSE，独立端口 5099，测完自动关）
venv/Scripts/python.exe -B check_e2e.py
```

---

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 前端页面 |
| POST | `/query` | 非流式问答，返回 `{answer, elapsed, used_agent}` |
| **POST** | **`/query/stream`** | **流式问答，SSE 事件流** |
| GET | `/health` | 健康检查 |

**SSE 事件协议**：

```
data: {"type": "start"}
data: {"type": "tool", "name": "search_knowledge", "content": "…检索到的片段…"}
data: {"type": "reset"}
data: {"type": "token", "text": "过拟合"}
data: {"type": "done", "elapsed": 19.7}
```

| 事件 | 用途 |
|---|---|
| `tool` | 前端据此渲染「检索知识库」区块 |
| `reset` | **作废前面已显示的文字**（第一轮 LLM 的过渡语） |
| `token` | 答案增量 |

---

## 目录结构

```
agent_remake/
├── app.py                        入口：create_app() 工厂
├── config.py                     全部配置（路径已锚定项目根）
├── README.md
├── requirements.txt
├── core/                         能力层（禁止 import flask）
│   ├── llm.py                    模型工厂（唯一创建 LLM 的地方）
│   ├── prompts.py                四套专用提示词 + SYSTEM_PROMPT 兜底
│   ├── tools.py                  三个工具
│   ├── rag.py                    索引构建与检索
│   └── agent_builder.py          按路线组装 Agent（带缓存）
├── services/
│   ├── router.py                 路由层：Jev + 本地规则
│   └── qa_service.py             编排层：分发 + 超时 + 降级
├── web/
│   ├── routes.py                 蓝图：/ /query /query/stream /health
│   └── templates/index.html      前端页面（流式渲染 + 耗时徽章）
├── scripts/
│   ├── build_index.py            一次性构建向量索引
│   ├── split_pdf.py              按章节拆分大 PDF
│   └── peek_pdf.py               抽查 PDF 文本质量 / 关键词定位
├── tests/
│   └── test_smoke.py             冒烟测试（10 个用例）
├── check_e2e.py                  端到端测试（24 项断言）
├── check_route.py                路由 + 流式事件回归
├── check_*.py                    其他开发期诊断脚本（可删）
├── knowledge/                    放 PDF（不入库）
└── .chroma/                      向量库（不入库，可重建）
```

---

## 设计要点

- **分层铁律**：依赖单向 `web → services → core`，`core/` 内不出现任何 HTTP 代码
- **路由分层**：Jev 只输出一个标签，不碰提示词、不碰工具、不碰生成（低耦合）；
  它不可用时退回**本地关键词规则**，而**不是**退回"三工具 Agent 自主决策"——后者成功率仅约 1/3
- **确定性优先**：凡是代码能确定完成的（检索），就不交给模型判断；
  模型只做它擅长的（理解语义、从自然语言里抽参数）
- **单工具路线**：每条路由只挂一个工具，把模型的"三选一"降级成"一对一"
- **索引离线构建**：向量化是重操作，走 `scripts/build_index.py` 单独执行，绝不放在 Web 请求里
- **懒加载 + 降级**：索引不存在时 `get_retriever()` 返回 `None`，RAG 降级为"工具不可用"，服务照常启动
- **工具自吞异常**：异常一旦逃逸会中断整轮 Agent 推理，因此工具必须把错误转成字符串返回
- **中文切块**：`separators` 显式包含中文标点，并预先清洗 PDF 的"视觉换行"
- **路径锚定**：所有目录基于 `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`，不依赖当前工作目录
- **流式不缓存**：token 直接下发，需要作废时发 `reset` 事件（早期"先缓存后补发"的方案会让
  所有不调工具的回答退化成一次性输出）

---

## 开发历程

从 2026-09-20 到 09-22，三天里跨过五道坎。下面按时间记录**改了什么、为什么、结果如何**。

### 阶段一 · 09-20　项目重建（classic → v1）

按"依赖倒序"分批重写 13 个文件：

| 批次 | 文件 |
|---|---|
| 1 | `config.py`、`core/__init__.py`、`core/llm.py`、`core/prompts.py` |
| 2 | `core/rag.py`、`core/tools.py`、`core/agent_builder.py` |
| 3 | `services/qa_service.py`、`web/routes.py`、`app.py` |
| 4 | `requirements.txt`、`tests/test_smoke.py` |

**踩到的三个坑**：

| 坑 | 现象 | 根因与解法 |
|---|---|---|
| 模型填反 | `"bge-m3" does not support chat` | `CHAT_MODEL` 与 `EMBED_MODEL` 写反了。**模型分用途，嵌入模型不会对话** |
| 依赖缺失 | `No module named 'pypdf'` / `'chromadb'` | 补齐依赖 |
| **相对路径** | `scripts/` 下凭空多出一个空的 `knowledge/` | `KNOWLEDGE_DIR = "knowledge"` 依赖**进程的当前工作目录**（CWD），而 PyCharm 运行时 CWD 是脚本所在目录。改用 `BASE_DIR` 锚定项目根 |

### 阶段二 · 09-20　`llama3.2:1b` 的工具调用不可用

三种失败模式，同一个根源（模型分不清"上下文里的东西"和"自己该做的事"）：

| # | 现象 |
|---|---|
| 1 | 把参数 **schema 定义**当参数值填进去 |
| 2 | `content` 里复读工具描述、schema 片段 |
| 3 | **伪造工具结果**（工具报错后自己编了一个"成功返回"的 JSON） |

经联网查证：`llama3.2:1b` **支持** tool calling，但在多工具、需要"构造参数"的场景下不可靠。
**换 `qwen2.5:3b` 后立刻正常** —— 参数层级正确、工具真正执行、噪声消失。

> 结论：**模型能力问题要换模型解决，不是改代码解决。**

### 阶段三 · 09-22 上午　RAG 质量的三个层次

**① 文档层**：检索"什么是过拟合"的相似度分数挤在 0.23~0.32，
**真正相关的那段只排第二**。加阈值反而会误杀 —— 根因是源文档。
原 PDF 是**思维导图/卡片式结构**（标题+说明交替、硬换行切断词语），
换成《动手学深度学习》连续文本后，分数拉开到 **0.43~0.51 且排序正确**。

**② 切块层**：句子被切成半截。两个原因：
`separators` 里 `\n` 排在 `。` 前面（PDF 的行边界不是语义边界）；
PDF 的**"视觉换行"**把词拆开（"保" / "留"）。
解法：先清洗视觉换行 → 中文标点提到换行之前 → 过滤 50 字以下的碎片块。
结果：最短块 12 → **51 字**，三个片段全部在句号处断开，检索分数全面提升。

**③ 性能层**：逐段计时定位到瓶颈 —— 生成为 **6.4 token/s**，读输入为 **165.6 token/s**（快 26 倍）。
→ **杠杆比 26:1，唯一有效的优化是"让模型少写"**。
加 `num_predict=192`（Ollama 原生硬限制）+ 提示词"200 字以内"，
回答 338 → 186 字，单次问答 **61s → 33s**。

### 阶段四 · 09-22 上午　工程化

- 冒烟测试 `tests/test_smoke.py`：**10/10 通过**（覆盖工具、配置、RAG 清洗、Agent 组装、Web 四层）
- 首次 Git 提交，仓库：`langchain-v1-agent`
- 补齐 `README.md`

### 阶段五 · 09-22 下午　交互与路由

**① 前端重做（15 行 → 约 400 行）**
聊天气泡布局、示例问题、**等待期实时计时器**、耗时徽章（`<8s` 绿 / `<45s` 橙 / `≥45s` 红）、
模式徽章、检索内容区块。

**② 流式输出（SSE）**
先写探路脚本确认 LangGraph `stream_mode=["messages","updates"]` 的返回结构，
再实现 `/query/stream`。
关键设计是 `reset` 事件：Agent 有两轮 LLM，第一轮会吐"我查一下资料"这类过渡语，
检测到 `chunk.tool_call_chunks` 就发 `reset` 让前端清空。

> 早期方案是"把 token 先缓存，发现调工具就丢弃"，**结果所有不调工具的回答全被缓存**，
> 流式退化成一次性输出 —— 这个失败方案也值得记住。

**③ Jev 路由**
`Jev`（TypeSafe System One）是**不做文本生成的决策模型**，只返回带概率的类型化标签。
实测中文路由 **6/6 正确、置信度全为 1.0**。
集成后做成两级：Jev 优先，不可用时**退回本地关键词规则**。

> 踩坑：`app.py` 从 PyCharm 启动时读不到代理，Jev 报 `SSLEOFError`。
> 经探测确认 `api.typesafe.ai` 本身可达（经代理返回 405）—— 纯粹是**进程环境快照**问题。
> 这次故障也证明了"降级链必须降到你确定的那一档"。

**④ knowledge 路线改预检索（本项目最关键的一次修复）**

四条路线实现后，单工具 knowledge 路线连跑 4 次**全部跳过检索**（0/4），
答案逐字一致的「知识库中没有找到相关内容」。

**根因**：`KNOWLEDGE_PROMPT` 里有一句
`资料不足以回答时，明确说明"知识库中没有找到相关内容"` —— 模型推理成
「要求说回答必须来自资料 → 我手上没有资料 → 按这条输出」→ **跳过了"先去取资料"**。

**这是同一机制的第三次发作**：

| # | 提示词里的那句话 | 模型的反应 |
|---|---|---|
| 1 | "工具调用失败就告诉用户**检索失败**" | 直接输出「检索失败」 |
| 2 | "闲聊…**不要调用工具**" | 从此彻底不调工具 |
| 3 | "资料不足时说明**知识库中没有找到相关内容**" | 输出这句 + 跳过检索 |

> ### 📌 规律
> **提示词里任何一句"完整、可直接照搬的答复文本"，弱模型都会把它当成当前情况下
> 可以使用的快捷答案。** 写提示词时必须把它变成**有前置条件、不可独立成立**的表述。

**解法**：knowledge 路线改走 **代码直接检索 → 推送检索内容 → 裸 LLM + RAG 模板**，
完全不经过 Agent。模型没有可以"跳过"的空间。

### 各阶段数据对比

| 指标 | 初始 | 现在 |
|---|---|---|
| 工具调用成功率 | ~0%（1b）/ ~33%（3b 三工具） | **100%**（路由 + 单工具） |
| knowledge 检索率 | 0/4 | **100%** |
| 单次问答耗时 | 60s 超时（实际 >120s） | **19.7s**（概念题） |
| 交互 | 等 30 秒一次性出结果 | **逐字流式 + 实时计时器** |
| 检索可见性 | 只能看日志 | **前端检索区块** |
| 测试 | 无 | 冒烟 10/10 + **E2E 24/24** |

---

## 踩坑与经验

1. **先探路再写码**：涉及不确定的框架 API（`stream_mode` 结构、`create_agent` 签名），
   先写 10 行探路脚本实测，比凭记忆写完再看报错快得多。

2. **降级链要降到你确定的那一档**：外部依赖挂掉时，退回"成功率 1/3 的老方案"
   等于没降级。宁可退回一个"能力弱但必然成功"的方案。

3. **看现象先看耗时**：一次完整的工具调用至少需要 2 次 LLM 调用。
   **耗时 0.9~2.5 秒的"答案"必然没调工具** —— 这是最省事的判据。

4. **让代码做确定的事，让模型做需要理解的事**：
   检索不需要判断力（检索词就是问题本身）→ 交给代码；
   从自然语言里抽参数（城市名、表达式）需要理解 → 交给模型。

5. **提示词里的"台词"会被当成答案**（见上文三次发作的规律）。

6. **格式与措辞同等重要**：把工具说明从中文完整句子改成 `- 条件 → 动作` 的箭头列表后，
   模型反而不再调工具 —— 符号化表达落在小模型的训练分布之外。

7. **达到目标就停手**：性能调到 20 秒满足需求后，不再为省 1 秒牺牲检索质量。

8. **不要为验证而重跑破坏性操作**：`Chroma.from_documents` 对已存在的库是**追加**而非覆盖，
   重跑 `build_index.py` 会产生重复向量 —— 换文档或换嵌入模型时**必须先删 `.chroma/`**。

---

## v1 相对 classic 的主要差异

| 维度 | classic | v1 |
|---|---|---|
| 创建 Agent | `create_react_agent(llm, tools, prompt)` | `create_agent(model, tools, system_prompt=...)` |
| 执行器 | `AgentExecutor(agent=..., tools=...)` | 不存在，`create_agent` 返回值直接可执行 |
| 调用入参 | `{"input": q}` | `{"messages": [{"role": "user", "content": q}]}` |
| 取结果 | `result["output"]` | `result["messages"][-1].content` |
| 提示词 | 一大段 ReAct 格式约定 + few-shot 示例 | 每套提示词只描述一件事 |
| 输出解析 | `ReActSingleInputOutputParser` + `handle_parsing_errors` | 概念消失（模型原生返回 `tool_calls`） |
| 步数限制 | `max_iterations=4` | `config={"recursion_limit": N}`（N 是图步数，一轮工具调用 = 2 步） |
| 流式 | `astream_events` | `agent.stream(..., stream_mode=["messages","updates"])` |
| 底层机制 | 提示词约定文本格式 + 正则解析 | LangGraph 图 + 模型原生结构化输出 |

核心变化：**老路线靠"提示词约定文本格式 → 正则解析"，v1 靠"模型原生返回 `tool_calls`
结构化字段"**，解析器因此失去存在意义。

> ⚠️ 但要注意：**v1 对模型的要求高于 classic**。classic 只需模型模仿文本格式，
> v1 需要模型输出正确的嵌套结构 —— 在 1B 这个量级的模型上，老办法反而更容易工作。

---

## 后续可做

| 方向 | 说明 |
|---|---|
| **对话记忆** | `create_agent(..., checkpointer=InMemorySaver())` + `config={"configurable": {"thread_id": ...}}`。目前**完全无状态**，"|
| **引用溯源** | `Document.metadata` 里有 `source` / `page`，拼进资料即可让答案标注"来自第 16 页" |
| 修复 Jev 代理 | 把代理设为用户级环境变量并重启 IDE，路由即可稳定走 Jev |
| 换更小模型提速 | `qwen2.5:1.5b` 生成速度约翻倍 |
