# LangChain v1 多步骤推理机器人

基于 **LangChain v1** + **Flask** + **Ollama** 的多步骤推理机器人。

用户在前端提问 → Agent 自主决定是否调用工具（数学计算 / 实时天气 / 本地知识库检索）→ 返回有依据的中文答案。

> 本项目是经典 ReAct 实现（Flask + LangChain classic + Ollama + RAG）的 **v1 重写版**，
> 用于对比两代 Agent API 的设计差异。

## 技术栈

| 层 | 技术 |
|---|---|
| Agent | LangChain v1 —— `create_agent`（原生 tool calling） |
| 底层运行时 | LangGraph（`create_agent` 返回的是一张已编译的图） |
| 对话模型 | Ollama `qwen2.5:3b` |
| 嵌入模型 | Ollama `bge-m3`（1024 维，中文友好） |
| 向量库 | ChromaDB，持久化到 `.chroma/` |
| Web | Flask 3 + 蓝图 |

## 架构

依赖单向：`web → services → core → config.py`

```
浏览器
  │  POST /query
  ▼
web/routes.py            接口层：只管 HTTP（收请求、校验、序列化）
  ▼
services/qa_service.py   编排层：超时控制 + 降级
  ▼
core/                    能力层
  ├── agent_builder.py    组装 Agent（create_agent）
  ├── tools.py            三个工具：calculator / get_weather / search_knowledge
  ├── rag.py              知识库索引与检索
  ├── llm.py              模型工厂
  └── prompts.py          系统提示词
  ▼
Ollama（qwen2.5:3b 对话 / bge-m3 嵌入）
```

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

然后构建索引（耗时操作，会对每一段调用一次嵌入模型）：

```bash
venv/Scripts/python.exe scripts/build_index.py
```

### 4. 启动服务

```bash
venv/Scripts/python.exe app.py
```

浏览器打开 http://127.0.0.1:5023/

### 5. 跑冒烟测试

```bash
venv/Scripts/python.exe -B -m pytest tests/ -v -p no:cacheprovider
```

## 目录结构

```
agent_remake/
├── app.py                        入口：create_app() 工厂
├── config.py                     全部配置（路径已锚定项目根）
├── requirements.txt
├── core/                         能力层（禁止 import flask）
│   ├── llm.py                    模型工厂
│   ├── prompts.py                系统提示词
│   ├── tools.py                  三个工具
│   ├── rag.py                    索引构建与检索
│   └── agent_builder.py          Agent 组装
├── services/
│   └── qa_service.py             超时 + 降级编排
├── web/
│   ├── routes.py                 蓝图：/ /query /health
│   └── templates/index.html      前端页面
├── scripts/
│   ├── build_index.py            一次性构建向量索引
│   ├── split_pdf.py              按章节拆分大 PDF
│   └── peek_pdf.py               抽查 PDF 文本质量 / 关键词定位
├── tests/
│   └── test_smoke.py             冒烟测试（10 个用例）
├── check_*.py                    开发期验证脚本（可删）
└── knowledge/                    放 PDF（不入库）
```

## 设计要点

- **分层铁律**：依赖单向 `web → services → core`，`core/` 内不出现任何 HTTP 相关代码
- **索引离线构建**：向量化是重操作，走 `scripts/build_index.py` 单独执行，绝不放在 Web 请求里
- **懒加载 + 降级**：索引不存在时 `get_retriever()` 返回 `None`，RAG 降级为"工具不可用"，服务照常启动
- **工具自吞异常**：异常一旦逃逸会中断整轮 Agent 推理，因此工具必须把错误转成字符串返回
- **中文切块**：`separators` 显式包含中文标点，并预先清洗 PDF 的"视觉换行"
- **路径锚定**：所有目录基于 `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`，不依赖当前工作目录

## v1 相对 classic 的主要差异

| 维度 | classic | v1 |
|---|---|---|
| 创建 Agent | `create_react_agent(llm, tools, prompt)` | `create_agent(model, tools, system_prompt=...)` |
| 执行器 | `AgentExecutor(agent=..., tools=...)` | 不存在，`create_agent` 返回值直接可执行 |
| 调用入参 | `{"input": q}` | `{"messages": [{"role": "user", "content": q}]}` |
| 取结果 | `result["output"]` | `result["messages"][-1].content` |
| 提示词 | 一大段 ReAct 格式约定 + few-shot 示例 | 一句 `system_prompt` |
| 输出解析 | `ReActSingleInputOutputParser` + `handle_parsing_errors` | 概念消失（模型原生返回 `tool_calls`） |
| 步数限制 | `max_iterations=4` | `config={"recursion_limit": N}`（N 是图步数，一轮工具调用 = 2 步） |
| 底层机制 | 提示词约定文本格式 + 正则解析 | LangGraph 图 + 模型原生结构化输出 |

核心变化：**老路线靠"提示词约定文本格式 → 正则解析"，v1 靠"模型原生返回 `tool_calls` 结构化字段"**，
解析器因此失去存在意义。
