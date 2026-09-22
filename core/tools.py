"""工具定义。每个工具的 docstring 就是写给模型看的说明书，必须讲清"什么时候用"。

铁律：工具函数必须自己吞掉异常并返回字符串。
异常一旦逃逸出工具，整轮 Agent 推理会被直接中断。
"""
import math

import requests
from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。expression 是合法 Python 表达式，例如 'sqrt(16)' 或 '2**8'。"""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_weather(city: str) -> str:
    """查询城市实时天气。city 是城市名，中英文均可，例如 '北京' 或 'Beijing'。"""
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3&lang=zh", timeout=8)
        if resp.status_code == 200:
            return resp.text.strip()
        return f"查询失败: HTTP {resp.status_code}"
    except Exception as e:
        return f"查询失败: {e}"


@tool
def search_knowledge(query: str) -> str:
    """在本地知识库中检索相关内容。当问题涉及机器学习概念、术语、原理时使用本工具。
    query 是检索用的关键词或问题，例如 '什么是过拟合' 或 '正则化的作用'。"""
    from core.rag import get_retriever   # 延迟导入：避免启动时就去碰索引

    retriever = get_retriever()
    if retriever is None:
        return "知识库索引尚未构建（请先运行 scripts/build_index.py）"
    docs = retriever.invoke(query)
    if not docs:
        return "未检索到相关内容"
    return "\n---\n".join(d.page_content[:300] for d in docs)
