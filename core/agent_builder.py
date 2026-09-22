"""组装 Agent（LangChain v1）。

与 classic 版的三点关键差异：
1. 没有 AgentExecutor —— create_agent 返回的图对象本身就可执行；
2. 没有提示词模板与 ReAct 输出解析器 —— 模型原生返回 tool_calls，不靠正则解析文本；
3. 单例懒加载 —— 首次调用才构建，import 这个模块不会去连模型。
"""
from langchain.agents import create_agent

from core.llm import get_chat_llm
from core.prompts import SYSTEM_PROMPT
from core.tools import calculator, get_weather, search_knowledge

TOOLS = [search_knowledge, calculator, get_weather]

_agent = None


def get_agent():
    """单例懒加载：第一次调用才组装。"""
    global _agent
    if _agent is None:
        _agent = create_agent(
            model=get_chat_llm(),
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent
