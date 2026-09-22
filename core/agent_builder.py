"""组装 Agent（LangChain v1）。

与 classic 版的三点关键差异：
1. 没有 AgentExecutor —— create_agent 返回的图对象本身就可执行；
2. 没有提示词模板与 ReAct 输出解析器 —— 模型原生返回 tool_calls，不靠正则解析文本；
3. 按路由缓存 —— 每条路一个只带单工具的 Agent，组装一次反复用。

    为什么按路拆分：
    qwen2.5:3b 在"三个工具里该选哪个"这件事上只有约 1/3 的成功率。
    由 Jev 先把路由定下来，每条路只带一个工具，模型就不需要选，只需要填参数。

    fallback 是带全部工具的旧 Agent，仅在"预检索拿不到资料"时兜底。
"""
from langchain.agents import create_agent

from core.llm import get_chat_llm
from core.prompts import (
    CALCULATOR_PROMPT,
    CHAT_PROMPT,
    SYSTEM_PROMPT,
    WEATHER_PROMPT,
)
from core.tools import calculator, get_weather, search_knowledge

# 每条路由对应的 (工具集, 专用提示词)
#
# 注意：这里**没有 knowledge** —— 概念题走的是
# 「qa_service 预检索 → 裸 LLM + KNOWLEDGE_PROMPT 模板」，
# 不经过 Agent。原因见 core/prompts.py 里 KNOWLEDGE_PROMPT 上方的说明
# （单工具 Agent 实测 0/4 会跳过检索）。
_ROUTES = {
    "calculator": ([calculator], CALCULATOR_PROMPT),
    "weather": ([get_weather], WEATHER_PROMPT),
    "chat": ([], CHAT_PROMPT),
}

_agents = {}
_fallback = None


def get_agent(route_name: str = "fallback"):
    """按路由名取对应的 Agent（组装一次后缓存复用）。

    route_name 取值：
        calculator / weather / chat —— 单工具 Agent（chat 不带工具）
        fallback（默认）—— 带全部工具的老 Agent，预检索拿不到资料时兜底
    """
    global _fallback

    if route_name == "fallback" or route_name not in _ROUTES:
        if _fallback is None:
            _fallback = create_agent(
                model=get_chat_llm(),
                tools=[search_knowledge, calculator, get_weather],
                system_prompt=SYSTEM_PROMPT,
            )
        return _fallback

    if route_name not in _agents:
        tools, prompt = _ROUTES[route_name]
        _agents[route_name] = create_agent(
            model=get_chat_llm(),
            tools=tools or None,      # chat 路线不带工具
            system_prompt=prompt,
        )
    return _agents[route_name]
