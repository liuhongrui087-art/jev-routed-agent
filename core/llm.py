"""模型工厂：全项目唯一创建 LLM / Embedding 的地方。

统一走 langchain_ollama（langchain_community 的 Ollama 集成在本机有版本冲突）。
故意不做单例缓存 —— 每次返回新对象，改配置后立即生效。
"""
from langchain_ollama import ChatOllama, OllamaEmbeddings

import config


def get_chat_llm() -> ChatOllama:
    """对话模型：负责推理与生成。"""
    return ChatOllama(
        model=config.CHAT_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,      # 确定性输出，减少格式漂移
        num_ctx=4096,       # 上下文窗口：系统提示词 + 工具结果 + 历史
    )


def get_embed_model() -> OllamaEmbeddings:
    """嵌入模型：文本 -> 向量，与对话模型相互独立。"""
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
