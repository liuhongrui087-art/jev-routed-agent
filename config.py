"""全局配置：模型、超时、检索、服务端口。只放常量，不写任何逻辑。"""
import os

# 项目根目录（即 config.py 所在目录）。用它把下面两个目录锚成绝对路径，
# 这样不管从哪个目录启动进程，都能找到 knowledge/ 和 .chroma/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:3b"      # 对话模型：负责推理与生成，必须支持 tool calling
EMBED_MODEL = "bge-m3"          # 嵌入模型：文本转向量，中文友好

# --- Agent ---
AGENT_TIMEOUT = 60              # 1B 冷启动约 3.5s，8 秒必然误判超时
AGENT_MAX_ITERATIONS = 4        # v1 下对应 invoke 的 recursion_limit

# --- RAG ---
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")   # PDF 存放目录（绝对路径）
CHROMA_DIR = os.path.join(BASE_DIR, ".chroma")        # 向量库目录（绝对路径）
RETRIEVE_K = 3                                        # 每次检索返回的片段数

# --- Flask ---
HOST = "127.0.0.1"              # 仅本机可访问，0.0.0.0 会暴露给局域网
PORT = 5023
DEBUG = True
