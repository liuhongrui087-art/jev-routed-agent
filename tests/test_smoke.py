"""冒烟测试：只测确定性逻辑，不碰 AI 与网络。

用法：
    D:\agent_remake\venv\Scripts\python.exe -m pytest tests/ -v -p no:cacheprovider
"""
import os

from core.tools import calculator


# ---------------- 工具层 ----------------

def test_calculator_basic():
    assert calculator.invoke("sqrt(16) + 2**8") == "260.0"


def test_calculator_error_handling():
    """工具必须自吞异常、返回字符串，绝不能抛出。"""
    result = calculator.invoke("not_a_python_expression!!")
    assert result.startswith("计算错误")


def test_calculator_blocks_builtins():
    """eval 白名单必须挡住内置函数 —— 安全底线。"""
    assert calculator.invoke("open('x')").startswith("计算错误")


# ---------------- 配置层 ----------------

def test_config_paths_are_absolute():
    """路径必须锚定到项目根，不依赖当前工作目录。

    这是踩过的坑：曾经因为相对路径，把 knowledge/ 建到了 scripts/ 下面。
    """
    import config

    assert os.path.isabs(config.KNOWLEDGE_DIR)
    assert os.path.isabs(config.CHROMA_DIR)


# ---------------- RAG 文本清洗 ----------------

def test_clean_pdf_text_joins_visual_newlines():
    """PDF 的视觉换行必须被拼回去，段落空行要保留。"""
    from core.rag import _clean_pdf_text

    raw = "第一行被换\n行切断了。\n\n这是第二段。"
    out = _clean_pdf_text(raw)

    assert "第一行被换行切断了。" in out      # 行内换行已拼合
    assert "\n\n" in out                       # 段落分隔保留


# ---------------- Agent 组装（只验证"能装起来"，不调用模型）----------------

def test_agent_builds():
    from core.agent_builder import get_agent

    assert get_agent() is not None


def test_agent_is_singleton():
    """懒加载单例：两次调用必须是同一个对象。"""
    from core.agent_builder import get_agent

    assert get_agent() is get_agent()


# ---------------- Web 层 ----------------

def test_health():
    from app import create_app

    client = create_app().test_client()
    assert client.get("/health").status_code == 200


def test_query_empty_returns_400():
    from app import create_app

    client = create_app().test_client()
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_query_without_json_returns_400():
    """请求体不是合法 JSON 时，也要返回 400 JSON，而不是 415。"""
    from app import create_app

    client = create_app().test_client()
    r = client.post("/query", data="not json at all")
    assert r.status_code == 400
