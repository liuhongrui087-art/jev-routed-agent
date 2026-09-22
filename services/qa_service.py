"""编排层：串起 Agent 调用，并处理超时与降级。

两层兜底：
  第一层：Agent 调用（限时 AGENT_TIMEOUT 秒）
    │ 超时     ──▶ 第二层：裸 LLM 直接回答
    └ 其他异常 ──▶ 第二层：裸 LLM 直接回答

⚠️ ThreadPoolExecutor 不用 with 语句：
with 块退出时会 wait=True 等任务跑完，导致 future.result(timeout) 的超时形同虚设。
必须手动 shutdown(wait=False)。

v1 相对 classic 的改动：
1. 入参：{"input": q}  →  {"messages": [{"role": "user", "content": q}]}
2. 取结果：result["output"]  →  result["messages"][-1].content
3. 删掉了 classic 里判断 "Agent stopped due to iteration limit" 的第三层降级
   —— v1 没有文本解析环节，不会再产生这种文本
"""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import config
from core.agent_builder import get_agent
from core.llm import get_chat_llm
from core.prompts import KNOWLEDGE_PROMPT
from services.router import route


def handle_question(question: str) -> dict:
    """回答一个问题，并返回本次调用的元信息（非流式）。

    返回：
        {
            "answer":     str,    # 回答正文
            "elapsed":    float,  # 服务端耗时（秒，1 位小数）
            "used_agent": bool,   # False 表示走了降级（超时或异常）
        }
    """
    t0 = time.time()
    llm = get_chat_llm()
    used_agent = True

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        get_agent().invoke,
        {"messages": [{"role": "user", "content": question}]},
        {"recursion_limit": config.AGENT_MAX_ITERATIONS * 2},
    )
    try:
        result = future.result(timeout=config.AGENT_TIMEOUT)
        answer = result["messages"][-1].content
    except TimeoutError:
        used_agent = False
        print(f"[qa] agent 超时（{config.AGENT_TIMEOUT}s），走降级")
        answer = llm.invoke(f"{question}\n请在 2-3 句话内直接给出答案").content
    except Exception as e:
        used_agent = False
        print(f"[qa] agent 异常，走降级: {type(e).__name__}: {e}")
        answer = llm.invoke(question).content
    finally:
        executor.shutdown(wait=False)   # 不等后台任务，超时立即返回

    elapsed = round(time.time() - t0, 1)
    print(f"[qa] q={question[:30]!r} agent_used={used_agent} elapsed={elapsed}s")
    return {"answer": answer, "elapsed": elapsed, "used_agent": used_agent}

def _prefetch_context(question: str):
    """无条件检索知识库，返回 (给模型看的资料文本, 原始片段列表)。

    用于 knowledge 路线：不让模型决定"要不要查知识库"，直接把资料准备好塞进 prompt。
    """
    from core.rag import get_retriever

    retriever = get_retriever()
    if retriever is None:
        print("[qa] 预检索跳过：索引不存在")
        return "", []

    try:
        docs = retriever.invoke(question)
    except Exception as e:
        print("[qa] 预检索失败: " + type(e).__name__ + ": " + str(e))
        return "", []

    if not docs:
        return "", []

    context = "\n---\n".join(
        "[第 " + str(d.metadata.get("page", "?")) + " 页]\n" + d.page_content[:300]
        for d in docs
    )
    return context, docs

def stream_question(question: str):
    """流式回答，逐步 yield 事件字典（供 SSE 使用）。

    事件类型：
        {"type": "start"}
        {"type": "reset"}                               # 作废前面已显示的文字
        {"type": "tool", "name": str, "content": str}   # 工具执行结果（检索内容）
        {"type": "token", "text": str}                  # 答案的增量文本
        {"type": "done", "elapsed": float}
        {"type": "error", "message": str}

    处理「第一轮过渡语」的方式：
        Agent 决定调工具时，往往先吐一句"我查一下资料"之类的过渡语。
        chunk 里一旦出现 tool_call_chunks，说明这一轮在调工具，
        此时发一个 reset 事件，让前端把已显示的文字作废。
        （早先的写法是把所有 token 先缓存起来，但那会导致
          "不调工具的回答"也要等整段生成完才一次性吐出，失去流式效果。）
    """
    t0 = time.time()
    yield {"type": "start"}

    # ---------- 路由 ----------
    intent, confidence = route(question)
    print(f"[qa] 路由 = {intent}  置信度 = {confidence}")

    # ---------- knowledge：预检索 + 裸 LLM（不经过 Agent）----------
    #
    # 为什么不交给 Agent：实测把"必须先调用 search_knowledge"写成 Agent 提示词后，
    # qwen2.5:3b 连跑 4 次全部跳过检索（0/4），直接吐出提示词里那句现成的
    # "知识库中没有找到相关内容"。检索改由代码执行后，模型没有可跳过的空间。
    if intent == "knowledge":
        context, docs = _prefetch_context(question)
        if docs:
            yield {
                "type": "tool",
                "name": "search_knowledge",
                "content": "\n---\n".join(d.page_content[:300] for d in docs),
            }
            print(f"[qa] 预检索命中 {len(docs)} 段，上下文 {len(context)} 字")
            try:
                prompt = KNOWLEDGE_PROMPT.format(context=context, question=question)
                for chunk in get_chat_llm().stream(prompt):
                    text = getattr(chunk, "content", "") or ""
                    if text:
                        yield {"type": "token", "text": text}
            except Exception as e:
                yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
            yield {"type": "done", "elapsed": round(time.time() - t0, 1)}
            return
        print("[qa] 预检索无结果，退回 fallback Agent")

    # ---------- 其余路线（calculator / weather / chat / 兜底）继续走 Agent ----------
    route_name = intent if intent in ("calculator", "weather", "chat") else "fallback"
    print(f"[qa] 采用路线 = {route_name}")

    tool_started = False

    try:
        for mode, payload in get_agent(route_name).stream(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": config.AGENT_MAX_ITERATIONS * 2},
            stream_mode=["messages", "updates"],     # messages 拿 token，updates 拿工具结果
        ):
            if mode == "messages":
                chunk = payload[0] if isinstance(payload, tuple) else payload

                # 这一轮在生成工具调用 → 前面流出的都是过渡语，作废
                if getattr(chunk, "tool_call_chunks", None):
                    if not tool_started:
                        tool_started = True
                        yield {"type": "reset"}
                    continue

                text = getattr(chunk, "content", "") or ""
                if text:
                    yield {"type": "token", "text": text}

            elif mode == "updates" and isinstance(payload, dict):
                # 遍历所有节点，不硬编码 "model" / "tools" 这些名字
                for update in payload.values():
                    if not isinstance(update, dict):
                        continue
                    for msg in update.get("messages", []):
                        if type(msg).__name__ == "ToolMessage":
                            yield {
                                "type": "tool",
                                "name": getattr(msg, "name", "tool"),
                                "content": msg.content,
                            }

    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}

    yield {"type": "done", "elapsed": round(time.time() - t0, 1)}
