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

    tool_started = False

    try:
        for mode, payload in get_agent().stream(
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
