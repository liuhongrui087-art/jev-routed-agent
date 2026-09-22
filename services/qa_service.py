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


def handle_question(question: str) -> str:
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

    print(f"[qa] q={question[:30]!r} agent_used={used_agent} elapsed={time.time() - t0:.1f}s")
    return answer
