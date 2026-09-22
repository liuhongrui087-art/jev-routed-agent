"""第 2 批验证：跑 Agent 闭环，打印每步的长度与首尾。用完可删。"""
from core.agent_builder import get_agent
import time
t0 = time.time()

result = get_agent().invoke(
    {"messages": [{"role": "user", "content": "什么是过拟合"}]}
)

for i, m in enumerate(result["messages"]):
    name = type(m).__name__
    print(f"\n--- 消息 {i} [{name}] ---")
    if m.content:
        text = m.content
        print(f"  content 长度: {len(text)} 字")
        print(f"  content 开头: {text[:80]}")
        print(f"  content 结尾: {text[-80:]}")
    tcs = getattr(m, "tool_calls", None) or []
    print(f"  tool_calls  : {len(tcs)} 个")
    for tc in tcs:
        print(f"    - {tc['name']}({tc['args']})")

print(f"\n===== 最终回答 =====\n{result['messages'][-1].content}")
print(f"\n总耗时: {time.time() - t0:.1f}s")
