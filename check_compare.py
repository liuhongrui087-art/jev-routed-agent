"""对比「非流式」与「流式」两条路径的答案，定位「复读资料」发生在哪一层。

用法（项目根目录；跑之前先停掉 app.py，避免两个进程抢 Ollama）：
    D:\agent_remake\venv\Scripts\python.exe D:\agent_remake\check_compare.py
"""
from core.agent_builder import get_agent

QUESTION = "什么是过拟合"
agent = get_agent()


def show(title, text):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("长度: " + str(len(text)) + " 字")
    print("--- 全文 ---")
    print(text)


# ---------- 1. 非流式 ----------
print("\n>>> 跑非流式 invoke() ...")
r1 = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
msgs = r1["messages"]

print("\n[消息链]")
for i, m in enumerate(msgs):
    tcs = getattr(m, "tool_calls", None) or []
    print("  " + str(i) + ". " + type(m).__name__ +
          "  content=" + str(len(m.content or "")) + " 字  tool_calls=" + str(len(tcs)))
    for tc in tcs:
        print("       -> " + str(tc["name"]) + "(" + str(tc["args"]) + ")")

answer_sync = msgs[-1].content
show("【非流式】最终答案", answer_sync)


# ---------- 2. 流式 ----------
print("\n\n>>> 跑流式 stream() ...")
chunks = []
tool_snippets = []

for mode, payload in agent.stream(
    {"messages": [{"role": "user", "content": QUESTION}]},
    config={"recursion_limit": 8},
    stream_mode=["messages", "updates"],
):
    if mode == "messages":
        chunk = payload[0] if isinstance(payload, tuple) else payload
        text = getattr(chunk, "content", "") or ""
        if text:
            chunks.append(text)
    elif mode == "updates" and isinstance(payload, dict):
        for update in payload.values():
            if not isinstance(update, dict):
                continue
            for m in update.get("messages", []):
                if type(m).__name__ == "ToolMessage":
                    tool_snippets.append(m.content)

answer_stream = "".join(chunks)
show("【流式】拼接后的答案", answer_stream)


# ---------- 3. 对比 ----------
print("\n\n" + "=" * 70)
print("对比结论")
print("=" * 70)
print("非流式长度 : " + str(len(answer_sync)) + " 字")
print("流式长度   : " + str(len(answer_stream)) + " 字")
print("工具调用   : " + (("是，共 " + str(len(tool_snippets)) + " 条 ToolMessage")
                         if tool_snippets else "否"))

if answer_sync and answer_stream and answer_sync == answer_stream:
    print("→ 两条路径完全一致：问题在模型/提示词层，与流式实现无关")
elif answer_sync and answer_stream:
    print("→ 两条路径存在差异，把上面两段全文对照一下")
print("\n[非流式答案前 120 字]")
print(answer_sync[:120])
