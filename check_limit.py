"""验证 num_predict 是否生效：对比「裸调 API」与「走 get_chat_llm()」。"""
import time

from core.llm import get_chat_llm

print("=== 走 get_chat_llm()（会带上 num_predict）===")
llm = get_chat_llm()
t0 = time.time()
r = llm.invoke("什么是过拟合")
elapsed = time.time() - t0
print(f"耗时  : {elapsed:.1f}s")
print(f"输出  : {len(r.content)} 字")
print(f"内容  : {r.content[:100]}...")

if elapsed < 25:
    print("\n>> num_predict 生效了（输出被压住，耗时明显下降）")
else:
    print("\n>> 仍然是 30 秒级 —— num_predict 没传进去，检查参数名")
