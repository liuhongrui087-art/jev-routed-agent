"""测两种场景下 LLM 调用的稳定性，看 embed 的介入是否拖慢 chat。"""
import time

from core.llm import get_chat_llm, get_embed_model

llm = get_chat_llm()
embed = get_embed_model()

print("=== A. 连续 5 次纯 chat ===")
for i in range(5):
    t0 = time.time()
    llm.invoke("什么是过拟合")
    print(f"  {i + 1}: {time.time() - t0:.1f}s")

print("\n=== B. chat 与 embed 交替 5 轮 ===")
for i in range(5):
    t0 = time.time()
    llm.invoke("什么是过拟合")
    t_chat = time.time() - t0

    t0 = time.time()
    embed.embed_query("过拟合")
    t_emb = time.time() - t0

    print(f"  {i + 1}: chat {t_chat:.1f}s  /  embed {t_emb:.1f}s")
