"""看检索片段的分数、长度与切块边界质量。"""
from collections import Counter

from langchain_chroma import Chroma

import config
from core.llm import get_embed_model

vs = Chroma(
    persist_directory=config.CHROMA_DIR,
    embedding_function=get_embed_model(),
)

print("===== 检索结果 =====")
results = vs.similarity_search_with_relevance_scores("什么是过拟合", k=3)
for i, (doc, score) in enumerate(results, 1):
    text = doc.page_content
    print(f"--- 片段 {i}  分数={score:.4f}  长度={len(text)} 字  第 {doc.metadata.get('page')} 页 ---")
    print(f"  开头: {text[:50]}")
    print(f"  结尾: {text[-50:]}")
    print()

print("===== 整库统计 =====")
data = vs.get()
texts = [t for t in data["documents"] if t.strip()]
lens = [len(t) for t in texts]
print(f"总块数 : {len(lens)}")
print(f"长度   : 最长 {max(lens)} / 最短 {min(lens)} / 平均 {sum(lens) // len(lens)}")

starts = Counter(t.strip()[0] for t in texts)
total = sum(starts.values())
print("\n开头字符分布（keep_separator 把分隔符放在下一块开头）：")
for ch, n in starts.most_common(8):
    print(f"  {ch!r:>8}  {n:>4} 个  ({n * 100 // total}%)")
