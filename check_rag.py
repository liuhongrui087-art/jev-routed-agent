"""验证 RAG 检索质量。看召回片段里有没有真正相关的内容。"""
from core.rag import get_retriever

retriever = get_retriever()
print("retriever =", retriever)

if retriever is None:
    print(">> 索引不存在，先去跑 scripts/build_index.py")
else:
    docs = retriever.invoke("什么是过拟合")
    print(f"\n检索到 {len(docs)} 个片段\n")
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        print(f"--- 片段 {i}（第 {page} 页）---")
        print(d.page_content[:200])
        print()
