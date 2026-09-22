"""看清 pdf_files / pages / docs 三层到底装了什么。"""
import os

import config
from langchain_community.document_loaders import PyPDFLoader

pdf_files = [f for f in os.listdir(config.KNOWLEDGE_DIR) if f.endswith(".pdf")]
print("第1层 pdf_files =", pdf_files, "| 长度 =", len(pdf_files))
print("  pdf_files[0] 的类型 =", type(pdf_files[0]).__name__)

pages = PyPDFLoader(os.path.join(config.KNOWLEDGE_DIR, pdf_files[0])).load()
print("\n第2层 pages | 长度 =", len(pages), "（这就是页数）")
print("  pages[0] 类型 =", type(pages[0]).__name__)
print("  pages[0] 正文前 40 字 =", pages[0].page_content[:40])
print("  pages[1] 正文前 40 字 =", pages[1].page_content[:40])

docs = []
docs.extend(pages)
print("\n第3层 docs | 长度 =", len(docs))
print("  docs[0] is pages[0] ?", docs[0] is pages[0])
