"""RAG：知识库加载与检索。

设计要点：
- 索引持久化到 .chroma/，构建一次反复用，绝不在 Web 请求里做；
- get_retriever() 懒加载：索引不存在就返回 None，不拖慢服务启动；
- 建索引入口是 scripts/build_index.py，单独执行。
"""
import os
import config

_retriever = None
import re


def _clean_pdf_text(text: str) -> str:
    """清理 PDF 提取出的文本。

    PDF 的换行是"视觉换行"（一行排满就换），不是语义边界，
    一句话常被拆成好几行。不处理的话，切块会在词中间断开。
    """
    text = re.sub(r"\n{3,}", "\n\n", text)          # 3 个以上换行 → 段落分隔
    text = re.sub(r"(?<!\n)\n(?!\n)", "", text)     # 删掉"行内换行"
    text = re.sub(r"[ \t]{2,}", " ", text)          # 多余空格压缩
    return text


def build_index() -> int:
    """扫描 knowledge/ 下的 PDF，切块、向量化、持久化。返回切块数。"""
    global _retriever
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyMuPDFLoader   # ← 换掉 PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from core.llm import get_embed_model

    knowledge_dir = config.KNOWLEDGE_DIR
    if not os.path.isdir(knowledge_dir):
        os.makedirs(knowledge_dir)
        print(f"[rag] 已创建 {knowledge_dir}，请把 PDF 放进去")
        return 0

    pdf_files = [f for f in os.listdir(knowledge_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"[rag] {knowledge_dir} 中没有 PDF，RAG 功能禁用")
        return 0

    pages = []
    for name in pdf_files:
        print(f"[rag] 开始解析 {name}")
        loader = PyMuPDFLoader(os.path.join(knowledge_dir, name))
        for i, page in enumerate(loader.lazy_load(), 1):
            page.page_content = _clean_pdf_text(page.page_content)  # ← 新增：清洗
            pages.append(page)
            if i % 50 == 0:
                print(f"[rag]   已解析 {i} 页")
        print(f"[rag] {name} 解析完成，累计 {len(pages)} 页")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "。", "！", "？", "；", "\n", "，", " ", ""],  # ← 顺序调整
    )
    chunks = splitter.split_documents(pages)

    before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 50]  # ← 新增：过滤碎片
    print(f"[rag] 切块后 {before} 段，过滤掉 {before - len(chunks)} 个碎片，保留 {len(chunks)} 段")

    print(f"[rag] 开始向量化（最耗时的一步，{len(chunks)} 段，请耐心等待）...")
    vectorstore = Chroma.from_documents(
        chunks, get_embed_model(), persist_directory=config.CHROMA_DIR
    )
    _retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVE_K})
    print(f"[rag] 索引已持久化到 {config.CHROMA_DIR}")
    return len(chunks)


def get_retriever():
    """三级查找：内存 → 持久化索引 → None。"""
    global _retriever
    if _retriever is not None:
        return _retriever
    if os.path.isdir(config.CHROMA_DIR) and os.listdir(config.CHROMA_DIR):
        from langchain_chroma import Chroma

        from core.llm import get_embed_model

        vectorstore = Chroma(
            persist_directory=config.CHROMA_DIR,
            embedding_function=get_embed_model(),
        )
        _retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVE_K})
    return _retriever
