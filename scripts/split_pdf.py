"""把大 PDF 按顶层书签（章节）拆分成多个小 PDF。

用法：
    D:\agent_remake\venv\Scripts\python.exe D:\agent_remake\scripts\split_pdf.py

输出到 knowledge/parts/ 子目录 —— 放在子目录里，build_index 扫顶层时就不会碰到它们，
你可以按需把某一章拷到 knowledge/ 根目录去做索引。
"""
import os
import re
import sys

from pypdf import PdfReader, PdfWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SRC_NAME = "d2l-zh.pdf"
SRC = os.path.join(config.KNOWLEDGE_DIR, SRC_NAME)
OUT_DIR = os.path.join(config.KNOWLEDGE_DIR, "parts")


def safe_filename(text: str, max_len: int = 36) -> str:
    """把章节标题变成合法的文件名。"""
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", text.strip())
    return text[:max_len].strip("_") or "section"


def collect_chapters(reader):
    """取出顶层书签：(标题, 起始页索引)。跳过子节点。"""
    chapters = []
    for node in reader.outline:
        if isinstance(node, list):
            continue
        try:
            start = reader.get_destination_page_number(node)
        except Exception:
            continue
        chapters.append((node.title, start))
    return chapters


def main():
    if not os.path.isfile(SRC):
        print(f"[split] 找不到源文件: {SRC}")
        return

    reader = PdfReader(SRC)
    total = len(reader.pages)
    print(f"[split] 源文件   : {SRC_NAME}")
    print(f"[split] 总页数   : {total}")
    print(f"[split] 书签条目 : {len(reader.outline)}")

    chapters = collect_chapters(reader)
    print(f"[split] 顶层章节 : {len(chapters)}")
    for i, (title, start) in enumerate(chapters):
        print(f"        {i:>2}. p{start + 1:>4}  {title}")

    if len(chapters) < 2:
        print("[split] 顶层书签不足，无法按章节拆分")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n[split] 输出目录 : {OUT_DIR}\n")

    written = 0
    for i, (title, start) in enumerate(chapters):
        end = chapters[i + 1][1] if i + 1 < len(chapters) else total
        if end <= start:
            continue

        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])

        fname = f"{i:02d}_{safe_filename(title)}.pdf"
        with open(os.path.join(OUT_DIR, fname), "wb") as f:
            writer.write(f)

        print(f"[split] 写出 {fname}   ({end - start} 页)")
        written += 1

    print(f"\n[split] 完成，共写出 {written} 个文件")


if __name__ == "__main__":
    main()
