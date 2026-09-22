"""抽查 PDF 的文本提取质量，或按关键词定位页码。

用法：
    D:\agent_remake\venv\Scripts\python.exe D:\agent_remake\scripts\peek_pdf.py <pdf路径>
    D:\agent_remake\venv\Scripts\python.exe D:\agent_remake\scripts\peek_pdf.py <pdf路径> --find 过拟合
"""
import sys

from pypdf import PdfReader


def show_pages(reader, indexes, limit=600):
    for i in indexes:
        text = reader.pages[i].extract_text() or ""
        print(f"===== 第 {i + 1} 页（前 {limit} 字）=====")
        print(text[:limit])
        print()


def main():
    if len(sys.argv) < 2:
        print("用法: peek_pdf.py <pdf路径> [--find 关键词]")
        return

    path = sys.argv[1]
    reader = PdfReader(path)
    print(f"文件   : {path}")
    print(f"总页数 : {len(reader.pages)}\n")

    if "--find" in sys.argv:
        keyword = sys.argv[sys.argv.index("--find") + 1]
        hits = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if keyword in text:
                hits.append((i, text.count(keyword)))
        print(f'关键词 "{keyword}" 命中 {len(hits)} 页：')
        for i, n in hits[:10]:
            print(f"  第 {i + 1} 页  ×{n}")
        if hits:
            print()
            show_pages(reader, [hits[0][0]])
        return

    show_pages(reader, range(min(2, len(reader.pages))))


if __name__ == "__main__":
    main()
