"""一次性构建知识库向量索引（耗时操作，独立运行）。

用法：
    D:\agent_remake\venv\Scripts\python.exe D:\agent_remake\scripts\build_index.py
"""
import os
import sys

# 把项目根目录插进模块搜索路径，这样从 scripts/ 里也能 import core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag import build_index

if __name__ == "__main__":
    n = build_index()
    print(f"索引完成，共 {n} 段")
