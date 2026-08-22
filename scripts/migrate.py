"""
migrate.py - 数据库迁移脚本

运行后自动建表，已存在的表不会重复创建。
"""
import asyncio
import sys
from pathlib import Path

# 确保项目在 Python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.db.migrations import run_migrations

if __name__ == "__main__":
    asyncio.run(run_migrations())
