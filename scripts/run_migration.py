"""
在已安装依赖的前提下，对 DATABASE_URL 指向的库执行 SQL 迁移文件（逐条 execute，无需 psql）。

用法（PowerShell）:
  $env:DATABASE_URL = "postgresql://用户:密码@主机:端口/库名"
  python scripts/run_migration.py migrations/001_init_no_pgvector.sql

或先创建 .env 写入 DATABASE_URL，再:
  python scripts/run_migration.py migrations/001_init_no_pgvector.sql
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

import asyncpg

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


def _split_sql(sql: str) -> list[str]:
    """按分号拆分语句，去掉单独成行或行尾的 SQL 注释。"""
    parts: list[str] = []
    buf: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        buf.append(line)
        if ";" in line and line.rstrip().endswith(";"):
            stmt = "\n".join(buf).strip()
            if stmt:
                stmt = stmt.rstrip(";").strip()
                stmt = re.sub(r"--[^\n]*", "", stmt).strip()
                if stmt:
                    parts.append(stmt)
            buf = []
    return parts


async def main_async(sql_path: Path) -> None:
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        print("请设置环境变量 DATABASE_URL，或在项目根目录配置 .env", file=sys.stderr)
        sys.exit(1)
    text = sql_path.read_text(encoding="utf-8")
    statements = _split_sql(text)
    if not statements:
        print(f"未解析到任何语句: {sql_path}", file=sys.stderr)
        sys.exit(1)

    conn = await asyncpg.connect(dsn)
    try:
        for i, stmt in enumerate(statements, 1):
            await conn.execute(stmt)
            print(f"ok ({i}/{len(statements)})")
    finally:
        await conn.close()
    print("迁移完成:", sql_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "sql_file",
        type=Path,
        nargs="?",
        default=Path("migrations/001_init_no_pgvector.sql"),
        help="SQL 文件路径（默认: migrations/001_init_no_pgvector.sql）",
    )
    args = p.parse_args()
    path = args.sql_file
    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main_async(path))


if __name__ == "__main__":
    main()
