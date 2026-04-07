"""
Export conversation turns + embeddings to JSONL or Parquet for training pipelines.

Usage:
  set DATABASE_URL=...
  python scripts/export_training_data.py --out data/train.jsonl
  python scripts/export_training_data.py --out data/train.parquet --format parquet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg


async def export_jsonl(conn: asyncpg.Connection, out_path: Path) -> None:
    rows = await conn.fetch(
        """
        SELECT
            ct.id AS turn_id,
            s.external_session_id,
            ct.user_text,
            ct.assistant_text,
            ct.created_at,
            e.embedding_model,
            e.embedding::text AS embedding
        FROM conversation_turns ct
        JOIN sessions s ON s.id = ct.session_id
        LEFT JOIN embeddings e ON e.turn_id = ct.id
        ORDER BY ct.created_at ASC
        """
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            d = dict(r)
            f.write(json.dumps(d, default=str, ensure_ascii=False) + "\n")


async def export_parquet(conn: asyncpg.Connection, out_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("parquet export requires pyarrow: pip install pyarrow") from e

    rows = await conn.fetch(
        """
        SELECT
            ct.id AS turn_id,
            s.external_session_id,
            ct.user_text,
            ct.assistant_text,
            ct.created_at,
            e.embedding_model,
            e.embedding::text AS embedding
        FROM conversation_turns ct
        JOIN sessions s ON s.id = ct.session_id
        LEFT JOIN embeddings e ON e.turn_id = ct.id
        ORDER BY ct.created_at ASC
        """
    )
    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pa.table({})
        pq.write_table(empty, out_path)
        return

    data = [dict(r) for r in rows]
    # Flatten for Arrow (embedding as string; parse downstream)
    table = pa.Table.from_pylist(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


async def main_async() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output file path")
    p.add_argument(
        "--format",
        choices=("jsonl", "parquet"),
        default="jsonl",
    )
    args = p.parse_args()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is required", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.out)
    conn = await asyncpg.connect(dsn)
    try:
        if args.format == "jsonl":
            await export_jsonl(conn, out_path)
        else:
            await export_parquet(conn, out_path)
    finally:
        await conn.close()
    print(f"Wrote {out_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
