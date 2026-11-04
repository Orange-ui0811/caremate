"""ch09 飞轮批处理:扫问题池未归并条目 → 标准化+查重 → 待审队列。
运行:make flywheel(需 mysql + 聊天上游在线)。定时跑给 cron 示例:
  */30 * * * * cd /path/to/mewhelp && make flywheel >> log/flywheel.log 2>&1
"""
import asyncio
import argparse
import json
from pathlib import Path
import sys

from app.core.flywheel import process_pending


async def main(pool_file: Path | None = None):
    input_texts = None
    if pool_file:
        from scripts.ch10.corpus_lib import desensitize
        rows = [json.loads(line) for line in pool_file.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        input_texts = {int(row["question_id"]): desensitize(row["text"]) for row in rows}
    stats = await process_pending(limit=200, input_texts=input_texts)
    print(f"本轮处理 {stats['processed']} 条:新建缺口 {stats['created']},"
          f"归并 {stats['merged']},跳过待重试 {stats['skipped']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-file", type=Path, help="仅处理固定快照中的问题 ID 和脱敏文本")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.pool_file)) or 0)
