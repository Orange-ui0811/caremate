"""ch10 语料流水线:捞池 → 脱敏去重 → LLM 修错别字 → LLM 预标 → 模拟补足 → 导出人工抽审。
运行:make ch10-corpus(需 mysql + 聊天上游)。产物落 data/ch10/,抽审文件给用户对话里审。"""
import asyncio
import argparse
import json
import hashlib
import pathlib
import random
from collections.abc import Callable

from pydantic import BaseModel

from app.core.llm import get_chat_model
from app.core.taxonomy import LABEL2ID, TOPIC_CLASSES, terminology_table
from app.db import repository
from scripts.ch10.corpus_lib import dedupe, desensitize
from scripts.ch10.prelabel import prelabel_batch

OUT = pathlib.Path("data/ch10")
TARGET_PER_CLASS = 100
SIM_BATCH = 20     # 单次造数条数,小批多次控质量
CALL_TIMEOUT = 120.0  # 离线造数有较长输出,单批超时后重试;不改变客服模型配置

CLEAN_PROMPT = """修正下面这句用户问题里的错别字和乱格式:不改语义、不改口语风格、不增删诉求,
没有错误就原样返回。只输出句子本身。

{text}"""


class _SimItem(BaseModel):
    text: str
    labels: list[str]


class _SimBatch(BaseModel):
    items: list[_SimItem]


SIMULATE_PROMPT = """你是电商客服语料造数员。为猫用品电商「喵喵优选」(卖猫粮、冻干、猫零食、猫砂盆、\
猫抓板、猫窝、猫爬架、猫碗、逗猫棒、项圈等)生成 {n} 条模拟用户问题,全部命中主题类目「{name}」。

17 类权威术语表:
{terminology}

要求:
1. 每条一句独立、语义完整的口语化用户问题,长短语气多样,贴近真实客服提问,彼此不重复。
2. 大约 15% 用方言土话(如「俺买的那玩意儿咋还没到俺这疙瘩」),10% 故意带错别字(如「退活」=退货),\
15% 是多诉求句——除「{name}」外再字面提到一个其他类目的诉求,labels 把两个类都打上。
3. 其余条目只含「{name}」一个诉求,labels 只有它。
4. 标签铁律:字面提到几个诉求打几个标签,一个不多一个不少;标签取术语表类目名原文。"""


def _dump(path: pathlib.Path, samples: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
                    encoding="utf-8")


async def clean_texts(texts: list[str], concurrency: int = 8) -> list[str]:
    model = get_chat_model()
    sem = asyncio.Semaphore(concurrency)

    async def one(t: str) -> str:
        async with sem:
            try:
                r = await model.ainvoke(CLEAN_PROMPT.format(text=t))
                out = r.content.strip()
                return out or t
            except Exception as e:
                print(f"[clean] 调用失败,原样保留: {t[:30]}… ({type(e).__name__})")
                return t

    return list(await asyncio.gather(*[one(t) for t in texts]))


async def simulate(name: str, need: int, on_batch: Callable[[list[dict]], None] | None = None) -> list[dict]:
    model = get_chat_model().with_structured_output(_SimBatch)
    out: list[dict] = []
    misses = 0
    while len(out) < need and misses < 5:
        n = min(SIM_BATCH, need - len(out))
        try:
            r: _SimBatch = await asyncio.wait_for(model.ainvoke(SIMULATE_PROMPT.format(
                n=n, name=name, terminology=terminology_table())), timeout=CALL_TIMEOUT)
        except Exception as e:
            print(f"[simulate] {name} 一批失败({type(e).__name__}),重试(misses={misses + 1})")
            misses += 1
            continue
        got = 0
        for it in r.items:
            labels = [lb for lb in it.labels if lb in LABEL2ID]
            if name not in labels:
                labels = [name] + labels
            if it.text.strip():
                out.append({"text": it.text.strip(), "labels": labels, "origin": "simulated"})
                got += 1
        misses = misses + 1 if got == 0 else 0
        if got:
            if on_batch:
                on_batch(out[:need])
            print(f"[simulate] {name} {min(len(out), need)}/{need}", flush=True)
    return out[:need]


async def main(pool_file: pathlib.Path | None = None, resume: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # 1) 捞池(标准化问法优先)
    pool = ([json.loads(line) for line in pool_file.read_text(encoding="utf-8").splitlines()
             if line.strip()] if pool_file else await repository.list_pool_texts())
    raw = [{"text": p["text"], "labels": [], "origin": "pool"} for p in pool]
    _dump(OUT / "corpus_raw.jsonl", raw)
    print(f"捞池 {len(raw)} 条")
    fingerprint = hashlib.sha256(json.dumps(raw, ensure_ascii=False).encode()).hexdigest()
    checkpoint = OUT / "corpus_checkpoint.json"

    def save_checkpoint(samples):
        temp = checkpoint.with_suffix('.tmp')
        temp.write_text(json.dumps({"input_sha256": fingerprint, "samples": samples},
                                   ensure_ascii=False), encoding="utf-8")
        temp.replace(checkpoint)

    if resume and checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["input_sha256"] != fingerprint:
            raise ValueError("问题快照已改变,不能复用上次检查点;请核对输入范围")
        labeled = state["samples"]
        print(f"检查点恢复 {len(labeled)} 条")
    else:
        # 2) 清洗:脱敏 → 去重 → LLM 修错别字 → 再去重
        cleaned = dedupe([{**s, "text": desensitize(s["text"])} for s in raw])
        fixed = await clean_texts([s["text"] for s in cleaned])
        cleaned = dedupe([{**s, "text": t} for s, t in zip(cleaned, fixed)])
        _dump(OUT / "corpus_clean.jsonl", cleaned)
        print(f"清洗后 {len(cleaned)} 条")
        # 3) 预标真实问题
        labels = await prelabel_batch([s["text"] for s in cleaned])
        labeled = [{**s, "labels": lb} for s, lb in zip(cleaned, labels)]
        save_checkpoint(labeled)
    # 4) 模拟补足:按标签计数补到每类 TARGET_PER_CLASS(多诉求句给命中的每类都记数)
    counts = {c.name: 0 for c in TOPIC_CLASSES}
    for s in labeled:
        for lb in s["labels"]:
            counts[lb] += 1
    for c in TOPIC_CLASSES:
        need = TARGET_PER_CLASS - counts[c.name]
        if need <= 0:
            continue
        sims = await simulate(c.name, need, on_batch=lambda batch: save_checkpoint(labeled + batch))
        labeled.extend(sims)
        for s in sims:
            for lb in s["labels"]:
                counts[lb] += 1
        print(f"{c.name}: 补造 {len(sims)} 条(当前 {counts[c.name]})")
        save_checkpoint(labeled)
        if len(sims) < need:
            raise RuntimeError(f"{c.name} 上游连续失败,检查点已保存;加 --resume 重试")
    labeled = dedupe(labeled)
    _dump(OUT / "corpus_labeled.jsonl", labeled)
    print(f"语料总量 {len(labeled)} 条;各类:{counts}")
    # 5) 抽审导出:真实池全量 + 每类模拟抽 5
    rng = random.Random(42)
    lines = ["# ch10 语料人工抽审(预标 + 模拟)", "",
             "> 格式:问题 → 标签。看到错标直接指出原句。", "",
             "## 真实池问题(全量)", ""]
    for s in labeled:
        if s["origin"] == "pool":
            lines.append(f"- {s['text']} → {'、'.join(s['labels'])}")
    lines += ["", "## 模拟问题(每类抽 5)", ""]
    for c in TOPIC_CLASSES:
        sims = [s for s in labeled if s["origin"] == "simulated" and c.name in s["labels"]]
        lines.append(f"### {c.name}")
        lines += [f"- {s['text']} → {'、'.join(s['labels'])}"
                  for s in rng.sample(sims, min(5, len(sims)))]
        lines.append("")
    (OUT / "sample_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"抽审文件已导出:{OUT / 'sample_review.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-file", type=pathlib.Path,
                        help="使用明确授权并已脱敏的固定问题快照,不读取当前会话池")
    parser.add_argument("--resume", action="store_true", help="从相同问题快照的检查点继续")
    args = parser.parse_args()
    asyncio.run(main(args.pool_file, args.resume))
