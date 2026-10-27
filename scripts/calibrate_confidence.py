"""ch09 置信度阈值校准:拿 ch04 评估集,A/B/C 桶(应可答)与 D 桶(应拒答)分别算
evidence_confidence 分布,扫阈值取 Youden J(=可答通过率 - 应拒放行率)最大的分离点。
输出分布表 + 推荐阈值 → 在管理端「观测与成本」应用推荐值。
运行:make calibrate-confidence(需 milvus + 上游可调通 + 知识库已建)。
产物:data/ch09/reports/confidence_calibration.{txt,json};「观测与成本」页只读这份 json。
"""
import asyncio
import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime

from app.config import settings
from app.core import model_pool, read_notes, retrieval
from app.core.confidence import SCORING_METHOD, compute_evidence_confidence

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data/ch09/reports"
_OUT = _OUT_DIR / "confidence_calibration.txt"
_OUT_JSON = _OUT_DIR / "confidence_calibration.json"
ANSWERABLE = {"A_policy", "B_model", "C_colloquial"}
_DATASET = _ROOT / "tests/data/eval_ch04.jsonl"
_BUSY_ATTEMPTS = 5
_LINES: list[str] = []


def _log(msg=""):
    print(msg, flush=True)
    _LINES.append(msg)


async def _conf(sample) -> tuple[str, float]:
    # 后台默认仅一个槽位。串行提交整题，避免八题互相排队超过模型池的 3 秒期限。
    # SDK 会把 PoolBusy 包成 APIConnectionError；只对确认为排队拥塞的异常重试。
    with model_pool.call_context(priority="background"):
        for attempt in range(_BUSY_ATTEMPTS):
            try:
                hits = await retrieval.search_knowledge(sample["query"], strategy="hybrid_rerank")
                return sample["bucket"], compute_evidence_confidence(hits).score
            except Exception as exc:
                cause, seen = exc, set()
                while cause is not None and id(cause) not in seen:
                    seen.add(id(cause))
                    if isinstance(cause, model_pool.PoolBusy):
                        break
                    cause = cause.__cause__ or cause.__context__
                else:
                    raise
                if attempt + 1 == _BUSY_ATTEMPTS:
                    raise RuntimeError("后台模型池持续繁忙，校准未完成；旧报告已保留。"
                                       "请等待其他后台任务结束后重跑。") from exc
                delay = 2 ** attempt
                _log(f"模型池繁忙，{delay} 秒后重试本题（{attempt + 1}/{_BUSY_ATTEMPTS - 1}）")
                await asyncio.sleep(delay)


def _write_atomic(path, text):
    """读者始终看到完整报告；不能在前端刷新时读到写了一半的 JSON。"""
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.stem + "-", suffix=".tmp", delete=False) as f:
            temp = pathlib.Path(f.name)
            f.write(text)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _dist(name, xs) -> dict | None:
    """打印一行分布,同时把这行的数原样返回给 json 产物——两边同源,不各算一遍。"""
    xs = sorted(xs)
    if not xs:
        return None
    def p(q):
        return xs[min(len(xs) - 1, int(q * len(xs)))]
    stat = {"n": len(xs), "min": round(xs[0], 3), "p25": round(p(.25), 3),
            "p50": round(p(.5), 3), "p75": round(p(.75), 3), "max": round(xs[-1], 3)}
    _log(f"{name:12s} n={stat['n']:3d} min={stat['min']:.3f} p25={stat['p25']:.3f} "
         f"p50={stat['p50']:.3f} p75={stat['p75']:.3f} max={stat['max']:.3f}")
    return stat


async def main():
    _LINES.clear()
    if settings.model_pool_enabled and model_pool.get_store().policy().background_limit == 0:
        raise RuntimeError("后台模型调用已暂停（后台并发为 0），请先在模型池设置中启用后台槽位。")
    samples = [json.loads(ln) for ln in _DATASET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # E 桶不参与当前校准口径；不为不用的分数占用后台模型槽位。
    samples = [s for s in samples if s["bucket"] in ANSWERABLE | {"D_absent"}]
    if not any(s["bucket"] in ANSWERABLE for s in samples) or not any(s["bucket"] == "D_absent" for s in samples):
        raise ValueError("校准集必须同时包含可答 ABC 桶和应拒 D 桶；旧报告已保留。")
    _log(f"开始置信度校准：{len(samples)} 题，后台串行执行；全部成功后才更新报告。")
    results = []
    for index, sample in enumerate(samples, 1):
        results.append(await _conf(sample))
        _log(f"进度 {index}/{len(samples)} · {sample['bucket']}")
    answerable = [c for b, c in results if b in ANSWERABLE]
    absent = [c for b, c in results if b == "D_absent"]

    _log("=== evidence_confidence 分布(hybrid_rerank)===")
    dist = {"answerable": _dist("可答(ABC)", answerable), "absent": _dist("应拒(D)", absent)}

    _log("\n=== 阈值扫描(通过率=conf>=t 的可答占比;放行率=conf>=t 的应拒占比)===")
    _log(f"{'t':>6s} {'可答通过率':>10s} {'应拒放行率':>10s} {'YoudenJ':>8s}")
    scan, best_t, best_j = [], 0.0, -1.0
    for i in range(5, 96):
        t = i / 100
        tpr = sum(1 for c in answerable if c >= t) / len(answerable)
        fpr = sum(1 for c in absent if c >= t) / len(absent)
        j = tpr - fpr
        scan.append({"t": round(t, 2), "pass_rate": round(tpr, 3),
                     "leak_rate": round(fpr, 3), "youden_j": round(j, 3)})
        if i % 5 == 0 or j > best_j:
            _log(f"{t:6.2f} {tpr:10.3f} {fpr:10.3f} {j:8.3f}")
        if j > best_j:
            best_t, best_j = t, j
    _log(f"\n推荐阈值 evidence_confidence_threshold = {best_t:.2f}(Youden J={best_j:.3f})")
    _log("可在管理端查看本轮结果并应用推荐值；本次校准保留当前在用阈值。")

    pick = next((r for r in scan if r["t"] == round(best_t, 2)), None)
    recommended = {"threshold": round(best_t, 2), "youden_j": round(best_j, 3),
                   "pass_rate": pick["pass_rate"] if pick else None,
                   "leak_rate": pick["leak_rate"] if pick else None,
                   "selection_rule": "youden_j",
                   "selection_rule_label": "Youden J 最大：可答通过率减应拒放行率"}
    # 读图小注只喂十分位那几条扫描线:整条 91 点曲线塞进去,模型容易挑个无关的点来说事。
    # 字段名一律换成页面上那几个中文表头,注里就不会出现 pass_rate 这种说法
    def _zh(stat):
        return None if not stat else {"题数": stat["n"], "最低": stat["min"], "p25": stat["p25"],
                                      "中位": stat["p50"], "p75": stat["p75"], "最高": stat["max"]}
    note = await read_notes.generate("confidence_calibration", {
        "证据置信度分布": {"可答(政策/型号/口语)": _zh(dist["answerable"]),
                          "应拒(库外)": _zh(dist["absent"])},
        "选定的线": {"阈值": recommended["threshold"], "可答通过率": recommended["pass_rate"],
                    "应拒放行率": recommended["leak_rate"], "YoudenJ": recommended["youden_j"]},
        "阈值扫描(每 0.1 一条)": [
            {"阈值": r["t"], "可答通过率": r["pass_rate"], "应拒放行率": r["leak_rate"]}
            for r in scan if round(r["t"] * 10, 3) == int(r["t"] * 10)]})
    _log("\n读图小注:" + (note if note else "本轮没有(页面用兜底句)"))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_atomic(_OUT, "\n".join(_LINES) + "\n")
    _write_atomic(_OUT_JSON, json.dumps({
        "meta": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "strategy": "hybrid_rerank", "scoring_method": SCORING_METHOD,
                 "source": "live_retrieval",
                 "source_label": "当前检索接口实跑；ABC 可答题与 D 应拒题参与校准，E 桶不参与。",
                 "dataset": "tests/data/eval_ch04.jsonl",
                 "n_answerable": len(answerable), "n_absent": len(absent)},
        "distribution": dist,
        # 保留未四舍五入的分数，支持管理端任意精度的在用阈值（含 0 与 1）。
        "scores": {"answerable": answerable, "absent": absent},
        "scan": scan,
        "recommended": recommended,
        "read_notes": {"confidence_calibration": note} if note else {},
    }, ensure_ascii=False, indent=1) + "\n")
    _log(f"报告已落 {_OUT} 与 {_OUT_JSON}")


if __name__ == "__main__":
    with model_pool.call_context(priority="background"):
        sys.exit(asyncio.run(main()) or 0)
