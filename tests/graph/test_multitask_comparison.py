"""Paired pre-interrupt comparison; fixed routing oracle, no model/network API calls.

Run: python -m pytest tests/graph/test_multitask_comparison.py -q
This is a constructed structural probe, not a blind semantic/end-to-end benchmark.
"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.db import repository, workflow
from app.graph import nodes, runtime
from app.graph.build import build_graph

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT / "data/issues/implementation-20261103/before/app/graph"
OUT = ROOT / "data/issues/implementation-20261103/comparison"
POSITIVE = ["我的猫砂盆漏电了", "猫砂盆外壳让我触电了", "设备正在冒烟", "猫砂盆起火了",
            "我家的机器漏电", "刚刚碰到猫砂盆时触电", "插上电就冒烟", "机器已经起火",
            "碰到猫砂盆外壳手被电麻了", "设备外壳带电，非常危险"]
NEGATIVE = ["我的猫砂盆没有漏电", "机器并未冒烟", "设备不是漏电", "没有发生触电",
            "假设设备漏电", "如果猫砂盆起火", "听说别人的机器漏电", "他说设备会冒烟",
            "设备未发生漏电", "设备不存在漏电现象"]
CASES = [{"id": f"P{i+1}", "query": text + "，我要退款，还要投诉商家。", "urgent_safety": True}
         for i, text in enumerate(POSITIVE)] + [
         {"id": f"N{i+1}", "query": text + "，我要退款，还要投诉商家。", "urgent_safety": False}
         for i, text in enumerate(NEGATIVE)]
RESULTS = []


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def save_report():
    RESULTS.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps(CASES, ensure_ascii=False, indent=2), encoding="utf-8")
    yield
    summary = {}
    for arm in ("A_before", "B_new_disabled", "C_new_enabled"):
        rows = [r for r in RESULTS if r["arm"] == arm]
        pos = [r for r in rows if r["urgent_safety"]]
        neg = [r for r in rows if not r["urgent_safety"]]
        summary[arm] = {"cases": len(rows), "urgent_n": len(pos), "nonurgent_n": len(neg),
                        "safety_before_interrupt_n": sum(r["early_safety"] for r in pos),
                        "unnecessary_early_safety_n": sum(r["early_safety"] for r in neg),
                        "missed_ids": [r["id"] for r in pos if not r["early_safety"]],
                        "false_positive_ids": [r["id"] for r in neg if r["early_safety"]]}
    report = {"scope": "constructed pre-interrupt structural probe, not full business or model evaluation",
              "routing_oracle": "same refund main intent forced in all arms; no model calls",
              "note": "No benchmark latency claim; no generated answer, final task coverage or hallucination evaluation.",
              "hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (BEFORE/'build.py', BEFORE/'runtime.py', ROOT/'app/graph/build.py',
                                   ROOT/'app/graph/runtime.py', ROOT/'app/graph/tasks.py')},
              "summary": summary, "rows": RESULTS}
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
async def test_paired_preinterrupt_probe(case, db_session_factory, _test_engine, monkeypatch, tmp_path):
    assert (ROOT/'app/graph/nodes.py').read_bytes() == (BEFORE/'nodes.py').read_bytes()
    monkeypatch.setattr(workflow.base, "engine", _test_engine)
    async with _test_engine.begin() as conn:
        await conn.run_sync(lambda c: workflow.WorkflowRecord.__table__.drop(c, checkfirst=True))
    await workflow.ensure_tables()
    async def coref(query, *args, **kwargs):
        return query
    async def intent(*args, **kwargs):
        return {"intent": "退款退货", "confidence": 1.0}
    monkeypatch.setattr(nodes.coref, "resolve", coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", intent)
    from app.graph import tasks
    async def fixed_plan(raw):
        return [{'kind':k,'query':raw,'source':raw,'order_id':None,'order_reference':''} for k in ('refund','complaint')]
    monkeypatch.setattr(tasks.task_planner,'extract',fixed_plan)
    monkeypatch.setattr(settings, "context_policy", "legacy")
    monkeypatch.setattr(settings, "context_legacy_recall_enabled", False)
    old_builder = load("comparison_old_build", BEFORE/'build.py')
    old_runtime = load("comparison_old_runtime", BEFORE/'runtime.py')
    arms = [("A_before", old_builder.build_graph, old_runtime, False),
            ("B_new_disabled", build_graph, runtime, False),
            ("C_new_enabled", build_graph, runtime, True)]
    shift = int(case['id'][1:]) % 3
    for arm, builder, runner, enabled in arms[shift:] + arms[:shift]:
        monkeypatch.setattr(settings, "multi_intent_enabled", enabled)
        cid = await repository.create_conversation("comparison")
        mid = await repository.append_message(cid, "user", content=case["query"])
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path/(arm+'.sqlite'))) as saver:
            graph = builder(saver)
            monkeypatch.setattr(runner, "_graph", graph)
            started = time.perf_counter()
            events = [e async for e in runner._stream_events(cid, runner._graph_input(
                "comparison", case['query'], cid, mid, "", 0))]
            elapsed = time.perf_counter() - started
        assert events[-1]["type"] == "interrupt" and events[-1]["kind"] == "select_order"
        early = any(e.get('type') == 'task_event' and e.get('task_id', '').endswith(':safety')
                    for e in events[:-1])
        RESULTS.append({**case, "arm": arm, "early_safety": early,
                        "elapsed_seconds_diagnostic": elapsed,
                        "events": [e['type'] for e in events]})
