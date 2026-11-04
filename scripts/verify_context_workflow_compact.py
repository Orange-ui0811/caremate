"""Frozen compact-context regression through the real customer-service workflow.

Only the context projection changes. The model, workflow, guards, business
fixtures, scoring, output ceiling and recovery-tool setting stay identical.
Default invocation prepares files only; --run explicitly makes model calls.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

from app.core.context_policy import estimate

_spec = importlib.util.spec_from_file_location(
    "scripts._compact_workflow_private", Path(__file__).with_name("verify_context_workflow.py"))
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
_BaseFixture = _impl.Fixture
_base_isolated, _base_audit = _impl.isolated_workflow, _impl.audit

VERSION = "workflow-compact-verification-v1"
ARMS = ("source", "compact_source")
CAPS = {"coref": 1800, "intent": 1000, "answer": 4200, "retrieval": 4200}
_impl.SOURCES += ["scripts/verify_context_workflow_compact.py", "app/core/context_compact.py",
                  "tests/test_context_workflow_compact.py"]


def dataset():
    value = deepcopy(_impl.dataset())
    value["version"] = "workflow-compact-synthetic-v1"
    for case in value["cases"]:
        case.update(stratum="existing_short_regression", initial_history=[])

    def step(query, kind, target=""):
        return {"query": query, "expected": {"kind": kind, "target": target}}

    def pressure(case_id, prefix, steps):
        rows = [{"role": role, "content": content} for role, content in prefix]
        # Deliberately uninformative, repeated packaging discussion. These are
        # generated history fixtures, not previously paid model completions.
        pair = [("user", "我还想了解一般的外包装保存注意事项。"),
                ("assistant", "外包装请放在干燥通风的位置，避免挤压、阳光直射和潮湿环境。"
                 "清理时先查看包装上的一般说明，保持标签完整并记录外观情况。"
                 "这段讨论只涉及外包装，不改变此前选定的商品、订单和操作限制。")]
        while sum(estimate(r["content"]) for r in rows) < 6500:
            rows.extend({"role": role, "content": content} for role, content in pair)
        return {"id": case_id, "stratum": "new_internal_budget_pressure", "initial_history": rows,
                "history_estimated_tokens": sum(estimate(r["content"]) for r in rows),
                "construction": "Repeated neutral packaging turns until >=6500 heuristic tokens; no gold is included in the prompt builder.",
                "steps": steps}

    value["cases"].extend([
        pressure("pressure_order_correction", [
            ("user", "现在处理订单710001，请记下这个订单。"),
            ("assistant", "已记录用户当前提及订单710001，尚未查询。"),
            ("user", "710001报错，应为710002，后续只处理更正后的订单。"),
            ("assistant", "已记录更正：本次订单是710002；这不表示旧订单失效。")], [
            step("按之前最后更正的那笔订单，查一下物流。", "logistics", "710002"),
            step("刚查的那笔再查询最新物流，不要提交申请。", "logistics", "710002"),
            step("现在明确改查订单710007，只查询它的物流。", "logistics", "710007")]),
        pressure("pressure_constraint_retraction", [
            ("user", "订单710003不要提交退款，也不要创建工单。"),
            ("assistant", "已记录订单710003的退款和工单限制。"),
            ("user", "订单710004也不要提交任何申请。"),
            ("assistant", "已分别记录两笔订单的限制。"),
            ("user", "只撤回订单710004的工单限制；订单710003的限制继续保留。现在处理订单710003。"),
            ("assistant", "已记录限制调整；这不是立即创建工单的确认。")], [
            step("只查当前那笔订单的物流，之前仍有效的限制不要改变。", "logistics", "710003"),
            step("改查订单710004的物流，这次仍然只做查询。", "logistics", "710004"),
            step("回到订单710003，再查物流，仍然不要提交退款或创建工单。", "logistics", "710003")]),
        pressure("pressure_earlier_product", [
            ("user", "办公室安装的是 MH-CAM1 摄像头，家里用的是 MH-LP100 猫砂盆。"),
            ("assistant", "已记录两个不同场所使用的商品型号。"),
            ("user", "先处理家里那款，暂时不问办公室的。"),
            ("assistant", "当前讨论家里的 MH-LP100 猫砂盆。")], [
            step("回到之前办公室安装的那款，它的镜头怎么清洁？", "care", "MH-CAM1"),
            step("现在明确查 MH-FD10 喂食器的食盆清洁方法。", "care", "MH-FD10"),
            step("刚查的这款，机身应该怎么清洁？", "care", "MH-FD10")]),
    ])
    value["limits"] = ["The six short cases have already been seen in earlier development.",
                       "The three pressure cases are internally authored synthetic additions, not an independent held-out set.",
                       "Preloaded histories are not generated during this run; their original generation cost is excluded."]
    return value


def prepare(out, arms=ARMS, case_ids=None, turn_timeout=180):
    out = Path(out).resolve()
    if (out / "manifest.json").exists():
        raise ValueError("existing manifest is immutable; use a new output directory")
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError("unknown arm")
    data = dataset()
    if case_ids:
        chosen = set(case_ids)
        data["cases"] = [case for case in data["cases"] if case["id"] in chosen]
        if {case["id"] for case in data["cases"]} != chosen:
            raise ValueError("unknown case selection")
    trigger = _impl.settings.context_summary_trigger_tokens
    for case in data["cases"]:
        estimated = sum(estimate(row["content"]) for row in case["initial_history"])
        # Reserve a declared 1200-token allowance per generated answer and all
        # three actual queries. Actual maintenance, if triggered, is still run
        # and charged; this assumption is never a truncation of model outputs.
        projected = estimated + sum(estimate(s["query"]) + 1200 for s in case["steps"])
        if case["initial_history"] and not 6500 <= estimated < projected < trigger:
            raise ValueError("pressure history is not below frozen summary trigger with declared allowance")
        case["predeclared_summary_check"] = {"history_estimate": estimated,
            "with_three_queries_and_1200_tokens_per_answer": projected, "trigger": trigger,
            "note": "Planning allowance only, not an answer limit; actual triggered maintenance is retained and charged."}
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in _impl.SOURCES:
        source, target = _impl.ROOT / name, out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    _impl.write(out / "dataset.json", data)
    manifest = {"created_at": _impl.stamp(), "version": VERSION, "arms": list(arms),
        "dataset_sha256": _impl.digest(data), "sources": sources, "config": _impl.config(),
        "compact_caps": CAPS, "thinking": _impl.OFF, "turn_timeout_seconds": turn_timeout,
        "scope": "real LangGraph and Gitee; synthetic in-memory DB, business and retrieval",
        "persistence_exception": "existing model pool may write technical SQLite call metadata; customer DB and business writes are isolated",
        "background_summary": "real scheduler with isolated segment store; a test-only completion barrier accounts for all maintenance before the next turn",
        "latency_limit": "end-to-end includes the test-only maintenance barrier if triggered; not production service latency",
        "recovery_tools": False, "group_order": "sequential", "frozen": True,
        "comparison": "source uses actual runtime stage caps; compact_source uses smaller stage caps with the same ownership/write guards and original sources",
        "retrieval_effective_cap": "The real retrieval node additionally passes context_coref_budget (currently 4000); compact retrieval is min(4200, caller budget), currently 4000, rather than an unconditional 4200.",
        "constraint_scope_limit": "pressure_constraint_retraction only performs logistics queries. It tests object selection and absence of write proposals, not correct write authorization after a retraction; write guards require their separate unit tests.",
        "token_scope": "all model calls for generated follow-ups and any triggered maintenance; excludes preloaded-history generation and real business/RAG services"}
    _impl.write(out / "manifest.json", manifest)
    return manifest


class Fixture(_BaseFixture):
    def __init__(self, case=None):
        super().__init__()
        self.case = case or {"id": "offline-fixture", "initial_history": []}
        self.rows = [{"id": i, "conversation_id": self.cid, **deepcopy(row)}
                     for i, row in enumerate(self.case["initial_history"], 1)]
        self.lease = None

    async def list_segments(self, cid, uid, *, upto_msg_id=None):
        self.check(cid, uid)
        return deepcopy([s for s in self.segments if upto_msg_id is None or s["upto_msg_id"] <= upto_msg_id])

    async def acquire_summary_lease(self, cid, uid, *, seconds=300):
        self.check(cid, uid)
        if self.lease is not None:
            return None
        self.lease = "synthetic-summary-lease"
        return self.lease

    async def release_summary_lease(self, cid, owner):
        self.check(cid)
        if self.lease == owner:
            self.lease = None

    async def put_summary_segment(self, cid, uid, **kwargs):
        self.check(cid, uid)
        rows = [r for r in self.rows if kwargs["from_msg_id"] <= r["id"] <= kwargs["upto_msg_id"]]
        if not rows or rows[0]["id"] != kwargs["from_msg_id"] or rows[-1]["id"] != kwargs["upto_msg_id"]:
            raise ValueError("summary source range missing")
        if _impl.context_repository.source_digest([SimpleNamespace(**r) for r in rows]) != kwargs["source_hash"]:
            raise ValueError("summary source hash changed")
        refs = [{"message_id": r["id"], "start": 0, "end": len(r["content"]),
                 "content_hash": hashlib.sha256(r["content"].encode()).hexdigest()} for r in rows]
        item = {"id": "fixture-summary-" + str(len(self.segments) + 1), "conversation_id": cid,
                **deepcopy(kwargs), "source_refs": refs, "created_at": _impl.stamp()}
        self.segments.append(item)
        return deepcopy(item)


@contextmanager
def isolated_workflow(fixture, arm, recorder):
    from app.core.context_compact import assemble_compact_context
    projections, maintenance = [], []
    with _base_isolated(fixture, "source", recorder) as graph, ExitStack() as stack:
        original_packet = _impl.context_runtime.packet
        def packet(state, query, stage="answer", budget=None):
            if arm == "compact_source" and not state.get("memory_error"):
                try:
                    result = assemble_compact_context(_impl.context_runtime.snapshot_from_state(state), query,
                        stage=stage, policy="compact_source", budget=budget,
                        summaries=state.get("memory_segments") or [])
                except (ValueError, KeyError, TypeError) as exc:
                    result = {"messages": [], "history": "", "blocked": True,
                              "reason": type(exc).__name__, "manifest": {"stage": stage}}
            else:
                result = original_packet(state, query, stage, budget)
            projections.append({"coords": deepcopy(_impl.COORDS.get()), "stage": stage,
                "blocked": result["blocked"], "reason": result.get("reason"),
                "manifest": deepcopy(result.get("manifest", {}))})
            _impl.write(recorder.out / "projections" / f"{arm}-{fixture.case['id']}.json", projections)
            return result
        stack.enter_context(patch.object(_impl.context_runtime, "packet", packet))
        for name in ("acquire_summary_lease", "release_summary_lease", "put_summary_segment"):
            stack.enter_context(patch.object(_impl.context_repository, name, getattr(fixture, name)))
        original_schedule = _impl.summarizer.maybe_schedule_summary
        async def schedule(cid):
            start = time.monotonic()
            before = len(fixture.segments)
            await original_schedule(cid)
            task = _impl.summarizer._running.get(cid)
            error = None
            if task is not None:
                try:
                    await task
                except Exception as exc:
                    error = type(exc).__name__
            maintenance.append({"coords": deepcopy(_impl.COORDS.get()), "triggered": task is not None,
                "error_type": error, "seconds": time.monotonic() - start,
                "new_segments": deepcopy(fixture.segments[before:])})
            _impl.write(recorder.out / "maintenance" / f"{arm}-{fixture.case['id']}.json", maintenance)
        stack.enter_context(patch.object(_impl.summarizer, "maybe_schedule_summary", schedule))
        yield graph


def audit(out):
    out = Path(out).resolve()
    if (out / "writer.lock").exists():
        raise ValueError("cannot audit while the live writer is active")
    result = _base_audit(out)
    manifest, data = _impl.read(out / "manifest.json"), _impl.read(out / "dataset.json")
    records = [_impl.read(p) for p in (out / "turns").glob("*.json")]
    for record in records:
        case = next(c for c in data["cases"] if c["id"] == record["coords"]["case_id"])
        expected = [{"id": i, "conversation_id": 900001, **row} for i, row in enumerate(case["initial_history"], 1)]
        if record["actual_dialog_rows"][:len(expected)] != expected:
            result["issues"].append("seeded history changed: " + str(record["coords"]))
    if manifest.get("version") != VERSION or manifest.get("compact_caps") != CAPS:
        result["issues"].append("compact experiment contract mismatch")
    calls = [_impl.read(p) for p in (out / "calls").glob("*.json")]
    maintenance = [r for p in (out / "maintenance").glob("*.json") for r in _impl.read(p)]
    for arm, group in result["groups"].items():
        rows = [r for r in records if r["coords"]["arm"] == arm]
        group["strata"] = {stratum: {"turns": len(part), "task_rule_pass": sum(r["score"]["task_rule_pass"] for r in part)}
            for stratum in ("existing_short_regression", "new_internal_budget_pressure")
            for part in [[r for r in rows if next(c for c in data["cases"] if c["id"] == r["coords"]["case_id"])["stratum"] == stratum]]}
        mcalls = [c for c in calls if c["coords"]["arm"] == arm and c["purpose"] == "conversation_summary"]
        group["maintenance_model_calls"] = len(mcalls)
        group["maintenance_input_tokens"] = sum((c.get("usage") or {}).get("input_tokens", 0) for c in mcalls)
        group["maintenance_output_tokens"] = sum((c.get("usage") or {}).get("output_tokens", 0) for c in mcalls)
        group["maintenance_failures"] = [r for r in maintenance if r["coords"]["arm"] == arm and r.get("error_type")]
    result["audit_passed"] = not result["issues"]
    result["complete"] = result["complete"] and result["audit_passed"]
    result["scope"] = f"{len(data['cases'])} synthetic trajectories (short regression plus repeated-background pressure); real model/workflow, fake business/RAG"
    result["limits"] = data["limits"] + [
        "All follow-up and triggered maintenance calls are charged. Preset-history generation is excluded.",
        "Pressure targets smaller raw-context budgets, usually below summary trigger; zero maintenance calls means summary effectiveness was not tested.",
        "A test-only summary completion barrier affects end-to-end latency if triggered.",
        "No recovery tools, production database latency, production RAG quality or independent human semantic grading are tested.",
        "Retrieval additionally applies the caller's context_coref_budget: the current compact effective retrieval cap is min(4200, 4000)=4000.",
        "The constraint-retraction pressure case only queries logistics; it does not establish correct write authorization after retraction. Write guard behavior depends on separate unit tests.",
        "Source state/write authorization guards are the same in both arms; this is a context projection comparison."]
    _impl.write(out / "audited_report.json", result)
    lines = ["# 紧凑上下文的真实 Workflow 验证", "", f"记录完整：{result['complete']}；审计通过：{result['audit_passed']}。", "",
        "|组别|规则通过|输入Token|输出Token|摘要维护调用|端到端P50秒|首次可见SSE P50秒|", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, g in result["groups"].items():
        lines.append(f"|{arm}|{g['task_rule_pass']}/{g['turns']}|{g['input_tokens']}|{g['output_tokens']}|{g['maintenance_model_calls']}|{g['end_to_end_p50_seconds']}|{g['first_visible_delta_p50_seconds']}|")
    lines += ["", "source 使用实际运行时的 4000/1800/9800 默认预算；compact_source 使用 1800/1000/4200 的分节点预算。实际配置以 manifest.json 为准，估算值不等于供应商Token。",
        "知识检索节点还会传入 context_coref_budget 作为上限；当前为4000，因此紧凑检索输入的有效预算是 min(4200,4000)=4000，不是无条件4200。",
        "两组使用相同模型、Workflow、权限/禁止写入检查、评分和工具夹具；预置历史不含本次生成费用。", "",
        "六条短轨迹是已见回归用例；三条新轨迹是内部合成预算压力补充，不是独立留出集。重复包装背景超过紧凑窗口，预期低于摘要触发阈值。",
        "全部本次生成和触发的维护调用均计费；若摘要维护为零，本轮不能说明摘要压缩效果。首次可见SSE可能是固定回复；延迟不代表生产数据库或完整知识库。", "",
        "测试在每轮结束等待实际摘要任务完成以完整记账，若触发维护，端到端时间包含这个测试专用等待。历史回读工具固定关闭，验证的是主动装配历史。", "",
        "限制撤回压力用例仅执行物流查询，检查对象选择及没有提出写入操作；不能据此证明撤回后的写授权正确，写权限仍依赖独立单元测试。", "",
        "审计问题：" + str(result["issues"]), "缺失记录：" + str(result["missing"])]
    (out / "AUDITED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


_impl.isolated_workflow = isolated_workflow
_impl.audit = audit


async def run(out):
    out = Path(out).resolve()
    manifest, data = _impl.read(out / "manifest.json"), _impl.read(out / "dataset.json")
    if manifest.get("version") != VERSION or manifest.get("compact_caps") != CAPS:
        raise ValueError("not a frozen compact workflow experiment")
    queue = iter([case for arm in manifest["arms"] for case in data["cases"]])
    def fixture():
        return Fixture(next(queue))
    with patch.object(_impl, "Fixture", fixture):
        return await _impl.run(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--turn-timeout", type=float, default=180)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run(args.output)) if args.run else audit(args.output) if args.audit else prepare(
        args.output, tuple(args.arms), args.cases, args.turn_timeout)
    print(json.dumps({key: result[key] for key in ("version", "complete", "audit_passed", "issues", "missing") if key in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
