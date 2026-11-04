"""Predeclared additions for lexical gaps; original regression cases unchanged."""
from copy import deepcopy

from scripts import context_unified_cases as original


def dataset(pilot=False):
    if pilot:
        cases = original.pilot_dataset()
        # Both pilots initialize at least one rolling summary before follow-ups.
        cases[0]["initial_history"] = pressure(original._pair(
            "本次只登记订单710008，之后查询它的配送，先不用查。", "已登记，尚未查询。"))
        return {"version": "hybrid-pilot-v1", "pilot": True, "cases": cases}
    cases = deepcopy(original.dataset())
    for c in cases:
        c["cohort"] = "previously_seen_regression"
    pair, step = original._pair, original._step
    additions = [
        ("distant_pure_reference", "唯一待办对象的远距离纯指代", pair(
            "我这次只有订单710004需要处理，记下这个号码，等资料问题问完再查询配送。",
            "已登记唯一待处理订单710004，尚未查询。"), [
            step("那单到哪了？请实际查一下。", "logistics", "710004"),
            step("还是那单，再刷新一次物流。", "logistics", "710004"),
            step("继续查询它的最新配送情况。", "logistics", "710004")]),
        ("alias_without_word_overlap", "别称关系需要跨原文理解", pair(
            "给工作室添置的那笔是710002，给家里的那笔是710006，先登记，不用查询。",
            "已记录两笔采购与用途，尚未查询。") + pair(
            "刚才提到的工作室就是摄影棚，这两个名字指同一处；家里是另一个地址。",
            "已记录这两个称呼的对应关系。"), [
            step("摄影棚那笔到哪了？请查物流。", "logistics", "710002"),
            step("换成家里那笔，查询最新物流。", "logistics", "710006"),
            step("再查摄影棚那笔的物流。", "logistics", "710002")]),
        ("adjacent_correction", "命中轮紧接的无重复关键词更正", pair(
            "工作室那单号码是710001，先登记。", "已登记这个号码，尚未查询。") + pair(
            "不对，我打错了，应当是710003，以这一条为准。",
            "已收到更正，还没有实际查询。"), [
            step("工作室那单现在到哪了？请查物流。", "logistics", "710003"),
            step("那单再刷新一次物流，别沿用上次结果。", "logistics", "710003"),
            step("继续核实它的最新配送进展。", "logistics", "710003")]),
        ("ambiguous_pure_reference", "两个未选定对象不能靠顺序猜测", pair(
            "先登记710005和710006这两个号码，都还没有决定先查哪个。",
            "已登记两个候选，等待你明确选择。") + pair(
            "我后面如果只说那单，没有明确选过，就先问我，别按先后顺序猜。",
            "好的，需要明确选择。"), [
            step("那单到哪了？", "clarify", candidates=["710005", "710006"]),
            step("明确选710006，请查询它的最新物流。", "logistics", "710006"),
            step("继续刷新那单的物流。", "logistics", "710006")]),
    ]
    for name, purpose, seed, steps in additions:
        cases.append({"id": "hybrid_" + name, "family": name, "stratum": "long",
            "cohort": "new_internal_lexical_challenge", "purpose": purpose,
            "initial_history": pressure(seed), "steps": steps,
            "label_scope": "internally authored, single replicate; not external blind data"})
    return {"version": "rolling30-keep6-hybrid-cases-v1", "synthetic": True, "pilot": False, "cases": cases}


def pressure(seed):
    rows = deepcopy(seed)
    # Exactly 32 completed rounds: maintenance occurs at 15 and 24 for keep6;
    # first query has 14 uncompressed rounds, next query sees the next summary.
    for i in range(32 - len(rows) // 2):
        q, a = original._DISTRACTORS[i % len(original._DISTRACTORS)]
        rows += original._pair(q, a)
    return rows
