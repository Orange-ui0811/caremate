"""Frozen-input five-arm study: 16 seen regressions and eight new cases.

All labels/tags are evaluator metadata. Only initial_history and each query may
reach the workflow. No labels depend on model outputs; no new scoring rules are
introduced here. The added cases are internal synthetic probes, not a blind or
representative customer-traffic test set.
"""
from __future__ import annotations

from copy import deepcopy

from scripts import context_hybrid_cases as previous
from scripts import context_unified_cases as original


VERSION = "context-keep3-five-arm-cases-20261027-v1"
_PAIR = original._pair


def _step(query, kind, target="", *, tags=(), **gold):
    result = original._step(query, kind, target, **gold)
    result["evaluation_tags"] = list(tags)
    return result


def _pad(seed, rounds, offset=0):
    """Append ordinary, non-operative paired dialogue to an exact round count."""
    rows = deepcopy(seed)
    if len(rows) % 2 or len(rows) // 2 > rounds:
        raise ValueError("padding requires complete pairs within the target")
    for i in range(rounds - len(rows) // 2):
        rows += _PAIR(*original._DISTRACTORS[(i + offset) % len(original._DISTRACTORS)])
    return rows


def _tag_regression(case):
    """Keep every old query/gold/seed unchanged, adding explicit metric tags."""
    family = case["family"]
    defaults = {
        "early_order_corrected": [("correction",), ("object_switch",), ("continuity",)],
        "natural_task_switch": [("object_switch",), ("object_switch",), ("continuity",)],
        "product_relation": [("relation",), ("relation",), ("relation",)],
        "constraint_scope_retraction": [("restriction",)] * 3,
        "ambiguous_object": [("ambiguity",), ("explicit_selection",), ("correction",)],
        "tool_authority_freshness": [("freshness",)] * 3,
        "distant_pure_reference": [("pure_reference",)] * 3,
        "alias_without_word_overlap": [("alias",)] * 3,
        "adjacent_correction": [("correction",)] * 3,
        "ambiguous_pure_reference": [("ambiguity",), ("explicit_selection",), ("continuity",)],
    }
    if family not in defaults:
        raise ValueError(f"unmapped regression family: {family}")
    for step, tags in zip(case["steps"], defaults[family], strict=True):
        step["evaluation_tags"] = list(tags)
    case["evaluation_tags"] = sorted({tag for step in case["steps"] for tag in step["evaluation_tags"]})
    case["cohort"] = "previously_seen_regression"
    case["history_estimate"] = original.history_estimate(case["initial_history"])
    return case


def _case(name, purpose, history, steps, *, stratum="long", **metadata):
    return {
        "id": "keep3_" + name,
        "family": name,
        "cohort": "new_internal_predeclared_probe",
        "stratum": stratum,
        "purpose": purpose,
        "initial_history": history,
        "history_estimate": original.history_estimate(history),
        "steps": steps,
        "evaluation_tags": sorted({tag for step in steps for tag in step["evaluation_tags"]}),
        "label_scope": "internally authored before this run; single replicate; not external blind data",
        **metadata,
    }


def _additions():
    short_delivery = _pad(_PAIR(
        "我先了解一下购物凭证怎么保管，暂时不查询任何订单。", "可以先保留原始凭证和清晰照片。"), 4, 1)
    ordinary_care = _pad(_PAIR(
        "我准备了解几件商品的日常清洁，之后会写出具体型号。", "查询时会按你本轮指定的型号核对说明。"), 8, 4)

    # With trigger=30 messages and keep=3, the first summary closes round 12;
    # the second only triggers at round 27. At round 24 this round-17 relation
    # is outside both the summary and the node's last six message projection.
    middle = _pad(_PAIR("先了解一些资料整理方法。", "可以，请说具体问题。"), 16, 3)
    middle += _PAIR("北仓库这笔订单是710008，南仓库这笔是710001，先记下两处和订单的对应关系。",
                    "已登记两处对应的号码，目前还没查询配送。")
    middle = _pad(middle, 24, 12)

    correction = _PAIR("摄影工作室的订单号码记为710002，家用的是710006，暂时都不查询。",
                       "已登记两笔号码与用途，尚未读取业务数据。")
    correction = _pad(correction, 5, 6)
    correction += _PAIR("纠正前面摄影工作室的号码：不是710002，应当是710004；家用号码不变。",
                        "已记录更正，旧号码不再用于摄影工作室的查询。")
    correction = _pad(correction, 32, 10)

    alias = _PAIR("青禾点的订单是710003，白露点的订单是710008，先登记不用查。",
                  "已记录两个点位与号码，尚未查询。")
    alias += _PAIR("青禾点后来改名叫书房店，白露点仍沿用原名；说书房店就是青禾点。",
                   "已登记名称对应关系，白露点的名称与订单对应没有变。")
    alias = _pad(alias, 30, 2)

    reference = _PAIR("现在只保留订单710008这个待查事项，等资料整理完再继续查询它的配送。",
                      "已登记这个待查事项，尚未读取配送记录。")
    reference = _pad(reference, 31, 7)

    restriction = _PAIR("工作室订单710002和家用订单710008都只许查询，不要催单，也不要创建工单。",
                        "已登记用途关系与只读要求，目前没有发起任何办理。")
    restriction += _PAIR("以后即使撤回某一项限制，办理前也仍须我另行明确确认，不能当成自动授权。",
                         "了解，限制撤回和实际操作授权要分别核对。")
    restriction = _pad(restriction, 29, 9)

    ambiguity = _PAIR("给北仓库的订单有710001和710004，两张都先登记，目前没选哪张先查。",
                      "已登记两个候选，等待明确选择。")
    ambiguity += _PAIR("它们用途相同，不要根据用途或提到的先后替我决定。", "需要你明确选定之后再读取物流。")
    ambiguity = _pad(ambiguity, 28, 11)

    return [
        _case("ordinary_explicit_delivery", "短会话显式订单查询，检查额外记忆开销与普通任务退化", short_delivery, [
            _step("请查询订单710001现在的物流。", "logistics", "710001", tags=("ordinary",)),
            _step("现在查询订单710004的最新物流。", "logistics", "710004", tags=("ordinary",)),
            _step("请重新查询订单710004当前的配送进展。", "logistics", "710004", tags=("ordinary",)),
        ], stratum="short"),
        _case("ordinary_explicit_care", "普通会话每问明确型号，无需长期记忆的知识查询", ordinary_care, [
            _step("MH-CAM1摄像头的镜头该怎样清洁？请查说明。", "care", "MH-CAM1", tags=("ordinary",),
                  fact_groups=[["断电", "关闭电源", "切断电源", "拔掉电源"], ["镜头布", "软布"], ["镜头"]]),
            _step("请查MH-FD10喂食器食盆的清洗方法。", "care", "MH-FD10", tags=("ordinary",),
                  fact_groups=[["断电", "关闭电源", "切断电源", "拔掉电源"], ["食盆"], ["拆", "取下"], ["晾干", "干燥", "擦干"]]),
            _step("MH-FD10喂食器的机身怎样清洁？请按这个型号说明回答。", "care", "MH-FD10", tags=("ordinary",),
                  fact_groups=[["机身", "主机"], ["软布"], ["擦"]]),
        ], stratum="short"),
        _case("unsummarized_middle_relation", "未摘要且已离开节点近期输入的中间区关系，检查回查覆盖盲区", middle, [
            _step("北仓库那笔现在到哪了？请实际查物流。", "logistics", "710008", tags=("unsummarized_middle", "relation")),
            _step("换查南仓库那笔的最新物流。", "logistics", "710001", tags=("unsummarized_middle", "relation")),
            _step("回到北仓库那笔，再刷新一次物流。", "logistics", "710008", tags=("unsummarized_middle", "relation")),
        ], relation_seed_round=17, seed_rounds=24),
        _case("nonadjacent_correction", "更正与首次登记之间隔四轮资料问答，不能只读取首次命中的相邻轮", correction, [
            _step("摄影工作室那笔现在到哪了？请用更正后的号码查询。", "logistics", "710004", tags=("correction",)),
            _step("同一笔再次刷新物流，不要换回更正前的号码。", "logistics", "710004", tags=("correction",)),
            _step("现在看家用那笔的配送进度。", "logistics", "710006", tags=("relation",)),
        ]),
        _case("renamed_location_alias", "点位改名与订单关系需联合使用，且另一点位不受影响", alias, [
            _step("书房店那笔到哪了？请查询物流。", "logistics", "710003", tags=("alias",)),
            _step("再查询白露点那笔的最新物流。", "logistics", "710008", tags=("alias",)),
            _step("回到书房店那笔，重新核实配送进展。", "logistics", "710003", tags=("alias",)),
        ]),
        _case("sole_pending_pure_reference", "长会话唯一待查对象以纯指代恢复，检验摘要的背景作用", reference, [
            _step("那单到哪了？请实际查一下。", "logistics", "710008", tags=("pure_reference",)),
            _step("还是那单，再刷新一下。", "logistics", "710008", tags=("pure_reference",)),
            _step("它现在有什么新的进展？再查一下。", "logistics", "710008", tags=("pure_reference",)),
        ]),
        _case("scoped_readonly_retraction", "部分撤回仍无正向办理授权，只考察正确查询且不提议或执行写操作", restriction, [
            _step("先查工作室那单最新物流，仍然不要催单和开工单。", "logistics", "710002", tags=("restriction",),
                  active_constraints={"710002": ["ticket", "expedite"], "710008": ["ticket", "expedite"]}),
            _step("只撤回家用那单不能开工单这一项，工作室限制不变。但本轮只查家用那单物流，尚未授权办理。", "logistics", "710008", tags=("restriction",),
                  active_constraints={"710002": ["ticket", "expedite"], "710008": ["expedite"]}),
            _step("换回工作室那单，刷新物流，之前针对它的限制继续有效。", "logistics", "710002", tags=("restriction",),
                  active_constraints={"710002": ["ticket", "expedite"], "710008": ["expedite"]}),
        ]),
        _case("same_alias_ambiguous_orders", "同一用途仍有两个未选候选，先澄清，再接收显式选择与更正", ambiguity, [
            _step("北仓库那单到哪了？", "clarify", tags=("ambiguity",), candidates=["710001", "710004"]),
            _step("明确选710001，请查最新物流。", "logistics", "710001", tags=("explicit_selection",)),
            _step("刚才选错，改成710004，重新查这张单。", "logistics", "710004", tags=("correction",)),
        ]),
    ]


def dataset(pilot=False):
    if pilot:
        # This smoke case is excluded from every formal denominator. Eighteen
        # rounds exercise initial trigger=30 messages, keep=3 maintenance.
        case = _case("pilot_delivery", "独立预检：摘要初始化与三轮实际查询链路", _pad(_PAIR(
            "只登记订单710007，稍后再查询它的配送。", "已登记，尚未查询。"), 18, 8), [
                _step("请查询订单710007的最新物流。", "logistics", "710007", tags=("pilot",)),
                _step("还是那单，请重新查询。", "logistics", "710007", tags=("pilot",)),
                _step("继续刷新它现在的配送进度。", "logistics", "710007", tags=("pilot",)),
            ], stratum="pilot")
        case["cohort"] = "separate_pilot"
        return {"version": VERSION + "-pilot", "synthetic": True, "pilot": True, "cases": [case]}
    cases = [_tag_regression(case) for case in deepcopy(previous.dataset()["cases"])]
    cases.extend(_additions())
    return {
        "version": VERSION, "synthetic": True, "pilot": False,
        "metric_tag_contract": {
            "correction": "step-tagged correction tasks; original task-rule scorer",
            "restriction": "correct read target and no proposed/executed write; not positive write authorization or exhaustive state validation",
            "ambiguity": "only expected.kind=clarify steps; later selected queries are excluded from clarification denominator",
            "episode": "all three actual sequential follow-ups pass; scripted seed dialogue is not generated or scored",
            "tags_overlap": "case tags are unions for description; metric denominators use step tags",
        },
        "cases": cases,
    }
