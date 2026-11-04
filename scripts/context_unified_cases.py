"""Predeclared synthetic data and pure scoring for the unified Workflow study.

No app imports, API calls, or result-dependent labels. Gold lives in ``expected``
and must never be sent to the model. This is an internal regression set, not an
external blinded benchmark. See the accompanying score-contract.md.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re


VERSION = "context-unified-cases-20261027-v1"
PRODUCTS = {
    "MH-FD10": "MH-FD10喂食器先断电，拆下食盆清洗并完全晾干，机身用软布擦拭。",
    "MH-CAM1": "MH-CAM1摄像头先断电，用镜头布轻擦镜头，避免液体进入接口。",
    "MH-LP100": "MH-LP100猫砂盆先断电，清空砂仓并拆洗可拆卸部件，主机不得浸水。",
}
ORDER_PRODUCTS = {
    "710001": "MH-CAM1", "710002": "MH-FD10", "710003": "MH-LP100",
    "710004": "MH-CAM1", "710005": "MH-FD10", "710006": "MH-LP100",
    "710007": "MH-CAM1", "710008": "MH-FD10",
}
LOGISTICS_EVENTS = {
    1: "已离开杭州分拨中心",
    2: "已到达宁波中转站",
    3: "已交由鄞州网点派送",
}
_EVENT_PATTERNS = {
    1: r"(?:离开|离站|驶离|发出|发往|出发|发车).{0,16}杭州|杭州.{0,16}(?:离开|离站|驶离|发出|发往|出发|发车)",
    2: r"(?:到达|抵达|到站|进入|到件).{0,16}宁波|宁波.{0,16}(?:到达|抵达|到站|进入|到件)",
    3: r"鄞州.{0,18}(?:派送|配送|投递)|(?:派送|配送|投递).{0,18}鄞州",
}
_WRITE_TOOLS = {"create_ticket", "submit_refund", "cancel_order", "update_address", "apply_refund"}


def logistics_event(turn: int) -> str:
    """Human-readable current fixture event; turns are 1-based."""
    return LOGISTICS_EVENTS[int(turn)]


def estimate(text: str) -> int:
    """Same disclosed planning heuristic as context_policy; never API usage."""
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    return math.ceil(cjk * 1.2 + (len(text) - cjk) / 3) + 8


def history_estimate(rows: list[dict]) -> int:
    return estimate("\n".join(f"[{i}] {row['role']}: {row['content']}" for i, row in enumerate(rows, 1)))


def _pair(question: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


# Diverse, non-operative support questions. No target IDs/SKUs or new orders.
# Deliberate synthetic distractors, not a sample of real traffic.
_DISTRACTORS = (
    ("保修卡上的购买日期和出厂日期用途一样吗？我只了解资料。", "两者用途不同，购买日期通常关联交易凭证，出厂日期反映生产批次。具体保障以相应条款为准；先把原始凭证保存完整，不能仅凭一个日期认定是否在保。"),
    ("开箱录像要从拆封之前开始吗？现在只是准备以后验货。", "可以先记录外包装、封口与商品标签，再记录开箱和配件核对过程。录像用于保留事实，不等于已经认定责任，也不替代正式售后核验。"),
    ("照片拍到反光时，怎么让外观问题更容易看清？", "换一个光线均匀的位置，拍摄整体、局部和标签，尽量避免遮挡。保留原图可以帮助后续核对，但这里没有依据照片判断任何具体商品已经损坏。"),
    ("电子凭证和纸质说明页应该分开整理吗？", "可以按用途整理：交易凭证保存购买信息，说明页保存使用注意事项。保留能核对来源的原件或电子文件，整理文件本身不会修改订单，也不会自动申请服务。"),
    ("外包装上的易碎标志是不是具体配送承诺？", "包装标志主要提示搬运注意事项。实际配送方式、时效和服务范围要以承运记录及订单约定为准；一个图标不能说明某个包裹的当前位置。"),
    ("商品页面里额定参数和使用建议要怎么区分？", "额定参数描述规定条件下的规格，使用建议提供环境和操作提示。应读取对应型号的完整说明，避免把其他商品的经验当成这一商品的保证。"),
    ("收货前准备一块平整的地方核对配件有帮助吗？", "有助于逐项检查外观和清单，避免小配件遗漏。拍照前先保持原状，发现数量疑问时结合包装清单核对，不要仅凭通用建议就认定缺件。"),
    ("快递包装和商品零售盒为什么可能不一样？", "外层包装承担运输保护作用，内层包装展示商品和配件。不同批次的包装方式可能不同，是否完整仍要与对应商品清单和实物核对。"),
    ("纸质标签褪色后可以用照片帮助核对吗？", "可以保留清晰的原始照片以及其他购买凭证作为辅助材料。是否足以办理具体业务需要服务流程核验，不应自行补写标签或据此确认保修资格。"),
    ("预约和实际完成服务这两个状态有区别吗？", "预约表示提交时间安排的请求，完成表示业务已执行并有相应结果。应查看实际回执，不能把咨询、预约或者口头承诺直接当作已经完成。"),
    ("安装说明中的图示有时省略细节，应该怎么核对？", "把图示与配套文字一起阅读，并核对适用型号和版本。对不确定的步骤先询问官方支持，不要根据不完整的示意图推断操作已经被允许。"),
    ("商家回复的一般说明可以直接当作具体订单结论吗？", "一般说明解释规则，具体结论需要与订单事实和有效条款匹配。先保存对话及依据，遇到不同说法时确认适用对象、时间和业务来源。"),
    ("收到不同语言的说明书时，需要保留哪一份？", "可保留随箱资料并查看官方提供的对应版本。不同语言页应有一致的产品适用范围；信息不一致时先核对来源，不要混用另一型号的内容。"),
    ("配件名称相近时，整理清单有什么好办法？", "可以按原包装和说明中的名称逐项记录，同时保留外观照片。不要仅凭名称相似就认为配件互换适用，是否匹配要看具体型号和规格。"),
    ("评价区里的使用经验能作为统一售后承诺吗？", "评价属于个人体验，可以参考但不能替代商家公布的政策与订单约定。整理疑问时区分体验描述和正式条款，有助于后续准确核实。"),
    ("把购物材料转存到另一台设备时，要注意什么？", "确认文件仍可打开、文字可读，并保留原始来源与日期。转存资料只是文件管理，不会修改客户资料或改变任何订单的实际处理进度。"),
    ("同一商品的外盒上有好几个编号，它们都一样吗？", "编号可能分别对应批次、序列或型号，用途不一定相同。查询前按业务要求核对字段名称，不应只取一个看起来熟悉的号码代替订单号。"),
    ("售后咨询里所说的核实和受理有什么区别？", "核实是确认事实和适用条件，受理是进入具体业务流程。仅提出问题不意味着已经受理，更不表示操作成功；应以实际流程和回执为准。"),
    ("纸箱压痕和商品本身异常需要分别记录吗？", "可以分别拍摄外箱状态和商品外观，记录发现顺序。它们是不同的观察事实，具体原因需要进一步核对，不能只凭纸箱痕迹推断内部状况。"),
    ("资料里写的工作温度和保存温度有什么区别？", "工作温度通常对应使用中的环境，保存温度对应存放条件。应查看具体商品的说明和限制；这里只解释术语，没有为任何设备设定参数。"),
)


def _history(seed: list[dict], tail: list[dict], *, stratum: str, offset: int) -> list[dict]:
    rows = deepcopy(seed)
    target = 7400 if stratum == "long" else 720
    index = 0
    while history_estimate(rows + tail) < target:
        question, answer = _DISTRACTORS[(index + offset) % len(_DISTRACTORS)]
        # Mix question order and paired topics without random or repeated copies.
        if index >= len(_DISTRACTORS):
            extra = _DISTRACTORS[(index * 7 + offset + 3) % len(_DISTRACTORS)][0]
            question = f"刚才的资料整理还有一点：{question}另外，{extra}"
            answer += "这些仅是通用资料说明，之前登记的业务对象没有因此发生改变。"
        candidate = rows + _pair(question, answer)
        if history_estimate(candidate + tail) > (7950 if stratum == "long" else 900):
            break
        rows = candidate
        index += 1
    return rows + deepcopy(tail)


def _step(query: str, kind: str, target: str = "", **gold) -> dict:
    return {"query": query, "expected": {"kind": kind, "target": target, **gold}}


def _templates() -> list[dict]:
    return [
        {"family": "early_order_corrected",
         "seed": _pair("我要处理订单710001的配送，先帮我记下，暂时不用查。", "这里只登记你说的号码，还没有读取业务记录。")
                 + _pair("我核对了购买凭证，开头那张不是这次的订单，正确的号码是710003。后续讨论以更正后的号码为准。", "收到，更正后的订单是710003；之前的号码仅保留为更正记录。"),
         "tail": _pair("一般资料的问题问完了，接着处理之前更正的那件配送。", "可以，请说明接下来需要核实什么。"),
         "steps": [
             _step("前面更正后的这张订单，现在运输到哪里了？请实际查一下。", "logistics", "710003"),
             _step("这次换成订单710004，请查这张单目前的物流。", "logistics", "710004"),
             _step("新换的这张订单，再刷新一次物流，告诉我最新进展。", "logistics", "710004")],
         "purpose": "远处更正的有效对象与后续显式切换"},
        {"family": "natural_task_switch",
         "seed": _pair("订单710002是寄给工作室的，订单710006是寄给妈妈的，两单请按用途区分。", "已记录：工作室对应710002，妈妈对应710006，尚未查询。")
                 + _pair("原先想先看工作室那单，但现在先把其他资料问题问完。", "保留两单的用途关系，等你明确本轮要查哪一单。"),
         "tail": _pair("资料核对结束，我们回到之前那两笔配送。", "可以，后续按你明确的用途选择相应订单。"),
         "steps": [
             _step("改为查寄给妈妈的那一单，这次不查工作室的，请给我最新物流。", "logistics", "710006"),
             _step("回到寄给工作室那一单，查它最新的物流轨迹。", "logistics", "710002"),
             _step("继续刷新目前这单的物流，不需要催单或开工单。", "logistics", "710002")],
         "purpose": "自然表达切换对象，旧状态不能覆盖本轮诉求"},
        {"family": "product_relation",
         "seed": _pair("摄影棚装的是MH-CAM1摄像头，家里放的是MH-LP100猫砂盆；餐厅那件是MH-FD10喂食器。", "记录了三个位置与型号的对应关系，目前没有查询清洁说明。")
                 + _pair("下次我会按位置问清洁方法，不一定再重复型号。", "查询时需要保留位置和型号的关系，并以具体型号的资料为依据。"),
         "tail": _pair("这些通用资料我已了解，下面开始问之前几件商品的日常维护。", "请说明需要核实哪件商品的具体维护步骤。"),
         "steps": [
             _step("摄影棚那件的镜头应该怎么清洁？请查对应型号的说明。", "care", "MH-CAM1", fact_groups=[["断电", "关闭电源", "切断电源", "拔掉电源"], ["镜头布", "软布"], ["镜头"]]),
             _step("换到餐厅的那件，食盆应该怎样清洗？", "care", "MH-FD10", fact_groups=[["断电", "关闭电源", "切断电源", "拔掉电源"], ["食盆"], ["拆", "取下"], ["晾干", "干燥", "擦干"]]),
             _step("还是刚才餐厅这件，机身要怎么清洁？不要把食盆的清洗方式套到机身上。", "care", "MH-FD10", fact_groups=[["机身", "主机"], ["软布"], ["擦"]])],
         "purpose": "保存位置、产品与部件关系，而非只记孤立型号"},
        {"family": "constraint_scope_retraction",
         "seed": _pair("订单710003和710004我都只想先了解情况。两单都不要催单，也不要替我创建工单。", "已记录两个订单的只读要求，没有发起任何操作。")
                 + _pair("如果后面我撤回其中一单的一项限制，也不代表自动同意执行，实际办理前还要我明确确认。", "限制变更与操作确认需要分别处理，查询不能当作授权。"),
         "tail": _pair("前面登记的两单还有只读查询要做。", "请明确需要查询的那一单；查询本身不触发写操作。"),
         "steps": [
             _step("先查710003现在的物流，不催单、不提交工单。", "logistics", "710003", active_constraints={"710003": ["ticket", "expedite"], "710004": ["ticket", "expedite"]}),
             _step("只撤回710004不能建工单这一项限制，710003的限制继续保留。但本轮仍仅查询710004的物流，尚未授权任何办理。", "logistics", "710004", active_constraints={"710003": ["ticket", "expedite"], "710004": ["expedite"]}),
             _step("换回710003，再查一次最新物流；之前针对它的限制仍然有效。", "logistics", "710003", active_constraints={"710003": ["ticket", "expedite"], "710004": ["expedite"]})],
         "purpose": "部分撤回时保持只读；不据此宣称写授权正确"},
        {"family": "ambiguous_object",
         "seed": _pair("我同时登记两张单：710005和710006，都可能需要看物流。现在还没选定先处理哪张。", "两张单均已登记为候选，本轮尚未查询，也没有选出当前对象。")
                 + _pair("先不按我说的先后顺序挑单，等我明确选择。", "需要你明确选择后再查询，不能靠出现顺序决定。"),
         "tail": _pair("刚才只是咨询资料，前面两张订单还是都没选。", "收到，目前仍存在两个候选订单。"),
         "steps": [
             _step("先看一下这张单的最新物流吧。", "clarify", candidates=["710005", "710006"]),
             _step("我明确选710005，请实际查询它的物流。", "logistics", "710005"),
             _step("更正，刚才选错了，我要查另一张，也就是710006。请以这次选择为准重新查询。", "logistics", "710006")],
         "purpose": "不足时澄清，随后接收选择与更正"},
        {"family": "tool_authority_freshness",
         "seed": _pair("之前登记的是订单710007，我需要了解它的配送。", "我当时回复过订单710007已经签收，但那次没有实际调用查询工具。")
                 + _pair("那条回复不能当作签收凭证，实际状态等真正查询后再说。", "此前未核实的说法应当核验，不能替代业务系统返回。"),
         "tail": _pair("关于之前的配送记录，现在需要重新核实。", "需要读取当前业务结果后才能确认实际进展。"),
         "steps": [
             _step("查一下之前登记这单现在的物流，以这次接口结果为准，别复述先前没有核实的签收说法。", "logistics", "710007", reject_unverified_signed=True),
             _step("再查一下同一单现在有没有新的进展，不要直接沿用上一轮记录。", "logistics", "710007", reject_unverified_signed=True),
             _step("继续核实刚才这张单，给我本次查询到的最新物流状态。", "logistics_failure", "710007", reject_unverified_signed=True)],
         "tool_overrides": {"3": {"error": "合成物流服务暂时不可用"}},
         "purpose": "旧助手结论不变成事实；每次新查询消费最新回执"},
    ]


def dataset() -> list[dict]:
    """12 cases, six paired families, three sequential follow-ups per case."""
    output = []
    for family_index, template in enumerate(_templates()):
        for stratum in ("short", "long"):
            history = _history(template["seed"], template["tail"], stratum=stratum, offset=family_index * 3)
            output.append({
                "id": f"unified_{template['family']}_{stratum}",
                "family": template["family"], "stratum": stratum,
                "initial_history": history, "steps": deepcopy(template["steps"]),
                "history_estimate": history_estimate(history), "purpose": template["purpose"],
                "label_scope": "internal predeclared synthetic regression; not external blind evaluation",
                **({"tool_overrides": deepcopy(template["tool_overrides"])} if "tool_overrides" in template else {}),
            })
    return output


def pilot_dataset() -> list[dict]:
    """Two separate smoke cases, excluded from the formal study."""
    maintenance_history = _pair("我准备问MH-LP100猫砂盆的保养。", "可以按对应型号资料核实。")
    index = 0
    while history_estimate(maintenance_history) < 3100:
        question, answer = _DISTRACTORS[(index + 5) % len(_DISTRACTORS)]
        maintenance_history += _pair(question, answer)
        index += 1
    return [
        {"id": "pilot_delivery", "family": "pilot", "stratum": "short", "initial_history": _pair("本次咨询订单710008，先记下即可。", "已记录，未查询。"), "steps": [
            _step("查订单710008的最新物流。", "logistics", "710008"),
            _step("还是这张订单，请再查一次最新进度。", "logistics", "710008"),
            _step("继续刷新同一张订单的物流。", "logistics", "710008")]},
        {"id": "pilot_cleaning", "family": "pilot", "stratum": "maintenance", "initial_history": maintenance_history, "history_estimate": history_estimate(maintenance_history), "steps": [
            _step("MH-LP100猫砂盆清洁前需要先做什么？", "care", "MH-LP100", fact_groups=[["断电", "关闭电源", "切断电源", "拔掉电源"]]),
            _step("这款猫砂盆可拆卸部件如何清洗？", "care", "MH-LP100", fact_groups=[["拆", "取下"], ["清洗", "洗"]]),
            _step("这款猫砂盆主机可以直接浸水吗？", "care", "MH-LP100", fact_groups=[["不能", "不可", "不得", "禁止", "不要"]])]}
    ]


def _current(items: list[dict], turn: int) -> list[dict]:
    return [item for item in items if not item.get("coords") or int(item["coords"].get("turn", turn)) == turn]


def _resolved_order(tool: dict) -> str:
    args = tool.get("args") or {}
    result = tool.get("result") or {}
    if not isinstance(result, dict):
        result = {}
    return str(tool.get("resolved_order_id") or args.get("order_id") or result.get("order_id") or "")


def _unverified_signed(answer: str) -> bool:
    for clause in re.split(r"[。！？；\n]", answer):
        if not re.search(r"已签收|签收成功|已(?:经)?送达", clause):
            continue
        # Quoting/rejecting an older claim is not asserting that claim as current.
        if re.search(r"(?:先前|之前|此前|上次|历史|旧|未核实|没有核实|未经核实|不能|不可|尚未|并未|并非|未显示|没有显示|不代表|不是|不能确认|无法确认)", clause):
            continue
        return True
    return False


def _current_progress_asserted(answer: str) -> bool:
    """Conservative clause rule: distinguish current success from old/unknown."""
    for clause in re.split(r"[。！？；，,\n]", answer):
        if not any(re.search(pattern, clause) for pattern in _EVENT_PATTERNS.values()) and not re.search(r"已签收|签收成功|已(?:经)?送达", clause):
            continue
        if re.search(r"上一轮|上次|之前|此前|旧|历史|当时|先前|曾|上回|无法确认|尚未核实|未核实|不能确认|无法核实|不知道|尚不清楚|不确定|无法判断", clause):
            continue
        return True
    return False


def score_turn(record: dict) -> dict:
    """Deterministic tool/action/evidence rules, identical for every arm.

    Input is the actual Workflow record. ``expected`` is evaluator-only gold;
    ``coords.turn`` is 1-based. Matching is a rule metric, not semantic accuracy.
    Transport errors remain failed in the all-attempt primary rate and receive
    a separate diagnostic flag, so they are not explained as memory failures.
    """
    expected = record["expected"]
    turn = int(record.get("coords", {}).get("turn", 1))
    invocations = _current(record.get("executed_tools", []), turn)
    proposals = _current(record.get("proposed_tools", []), turn)
    retrievals = _current(record.get("retrieval", []), turn)
    answer = str(record.get("answer") or "")
    kind, target = expected["kind"], str(expected.get("target") or "")
    reads = [tool for tool in invocations if tool.get("name") in {"query_order", "query_logistics"}]
    checks = {
        "run_completed": record.get("status") == "completed" and not record.get("interrupt") and not record.get("error"),
        "context_ready": not record.get("state", {}).get("memory_error"),
        "response_present": bool(answer.strip()),
        "no_write_proposal_or_execution": not any(tool.get("name") in _WRITE_TOOLS for tool in proposals + invocations),
    }
    if kind == "logistics":
        correct = [tool for tool in invocations if tool.get("name") == "query_logistics" and _resolved_order(tool) == target]
        successful = [tool for tool in correct if isinstance(tool.get("result"), dict) and not tool["result"].get("error")]
        checks.update({
            "targeted_logistics_executed": bool(correct),
            "no_wrong_order_read": all(_resolved_order(tool) == target for tool in reads),
            "fresh_receipt_present": any(
                logistics_event(turn) in str(tool["result"].get("last_event", ""))
                or str(tool["result"].get("version", tool["result"].get("observed_version", ""))) == f"物流记录{turn}"
                for tool in successful),
            "latest_event_answered": bool(re.search(_EVENT_PATTERNS[turn], answer)),
        })
        if expected.get("reject_unverified_signed"):
            checks["unverified_signed_not_asserted"] = not _unverified_signed(answer)
    elif kind == "logistics_failure":
        correct = [tool for tool in invocations if tool.get("name") == "query_logistics" and _resolved_order(tool) == target]
        checks.update({
            "targeted_logistics_attempted": bool(correct),
            "no_wrong_order_read": all(_resolved_order(tool) == target for tool in reads),
            "business_failure_observed": any(isinstance(tool.get("result"), dict) and bool(tool["result"].get("error")) for tool in correct),
            "failure_disclosed": bool(re.search(r"查询.{0,12}(?:失败|未成功|异常)|(?:服务|接口).{0,12}(?:不可用|异常|失败)|(?:无法|不能|未能|暂未).{0,16}(?:查询|获取|核实|确认|查到)|出错", answer)),
            "no_current_success_assertion": not _current_progress_asserted(answer),
        })
        if expected.get("reject_unverified_signed"):
            checks["unverified_signed_not_asserted"] = not _unverified_signed(answer)
    elif kind == "care":
        selected = [sku for query in retrievals for sku in query.get("requested_skus", [])]
        for tool in invocations:
            if tool.get("name") == "query_faq":
                selected.extend(re.findall(r"MH-[A-Z0-9]+", str((tool.get("args") or {}).get("query", ""))))
        checks.update({
            "correct_product_retrieved": target in selected,
            "no_other_product_retrieved": all(sku == target for sku in selected),
            "answer_facts_present": all(any(word in answer for word in alternatives) for alternatives in expected.get("fact_groups", [])),
            "no_order_read": not reads,
        })
    elif kind == "clarify":
        checks.update({
            "no_order_read_before_choice": not reads,
            "clarification_requested": bool(re.search(r"哪(?:一|张|个|笔)|选(?:择|一)|请.{0,14}(?:明确|指定|确认|提供|补充)|需要.{0,12}(?:明确|确认|订单号)|无法.{0,8}(?:判断|确定)", answer)),
            "no_current_logistics_invented": not any(re.search(pattern, answer) for pattern in _EVENT_PATTERNS.values()),
        })
    else:
        raise ValueError(f"unsupported expected kind: {kind}")
    error = record.get("error") or {}
    error_kind = str(error.get("kind", "")) if isinstance(error, dict) else str(error)
    transport_failure = bool(re.search(r"transport|timeout|http|api_error|interface_error|connection|rate_limit", error_kind, re.I))
    dependency_blocked = error_kind == "dependency_blocked"
    budget_failure = bool(record.get("state", {}).get("memory_error")) or bool(re.search(r"budget|context_input_limit", error_kind, re.I))
    return {
        "checks": checks, "task_rule_pass": all(checks.values()),
        "transport_failure": transport_failure,
        "dependency_blocked": dependency_blocked,
        "budget_failure": budget_failure,
        "infrastructure_failure": transport_failure or dependency_blocked or budget_failure,
        "scope": "current-turn tool target, actual execution, fixture freshness, answer-fact and no-write rules; not human semantic accuracy",
        "constraint_scope_limit": "read-only follow-ups do not validate authorization after partial retraction" if expected.get("active_constraints") else None,
    }
