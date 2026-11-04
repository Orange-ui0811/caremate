"""Frozen known regressions and new phrasing variants for matched-budget evaluation.

Only fictional customer conversations are used. Gold evidence and expected state
are evaluator metadata: context builders must receive only the visible history
prefix and the current query. New variants are not an external blind benchmark.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.context_benchmark_cases import KINDS, PRODUCTS, make_suite


BACKGROUND = (
    ("我在整理收货须知，外箱的防拆标识应该在哪一步检查？", "收到包裹时可以先观察外箱和封口，再记录是否有明显破损。检查外包装与检查产品功能是两件事，照片最好能同时看到包裹整体和具体位置。发现异常后保留包装并说明实际情况，不能只根据一个标签判断原因。这里仅解释检查步骤。"),
    ("通用安装说明为什么同时写开盖空间和摆放空间？", "摆放空间指产品主体占用的位置，开盖空间还包括维护时需要活动的范围。准备位置时可以分别核对桌面是否平稳、插座位置是否方便、周围是否留有操作空间。具体尺寸要看相应商品说明，这些通用建议不会替代实际型号的安装条件。"),
    ("我想把说明书和电子发票分类保存，这两类文件分别用来干什么？", "说明书用于查阅使用步骤、维护要求和禁止事项，电子发票用于记录交易与票据信息。保存时可以按用途分类，避免把发票上的商品名称当成完整的技术规格。申请服务时可能需要两类文件配合核对，但阅读此说明不会自动提交任何申请。"),
    ("商品展示图的颜色和实物照片有差异时，应该怎样向客服说明？", "可以分别提供商品页面所选规格和自然光下的实物照片，并说明观察环境。不同屏幕、拍摄光线和背景都可能影响颜色表现。是否存在发错规格仍要核对实际记录，不能仅凭展示图的视觉差异直接判断。此处讨论的是沟通材料的准备。"),
    ("包装清单里的主件、附件和赠送品，检查时应当分开记录吗？", "可以按照包装清单的项目分别记录实物是否齐全，主件和附件需要对应具体商品，赠送品还要核对适用活动。若某项名称不清楚，可以保存清单及实物照片供人工核实。这里没有确认任何一笔购买附带赠品，也没有变更此前登记的内容。"),
    ("为什么使用提示会把断开电源和清洁表面分成不同步骤？", "这类顺序用于提醒先满足操作条件，再开展维护工作。不同产品的可拆卸部件和可清洁区域不相同，不能把一款产品的做法照搬到另一款。实际维护应查阅对应型号的说明；这一轮只解释说明书的编排方式，不提供具体型号的操作结论。"),
    ("快递代收点通知和承运方物流记录应该如何区分？", "代收点通知通常用于告知取件安排，承运方记录用于描述运输过程中的扫描事件。两类信息的发布时间和更新节奏可能不同，查看时应注意它们分别来自哪里。若要确认某个包裹的当前情况，需要查询对应物流记录，这里没有查询任何包裹。"),
    ("买家评价提到的体验，可以直接当作售后政策吗？", "评价通常反映个人使用经历，政策则需要查看适用的正式条款。二者可以帮助了解商品，但不能相互替代。咨询时可以指出具体疑问和希望核实的条件，由客服结合商品及购买情况确认。此处没有对任何个案作出保修或退款承诺。"),
    ("保修卡上有使用提示，这和随箱说明书的关系是什么？", "保修卡主要承载服务相关信息，附带的使用提示可能只是简要提醒；完整操作条件仍应查看对应说明书。保存资料时可以把两者放在同一文件夹，并保留购买凭证。资料齐全有助于核实情况，但不能据此提前认定某次损坏属于保修范围。"),
    ("安装前检查配件时，如果一个配件名称看不懂，该如何描述？", "可以拍摄配件整体、接口形状和包装上的名称，并说明它位于清单哪个位置。尽量使用可观察到的特征，不自行猜测零件功能，也不要为了验证用途进行改装。客服需要结合对应说明书确认，通用描述不能替代对具体配件的识别。"),
    ("我要给家人整理服务渠道说明，在线咨询和提交申请有什么区别？", "在线咨询可以用来了解规则、补充信息和确认所需材料，提交申请则可能创建一条需要处理的业务记录。是否已经提交应以明确操作和系统结果为依据，不能因为聊过某项服务就认为流程已经开始。这轮没有创建任何服务申请。"),
    ("收货照片里为什么建议保留外箱、清单和配件这几个部分？", "这些照片分别帮助核对运输外观、应有物品和实际收到的物品。拍摄时尽量清晰完整，并避免遮挡需要辨认的位置。不同问题需要的材料可能不同，客服会根据情况补充说明；照片只是核实依据，不能单独证明某个处理结果已经获批。"),
)


def _put(history, turn, user, assistant):
    history[2 * turn - 2:2 * turn] = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _background(index, turns=32):
    rows = []
    for turn in range(turns):
        first = BACKGROUND[(turn + index) % len(BACKGROUND)]
        second = BACKGROUND[(turn * 5 + index + 3) % len(BACKGROUND)]
        rows += [{"role": "user", "content": first[0] + "另外，" + second[0]},
                 {"role": "assistant", "content": first[1] + second[1]}]
    # A moderately sized formatting distractor, rather than repeated huge text.
    labels = ("封口", "纸托", "内衬", "袋口", "提手", "隔板", "缓冲层", "折边")
    table = "\n".join(f"|{name}|查看完整外观|记录可见情况|保留对应照片|" for name in labels)
    user = "我在排版通用开箱记录表，下面这些栏目是否容易理解？表中没有本次购买的编号或型号。\n|部位|观察项|记录项|附件|\n|---|---|---|---|\n" + table
    _put(rows, 14, user, "栏目分别描述观察对象、应记录的信息和附带材料，含义清楚。实际发生问题时再填写观察结果，现在这张空表不能作为已经开箱或已经完成检查的记录。")
    return rows


def _span(history, turn):
    seq = 2 * turn - 1
    row = history[seq - 1]
    return {"seq": seq, "start": 0, "end": len(row["content"]),
            "role": row["role"], "text": row["content"]}


def _annotate(case, source_turns, rationales):
    for probe, turns, rationale in zip(case["probes"], source_turns, rationales):
        probe["gold_evidence"] = [_span(case["history"], turn) for turn in turns]
        probe["rationale"] = rationale
        probe["expected_snapshot"] = {
            **probe["expected"],
            "selection": "unresolved" if probe["expected"]["action"] == "clarify" else "determinate",
            "valid_at_turn": probe["at_turn"],
            "source_refs": [f"m{span['seq']}:{span['start']}-{span['end']}" for span in probe["gold_evidence"]],
            "authority": "historical user statements; not operation consent or backend verification",
        }
    return case


def _regressions(length_stratified=False):
    suite = make_suite()["cases"]
    selected = [copy.deepcopy(suite[6 * i + (2 if length_stratified and i % 2 else 0)])
                for i in range(len(KINDS))]
    reasons = {
        "early_order": ("用户明确给出最早订单的编号，是非记忆依赖控制。", "原始首次登记顺序确定最早订单。"),
        "early_product": ("原始首次登记确定商品型号。", "后续登记没有改变最早商品。"),
        "correction": ("第一次更正已发生，第二次更正尚不可见。", "第二次更正覆盖第一次更正。"),
        "topic_switch": ("当前处理对象由最近一次明确选择确定。", "后续明确切换替代此前任务对象。"),
        "multi_hop": ("办公室订单与给同事的称呼通过两条原话关联。", "没有后续关系更正，原有关系仍适用。"),
        "temporal_update": ("目前仅出现第一次承诺日期。", "后来承诺明确替代旧日期，仅复述承诺。"),
        "negation": ("用户排除第一单并明确选择第二单。", "用户撤回第二单查询并明确改回第一单。"),
        "ambiguous": ("两单已登记，但用户明确未选择。", "后续仍未选定，单数指代不能唯一定位。"),
    }
    for case in selected:
        late = max(20, case["turns"] * 2 // 3)
        turns = {
            "early_order": ([1], [1]), "early_product": ([1], [1]),
            "correction": ([1, 9], [1, 9, late]), "topic_switch": ([17], [case["turns"] - 2]),
            "multi_hop": ([1, 3, 7, 11], [1, 3, 7, 11]),
            "temporal_update": ([8], [8, late]), "negation": ([10], [10, late]),
            "ambiguous": ([1, 3, 10], [1, 3, 10, late]),
        }
        case["cohort"] = "regression"
        variant = 2 if case["turns"] == 64 else 0
        case["origin"] = f"context_benchmark_cases.make_suite variant {variant}; history, probes and labels unchanged"
        _annotate(case, turns[case["kind"]], reasons[case["kind"]])
    return selected


def _transfer(index, kind):
    a, b, c, d = [str(730000 + index * 10 + offset) for offset in range(4)]
    pa, sa, ca = PRODUCTS[(index + 2) % len(PRODUCTS)]
    pb, sb, cb = PRODUCTS[(index + 4) % len(PRODUCTS)]
    history = _background(index)
    _put(history, 1, f"我先把第一张购买截图转成文字：订单{a}，商品是{pa}，型号{sa}。", "收到，已把第一张截图中的购买信息记在这段会话里。")
    _put(history, 4, f"另一张截图属于订单{b}，买的是{pb}，型号{sb}。两张截图对应不同的购买记录。", "好的，这张截图作为另一笔购买信息保存。")
    expected = [{"action": "logistics", "order_id": a, "sku": "", "date": ""} for _ in range(2)]
    queries = ["我刚进来时发的第一张购买截图，对应包裹走到哪里了？", "回到最先那张购买截图，请查询它对应订单的物流。"]
    evidence, reasons = ([1], [1]), ("第一张截图是首次登记的订单，后续未改变该顺序。", "查询最先截图指向同一笔原始订单。")
    if kind == "early_product":
        expected = [{"action": "care", "order_id": "", "sku": sa, "date": ""} for _ in range(2)]
        queries = ["第一张购买截图里的设备，日常清洁该怎么做？", "还是一开始那张截图的设备，请告诉我清洁方法。"]
        reasons = ("第一张截图记录了设备与型号的对应关系。", "最初设备的指代未改变，按原型号查说明。")
    elif kind == "correction":
        _put(history, 8, f"我对照付款详情发现，开头抄的{a}不是这次咨询的编号。付款详情里对应这件商品的号码是{c}，商品和型号没有变。", "收到，以这次核对付款详情得到的编号处理后续咨询，商品信息保持原样。")
        _put(history, 24, f"又发现刚才复制的{c}属于另一条记录。这件商品最终核对的号码是{d}，请把前面两次发的号码都从这次查询目标里排除。", "已记下最终核对结果。前两次编号不作为这次咨询的查询目标。")
        expected[0]["order_id"], expected[1]["order_id"] = c, d
        queries = ["请按我核对付款详情后留下的那个号码查物流。", "用这件商品最终核对的号码查一下包裹进度。"]
        evidence = ([1, 8], [1, 8, 24])
        reasons = ("付款详情纠正首次抄录，后一次核对尚未发生。", "最终核对明确排除了前两次号码。")
    elif kind == "topic_switch":
        _put(history, 16, f"接下来先看订单{a}的配送情况，另一张截图我晚点再问。", "好，这一阶段围绕第一张截图对应的配送咨询。")
        _put(history, 25, f"第一张截图的事情先放一边。接下来的咨询切到订单{b}，我想了解它的配送。", "收到，接下来围绕另一张截图对应的配送咨询。")
        expected[1]["order_id"] = b
        queries = ["我们现在说的这个包裹运到哪了？", "接着刚才切换的话题，查询这个包裹的物流。"]
        evidence = ([16], [25])
        reasons = ("最近明确讨论的配送对象是第一单。", "明确切换后的配送对象是第二单。")
    elif kind == "multi_hop":
        _put(history, 6, f"订单{a}留在我住处，订单{b}送到研发组的公共工位。公共工位收到的设备由小周保管。", "已记录两笔购买的去向及公共工位的保管人。")
        _put(history, 12, "我提到小周那台设备，指的就是送到公共工位的那件；留在住处的归我自己保管。", "好的，小周负责的是公共工位收到的设备。")
        expected[0]["order_id"] = expected[1]["order_id"] = b
        queries = ["小周负责的那台设备，所在订单的物流到哪了？", "请查送到公共工位、由小周保管那件的配送进展。"]
        evidence = ([1, 4, 6, 12], [1, 4, 6, 12])
        reasons = ("从保管人经公共工位关联到第二笔订单。", "保管人与用途关系没有发生后续变更。")
    elif kind == "temporal_update":
        _put(history, 7, f"人工客服给订单{a}补发附件时，书面答复承诺在2026-11-03到货。我只想保存这条承诺，尚未确认实际送达。", "已保存书面答复所述日期，实际运输与送达情况另行查询。")
        _put(history, 23, f"同一位专员发来后续答复：订单{a}附件改到2026-11-08到货，前一条答复里的日期不再适用。两条答复我都留着，以后面这条为准。", "已保存后续答复，这仍然是承诺日期，并不是到货确认。")
        expected = [{"action": "recall", "order_id": "", "sku": "", "date": date} for date in ("2026-11-03", "2026-11-08")]
        queries = ["截至刚才，人工书面承诺的附件到货日期是哪天？请用YYYY-MM-DD格式。", "专员后面那条有效答复承诺哪天到货？请用YYYY-MM-DD格式，只说承诺日期。"]
        evidence = ([7], [7, 23])
        reasons = ("目前只有第一条书面承诺可见。", "后续答复明确替代前一日期，不表示已经到货。")
    elif kind == "negation":
        _put(history, 9, f"订单{a}暂时别动，我只想知道{b}的配送走到哪了；这轮不查第一张截图。", "明白，这次配送咨询限定在第二笔购买。")
        _put(history, 26, f"{b}的查询先停一下，这回要看的改回{a}。上一条排除第一张截图的安排到此结束。", "收到，最新安排已替代此前的查询安排。")
        expected[0]["order_id"], expected[1]["order_id"] = b, a
        queries = ["请查我刚刚允许查询的那个包裹。", "就照我最后那条安排查物流，停下来的那笔先不查。"]
        evidence = ([9], [9, 26])
        reasons = ("第一单被明确排除，第二单是唯一查询对象。", "用户明确撤回原限制并把查询对象改回第一单。")
    elif kind == "ambiguous":
        _put(history, 10, "我还在比较这两张截图，没决定先查哪张。刚才登记两笔购买，只是提供背景，请等我说清楚对象。", "好的，真正查询前会先确认你指哪张截图。")
        _put(history, 27, "前面两张截图都还可能要问，我还没作出选择。不要把最近提到的那张自动当成查询对象。", "明白，仍然需要明确这次查询对应哪笔购买。")
        expected = [{"action": "clarify", "order_id": "", "sku": "", "date": ""} for _ in range(2)]
        queries = ["那就帮我看看那张截图的包裹进度吧。", "现在查一下那笔的物流吧。"]
        evidence = ([1, 4, 10], [1, 4, 10, 27])
        reasons = ("已知两笔购买而没有唯一指代，必须澄清。", "用户再次声明未选择，后续含糊指代仍不能确定对象。")
    case = {
        "id": f"transfer-{index + 1:02d}", "kind": kind, "cohort": "transfer", "turns": 32,
        "origin": "new deterministic fictional phrasing/relationship variants, frozen before model calls; not external blind holdout",
        "history": history,
        "probes": [{"id": f"p{i + 1}", "at_turn": at, "query": queries[i], "expected": expected[i]} for i, at in enumerate((18, 32))],
        "tool_data": {
            "orders": {oid: {"order_id": oid, "tracking_no": "SF" + oid + "88", "status": "运输中", "last_event": "包裹已到杭州分拨中心"} for oid in (a, b, c, d)},
            "manuals": {sa: ca, sb: cb},
        },
        "all_orders": [a, b, c, d], "all_skus": [sa, sb],
    }
    return _annotate(case, evidence, reasons)


def dataset(length_stratified=False):
    value = {
        "version": "context-matched-cases-v1", "cases": _regressions(length_stratified) + [_transfer(i, kind) for i, kind in enumerate(KINDS)],
        "scope": "16 synthetic histories across 8 families: 8 unchanged known regressions and 8 new phrasing/relationship variants, 2 correlated checkpoints each. No customer data or external blind holdout.",
        "metadata_policy": "expected, gold_evidence, expected_snapshot, rationale and tool_data are evaluator-only; memory sees visible history prefix + query. Tools receive only the model-selected action/arguments.",
        "selection": "All eight prior scenario families, variant 0 selected without regard to prior scores. New variants drafted from business semantics and scorer contract, without consulting new model outputs.",
        "limitations": ["Same eight scenario families as development history; transfer means new expression/relationship variants, not unseen business domains.", "Gold source spans are minimal labelled support sets, not an exhaustive set of all equivalent assistant restatements.", "Existing frozen rule scorer can reject valid negative references to obsolete identifiers; do not selectively revise labels or retry semantic errors."],
    }
    if length_stratified:
        value["version"] = "context-matched-cases-length-stratified-v2"
        value["selection"] = "Fixed family order early_order, early_product, correction, topic_switch, multi_hop, temporal_update, negation, ambiguous. Regression family indices 0/2/4/6 use existing variant 0 (24 turns); indices 1/3/5/7 use variant 2 (64 turns). Selection ignores prior scores. All eight transfer cases are byte-for-byte equivalent as JSON values to frozen v1. No new noise or answer changes."
        value["scope"] = "16 synthetic histories across 8 families: 8 known regressions stratified by predeclared family parity into 24/64 turns, and 8 unchanged 32-turn phrasing/relationship variants; 2 correlated checkpoints each. Length stratification is a known-regression control, not an external blind holdout."
    return value


def export(out, *, length_stratified=False):
    out = Path(out)
    if out.exists():
        raise ValueError("new dataset directory required; frozen cases are never overwritten")
    value = dataset(length_stratified)
    blob = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    out.mkdir(parents=True)
    (out / "dataset.json").write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    (out / "dataset.sha256").write_text(digest + "  dataset.json\n", encoding="utf-8")
    (out / "README.json").write_text(json.dumps({
        "file_sha256": digest, "case_count": len(value["cases"]), "probe_count": sum(len(c["probes"]) for c in value["cases"]),
        "scope": value["scope"], "metadata_policy": value["metadata_policy"], "limitations": value["limitations"],
        "generator": "scripts/context_matched_cases.py", "freeze": "Generated before comparison calls; changes require a new version and output directory.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--length-stratified", action="store_true", help="v2: select 24/64-turn known regressions by fixed family parity; keep v1 transfer cases unchanged")
    args = parser.parse_args()
    print(export(args.output, length_stratified=args.length_stratified))
