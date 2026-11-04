"""Synthetic customer-service replay cases with labels unavailable to memory builders."""
import random

PRODUCTS = [
    ("自动喂食器", "MH-FD10", "清洁前断电，食盆可拆下用温水清洗，主机只用软布擦拭。"),
    ("自动饮水机", "MH-W40", "清洁前断电，水箱用温水清洗，水泵外壳用软布擦拭。"),
    ("看护摄像头", "MH-CAM1", "清洁前断电，镜头用干燥软布擦拭，禁止浸水。"),
    ("智能猫砂盆", "MH-LP100", "清洁前断电，集便仓拆下清洗，主机用软布擦拭。"),
    ("宠物暖垫", "MH-HEAT8", "清洁前断电，外套拆下清洗，发热内芯用软布擦拭。"),
    ("自动逗猫机", "MH-PLAY6", "清洁前断电，外壳用软布擦拭，绳头拆下清洗。"),
]
KINDS = ("early_order", "early_product", "correction", "topic_switch",
         "multi_hop", "temporal_update", "negation", "ambiguous")
FILLERS = [
    ("包装上的防拆标识破损时应该先做什么？我只是咨询一般流程。", "可以先拍摄外包装和标识，保留收货凭证，再通过售后渠道说明情况。是否属于运输问题需要结合具体记录判断，本次没有提交售后申请。"),
    ("商品尺寸里长宽高的顺序怎么理解？先不用查具体型号。", "通常按商品页面标注的长宽高顺序核对，安装空间还要考虑插头和开盖距离。不同页面的标注可能不同，应查看相应说明。"),
    ("电子发票和纸质说明书有什么区别？", "电子发票是交易票据，说明书介绍使用和维护方法。两者用途不同，需要分别保存；此处没有申请发票，也没有变更购买信息。"),
    ("快递代收点和送货上门一般有什么区别？", "代收点需要收件人领取，送货上门按物流约定投递。具体服务范围以承运方和地址条件为准，本次只是解释概念。"),
    ("赠品是不是也要查看包装清单？", "建议核对赠品是否在对应商品清单里，并保留开箱记录。是否随订单附送需要查看下单页面，不能仅凭一般说明确认。"),
    ("购买前怎么区分商品详情和买家评价？", "商品详情提供商家公布的规格和规则，评价反映购买者的使用体验。应结合实际需求核对信息，不能把评价当作统一售后承诺。"),
    ("说明书里的警告和日常维护提示需要分别看吗？", "两部分都应阅读。警告说明禁止事项和风险，维护提示说明日常检查和保养步骤，具体操作仍以相应型号的说明书为准。"),
    ("退货包装一般要保留哪些东西？现在不办理退货。", "可以保留原包装、配件和购买凭证。不同商品要求可能不同，申请前还需查看对应退货规则；这里没有发起或批准退货。"),
    ("预约配送前需要核对地址吗？不用改我的地址。", "通常需要核对联系人、联系电话、地区和详细门牌。这里只说明通用步骤，不读取或修改地址，也不确认具体配送时间。"),
    ("保修卡应该和购物凭证放在一起吗？", "建议一起保存，申请服务时可能需要核对购买渠道和产品信息。保修范围依具体条款确定，这里没有确认任何商品的保修资格。"),
    ("商品页面里的材质和颜色应该怎样核对？", "可以查看规格选项、详情说明和实物展示；拍摄光线可能影响颜色观感。此处只是通用建议，没有替换前面登记的商品。"),
    ("收到包裹后可以先核对配件再安装吗？", "可以先按清单检查配件和外观，确认齐全后再按照说明安装。发现异常时保留照片并联系售后，不建议自行改装。"),
]


def make_suite():
    """48 histories, eight scenario families, two causal checkpoints per history."""
    cases = []
    for kind_index, kind in enumerate(KINDS):
        for variant, turns in enumerate((24, 40, 64, 24, 40, 64)):
            number = len(cases) + 1
            rng = random.Random(81931 + number)
            a, b, c, d = [str(610000 + number * 10 + i) for i in range(4)]
            pa, sa, ca = PRODUCTS[variant]
            pb, sb, cb = PRODUCTS[(variant + 1) % len(PRODUCTS)]
            target_manual = a if variant % 2 else b
            first_date, last_date = "2026-10-21", "2026-10-27"
            history = []
            for t in range(1, turns + 1):
                q, answer = rng.choice(FILLERS)
                # Extra independent question keeps the long histories genuinely over a small window.
                q2, answer2 = rng.choice(FILLERS)
                history.extend([{"role": "user", "content": q + "另外，" + q2},
                                {"role": "assistant", "content": answer + answer2}])

            def put(turn, user, assistant):
                history[2 * turn - 2:2 * turn] = [
                    {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]

            put(1, f"先登记这笔订单{a}，商品是{pa}，型号{sa}。这是本次最早提到的商品。",
                f"已登记订单{a}与型号{sa}的对应关系。")
            put(3, f"再登记订单{b}，商品{pb}，型号{sb}。两笔分别记录。",
                f"已另外登记订单{b}与型号{sb}，不混用两笔信息。")
            late = max(20, turns * 2 // 3)
            if kind == "correction":
                put(9, f"刚才的订单号{a}报错了，应为{c}；商品还是{pa}，型号{sa}，旧号不用了。",
                    f"订单号更正为{c}，商品型号不变。")
                put(late, f"再次核对后，正确订单号是{d}，之前更正成{c}也不对。以后以{d}为准。",
                    f"以最后核准的订单{d}为准，前两个号码失效。")
            elif kind == "topic_switch":
                put(17, f"现在先处理订单{a}，另一单稍后说。", f"目前关注订单{a}。")
                put(turns - 2, f"换个话题，现在只查订单{b}，暂时不处理之前那单。", f"当前切换为订单{b}。")
            elif kind == "multi_hop":
                other = b if target_manual == a else a
                put(7, f"订单{target_manual}是我给办公室买的，订单{other}是给家里买的；办公室那款要寄给同事。",
                    "两笔订单的使用地点分别登记，不混淆。")
                put(11, "后面我说给同事那款，就是办公室那一款，不是家里的。", "已记录这个称呼指向。")
            elif kind == "temporal_update":
                put(8, f"订单{a}补寄配件的承诺到货日是{first_date}，帮我记一下。",
                    f"当前承诺日期记录为{first_date}。")
                put(late, f"专员刚改了承诺，订单{a}配件到货日以{last_date}为准，前面那个日期作废。",
                    f"已按更新记录为{last_date}，这是专员的承诺，不是系统查询结果。")
            elif kind == "negation":
                put(10, f"不要查询订单{a}，我现在要处理的是{b}，别因为先提到就选错。", f"将按要求处理订单{b}。")
                put(late, f"改一下安排：先别查{b}了，这次只查{a}。", f"本次按最新安排处理订单{a}。")
            elif kind == "ambiguous":
                put(10, "两笔订单都还没有确定先处理哪笔，我暂时只是登记，别默认挑一单。", "需要操作时请再确认具体订单。")
                put(late, "目前仍没有选定订单，两单都有可能要查。", "查询前需要确认目标订单。")

            probes = []
            for index, at in enumerate((18, turns)):
                target, sku, action = a, sa, "logistics"
                query = ("帮我查最早登记的那笔订单物流。", "前面第一笔订单现在运到哪里了？")[index]
                if kind == "early_product":
                    action = "care"
                    query = ("最开始提到那款商品，清洁时怎么处理？", "前面最早说的那款该怎么清洗？")[index]
                elif kind == "correction":
                    target = c if index == 0 else d
                    query = "按最后更正的订单号查询物流，别使用作废号码。"
                elif kind == "topic_switch":
                    target, sku = (a, sa) if index == 0 else (b, sb)
                    query = "就查我们目前正在处理的这笔订单物流。"
                elif kind == "multi_hop":
                    target = target_manual
                    sku = sa if target == a else sb
                    query = "帮我查给同事那款所在订单的物流。"
                elif kind == "temporal_update":
                    action = "recall"
                    query = "专员最后承诺的配件到货日期是哪天？请按YYYY-MM-DD格式告诉我。"
                elif kind == "negation":
                    target, sku = (b, sb) if index == 0 else (a, sa)
                    query = "按我最新明确指定的那笔查物流，别查我说暂时不要查的。"
                elif kind == "ambiguous":
                    action, target, sku = "clarify", "", ""
                    query = "查一下那单的物流。"
                # A few explicit queries provide non-memory-dependent controls.
                if variant == 0 and kind == "early_order" and index == 0:
                    query = f"帮我查订单{a}的物流。"
                expected = {"action": action, "order_id": target if action == "logistics" else "",
                            "sku": sku if action == "care" else "",
                            "date": (first_date if index == 0 else last_date) if action == "recall" else ""}
                probes.append({"id": f"p{index + 1}", "at_turn": at, "query": query, "expected": expected})
            orders = {oid: {"order_id": oid, "tracking_no": "SF" + oid + "88", "status": "运输中",
                            "last_event": "包裹已到杭州分拨中心"} for oid in (a, b, c, d)}
            cases.append({"id": f"svc-{number:02d}", "kind": kind, "turns": turns,
                          "history": history, "probes": probes,
                          "tool_data": {"orders": orders, "manuals": {sa: ca, sb: cb}},
                          "all_orders": [a, b, c, d], "all_skus": [sa, sb]})
    return {"version": "service-context-v2", "seed": 81931, "cases": cases,
            "scope": "Synthetic fixed-history replay, 8 scenario templates x 6 variants; no real customer records."}
