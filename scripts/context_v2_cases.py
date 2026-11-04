"""Frozen synthetic context-only acceptance workload; no customer conversations."""


def cases():
    # required maps user-requested task kinds to their independently labelled objects.
    specs = [
        ('A1','刚刚的订单是哪个，它现在物流到哪了，可以退货吗',['1001'],{'order':'1001','logistics':'1001','knowledge':'1001'},0),
        ('A2','刚刚那个订单现在送到哪了，里面是什么商品，有没有使用说明',['1001'],{'order':'1001','logistics':'1001','knowledge':'1001'},0),
        ('A3','那一单现在到哪儿了，买的是什么',['1001'],{'order':'1001','logistics':'1001'},0),
        ('A4','它现在送到哪里了，商品怎么清理',['1001'],{'logistics':'1001','knowledge':'1001'},0),
        ('B1','最开始确认的那笔订单现在到哪了，里面是什么商品',['1001'],{'order':'1001','logistics':'1001'},0),
        ('B2','刚刚更正的那单现在送到哪里了，里面是什么商品',['1001','2002'],{'order':'2002','logistics':'2002'},0),
        ('B3','回顾刚才讨论了哪笔订单，别再办理退款或投诉',['1001'],{},0),
        ('B4','刚刚那个订单现在送到哪了，里面是什么商品',[],{'order':None,'logistics':None},1),
        ('C1','刚才这两笔中，那个订单到哪了，里面是什么商品',['1001','2002'],{'order':None,'logistics':None},1),
        ('C2','查订单2002物流，我要给另一笔订单退款',['1001'],{'logistics':'2002','refund':None},1),
        ('C3','查订单1001状态，另外猫粮怎么保存',['1001'],{'order':'1001','knowledge':None},0),
        ('C4','查刚刚那单物流，平台通用运费规则是什么',['1001'],{'logistics':'1001','knowledge':None},0),
        ('D1','这笔订单到哪了，里面是什么商品，有没有使用说明',[],{'order':None,'logistics':None,'knowledge':None},1),
        ('D2','这笔订单到哪了，里面是什么商品，有没有使用说明，另外我要投诉商家',[],{'order':None,'logistics':None,'knowledge':None,'complaint':None},1),
        ('D3','这笔订单到哪了，里面是什么商品',[],{'order':None,'logistics':None},1),
        ('D4','不是1001，改查订单2002物流和商品信息',['1001'],{'order':'2002','logistics':'2002'},0),
        ('E1','这笔订单到哪了，里面是什么商品，有没有使用说明',[],{'order':None,'logistics':None,'knowledge':None},1),
        ('E2','这笔订单到哪了，里面是什么商品',[],{'order':None,'logistics':None},1),
        ('E3','请为订单1001申请质量问题退款，并登记投诉商家的工单',[],{'refund':'1001','complaint':None},0),
        ('E4','这笔订单到哪了，里面是什么商品',[],{'order':None,'logistics':None},1),
        ('F1','查刚才那单的商品，商品有没有使用说明',['1001'],{'order':'1001','knowledge':'1001'},0),
        ('F2','这笔订单到哪了，这个商品有没有使用说明',[],{'logistics':None,'knowledge':None},1),
        ('F3','查订单1001状态，它的火星维修站地址是什么',['1001'],{'order':'1001','knowledge':'1001'},0),
        ('F4','这笔订单到哪了，这个商品有没有使用说明',[],{'logistics':None,'knowledge':None},1),
    ]
    out=[]
    for key, raw, focus, required, selections in specs:
        out.append(dict(id=key,raw=raw,focus=focus,required=required,selections=selections,
            action={'D2':'cancel','D3':'invalid_selection','E1':'restart','E2':'duplicate_resume',
                    'E3':'lost_commit','E4':'supersede','F4':'order_failure'}.get(key,'complete'),
            knowledge='gate_may_refuse' if key not in {'F3','F4'} else {'F3':'must_refuse','F4':'must_not_retrieve'}[key]))
    return out
