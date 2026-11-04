"""Offline paired analysis; never modifies raw scores, calls or source records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")


def percentile(values,q):
    if not values:return None
    seq=sorted(values); f=(len(seq)-1)*q; i=int(f)
    return seq[i]+(seq[min(i+1,len(seq)-1)]-seq[i])*(f-i)


def analyze(root):
    root=Path(root).resolve()
    if (root/"writer.lock").exists():raise ValueError("evaluation still running")
    audit=read(root/"audited_report.json")
    if not audit["complete"] or not audit["audit_passed"]:raise ValueError("complete audit required")
    manifest,data=read(root/"manifest.json"),read(root/"dataset.json")
    cases={c["id"]:c for c in data["cases"]}
    turns=[read(p) for p in (root/"turns").glob("*.json")]
    calls=[read(p) for p in (root/"calls").glob("*.json")]
    indexed={(r["coords"]["arm"],r["coords"]["case_id"],r["coords"]["turn"]):r for r in turns}
    groups={}
    for arm in manifest["arms"]:
        rs=[r for r in turns if r["coords"]["arm"]==arm]
        cs=[c for c in calls if c["coords"]["arm"]==arm]
        def stats(chosen):
            ids={r["coords"]["case_id"] for r in chosen}
            cc=[c for c in cs if c["coords"]["case_id"] in ids]
            return {"pass":sum(r["score"]["task_rule_pass"] for r in chosen),"turns":len(chosen),
                "input_tokens":sum((c.get("usage") or {}).get("input_tokens",0) for c in cc),
                "output_tokens":sum((c.get("usage") or {}).get("output_tokens",0) for c in cc)}
        g={"overall":stats(rs),"episode_all_pass":sum(all(indexed[(arm,cid,n)]["score"]["task_rule_pass"]
            for n in range(1,len(cases[cid]["steps"])+1)) for cid in cases),"episodes":len(cases)}
        for field in ("cohort","stratum","family"):
            g[field]={v:stats([r for r in rs if cases[r["coords"]["case_id"]].get(field,"pilot")==v])
                for v in sorted({c.get(field,"pilot") for c in cases.values()})}
        g["position"]={str(n):{"pass":sum(r["score"]["task_rule_pass"] for r in rs if r["coords"]["turn"]==n),
                               "turns":sum(r["coords"]["turn"]==n for r in rs)} for n in (1,2,3)}
        purposes=defaultdict(lambda:{"calls":0,"input_tokens":0,"output_tokens":0,"unknown_usage":0})
        for c in cs:
            p=purposes[c.get("purpose","unknown")];p["calls"]+=1
            p["unknown_usage"]+=not bool(c.get("usage"))
            for key in ("input_tokens","output_tokens"):p[key]+=(c.get("usage") or {}).get(key,0)
        g["purposes"]=dict(purposes)
        g["latency"]={field:{"p50":percentile([r[field] for r in rs if r.get(field) is not None],.5),
                             "p95":percentile([r[field] for r in rs if r.get(field) is not None],.95)}
                      for field in ("end_to_end_seconds","foreground_complete_seconds","first_visible_sse_delta_seconds")}
        g["cached_input_tokens"]=sum((c.get("usage") or {}).get("input_token_details",{}).get("cache_read",0) or 0 for c in cs)
        groups[arm]=g
    paired={}
    for other in manifest["arms"]:
        if other=="hybrid6":continue
        outcomes=Counter(); details=[]
        for cid,c in cases.items():
            for n in range(1,len(c["steps"])+1):
                left,right=indexed[("hybrid6",cid,n)],indexed[(other,cid,n)]
                a,b=left["score"]["task_rule_pass"],right["score"]["task_rule_pass"]
                key="both_pass" if a and b else "both_fail" if not a and not b else "hybrid_only" if a else "reference_only"
                outcomes[key]+=1
                if a!=b:details.append({"case_id":cid,"turn":n,"outcome":key,"query":left["query"]})
        paired[other]={"counts":dict(outcomes),"discordant":details}
    failures=[]
    for r in turns:
        if not r["score"]["task_rule_pass"]:
            s=r.get("state",{})
            failures.append({"coords":r["coords"],"query":r["query"],"expected":r["expected"],
                "failed_checks":[k for k,v in r["score"]["checks"].items() if not v],"error":r.get("error"),
                "resolved_query":s.get("resolved_query"),"intent":s.get("intent"),"route":s.get("route"),
                "summary":r.get("summary"),"answer":r["answer"],
                "tools":[{"name":t["name"],"args":t["args"]} for t in r.get("executed_tools",[])]})
    diagnostic=[]
    for path in (root/"projections").glob("hybrid6-*.json"):
        for item in read(path):
            if item["stage"]!="coref":continue
            m=item["manifest"];cid=item["coords"]["case_id"];n=item["coords"]["turn"]
            target=cases[cid]["steps"][n-1]["expected"].get("target","")
            inp=item["assembler_input"];rows={r["id"]:r["content"] for r in inp["snapshot"]["rows"]}
            diagnostic.append({"coords":item["coords"],"query":inp["query"],"target":target,
                "old_units":m["old_units"],"recent_units":m["recent_units"],
                "top_matches":m["retrieval_candidates"][:3],
                "target_literal_in_summary":bool(target and target in inp["summary"]),
                "target_literal_in_old":bool(target and any(target in rows[mid] for mid in m["old_message_ids"])),
                "target_literal_in_recent":bool(target and any(target in rows[mid] for mid in m["recent_message_ids"])),
                "note":"Literal presence is diagnostic only, not correctness of relation, recency or authority."})
    result={"groups":groups,"paired_hybrid_vs":paired,"failures":failures,"coref_diagnostics":diagnostic,
            "audit":{k:audit[k] for k in ("complete","audit_passed","projection_replays")},
            "raw_hashes":{str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ("turns","calls","seeds","maintenance","projections") for p in (root/folder).glob("*.json")}}
    write(root/"ANALYSIS.json",result)
    lines=["# 滚动摘要与BM25回查：正式分析","",
        "真实模型与既有Workflow；业务和知识为隔离合成夹具。结果为单次内部任务规则通过，不是线上准确率。","",
        "|方案|任务通过|三轮全过|含维护总Token|摘要Token|模型调用|",
        "|---|---:|---:|---:|---:|---:|"]
    for arm,g in groups.items():
        ag=audit["groups"][arm];o=g["overall"]
        lines.append(f"|{manifest['labels'][arm]}|{o['pass']}/{o['turns']}|{g['episode_all_pass']}/{g['episodes']}|{ag['total_tokens']:,}|{ag['maintenance_input_tokens']+ag['maintenance_output_tokens']:,}|{ag['model_calls']}|")
    lines += ["","## 对照解释","",
        "summary6与hybrid6只在旧问答检索上有设计差异，但摘要独立生成且后续历史独立演进；小样本差异需逐条核对。",
        "legacy保留8轮及原裁剪；task保留此前完整选择器及预算，两者是系统参照。新两组不按Token裁剪，不是同预算四组对照。"]
    for other,p in paired.items():
        total=audit["groups"]["hybrid6"]["total_tokens"];ref=audit["groups"][other]["total_tokens"]
        lines.append(f"- hybrid6 相对 {other}：总Token变化 {(total/ref-1)*100:+.1f}%；配对结果 {p['counts']}。")
    for field,title in (("cohort","旧回归与新挑战"),("stratum","短长分层")):
        lines += ["",f"## {title}","","|方案|类别|通过|总Token|","|---|---|---:|---:|"]
        for arm,g in groups.items():
            for value,s in g[field].items():lines.append(f"|{arm}|{value}|{s['pass']}/{s['turns']}|{s['input_tokens']+s['output_tokens']:,}|")
    lines += ["","## 首问与连续追问","","|方案|首问|第二问|第三问|","|---|---:|---:|---:|"]
    for arm,g in groups.items():
        cells=[f"{g['position'][str(n)]['pass']}/{g['position'][str(n)]['turns']}" for n in (1,2,3)]
        lines.append("|"+arm+"|"+"|".join(cells)+"|")
    lines += ["","## 完整性与局限","",
        f"审计通过：{audit['audit_passed']}；上下文离线复算 {audit['projection_replays']} 次。",
        "成本含预置历史摘要初始化、三次真实追问全部节点、维护和重试；不含预置原问答生成以及真实RAG与业务服务费用。",
        "维护等待是测试屏障，不能把端到端耗时直接当生产后台维护时的响应延迟。",
        "12条已见回归＋4条新增长会话，每条3问；后续追问可能由上一轮回答带回对象，不能当作独立的远距离检索题。",
        "纯指代若无法唯一确定对象，应澄清；词面命中不等于关系理解，摘要也可能遗漏或混淆。",
        "输出Token与输入Token总和不是账单金额，缓存折扣未换算。",
        ""]
    for arm,g in audit["groups"].items():
        lines.append(f"- {arm}：接口错误 {g['interface_error_calls']}，运行失败轮 {g['failed_turns']}，摘要维护失败 {g.get('maintenance_failures',0)}，未知usage {g['unknown_usage_calls']}。")
    lines += ["","## 失败记录","","完整原文与失败检查位于 ANALYSIS.json；不改分、不重跑语义错误。"]
    for f in failures:
        c=f["coords"]
        lines += ["",f"- {c['arm']} / {c['case_id']} / 第{c['turn']}问：{f['query']}",
                  f"  未通过：{', '.join(f['failed_checks'])}；消解：{f['resolved_query']}；意图：{f['intent']}；路径：{f['route']}。"]
    (root/"FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    return {"groups":{k:v["overall"] for k,v in groups.items()},"failures":len(failures)}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    print(json.dumps(analyze(p.parse_args().output),ensure_ascii=False,indent=2))
