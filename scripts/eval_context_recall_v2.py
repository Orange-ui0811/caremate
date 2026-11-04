"""All-visible +/-1 recall with per-node dedup: two new arms, frozen summary control."""
from __future__ import annotations
import argparse
import asyncio
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import message_to_dict, messages_from_dict
from app.core import memory
from app.core.history_recall import dialogue_supplement, persistent_ids, VERSION
from scripts import eval_context_keep3 as previous

unified, base = previous.unified, previous.base
ROOT = unified.ROOT
PARENT = ROOT/"data/ch07/context-keep3/20261027-formal-v1"
TASK_REFERENCE = ROOT/"data/ch07/context-full-retrieval/20261028-formal-v1"
ARMS = ("retrieval","hybrid")
LABELS = {"summary":"滚动摘要＋滑窗（冻结对照）",
          "retrieval":"全可见历史回查＋前后邻轮＋节点去重＋滑窗",
          "hybrid":"滚动摘要＋滑窗＋全可见回查＋前后邻轮＋节点去重"}
COMMON = {**previous.COMMON,"context_legacy_recall_mode":"all_visible_neighbors"}
EXTRAS = ["scripts/eval_context_recall_v2.py","scripts/audit_context_recall_v2.py",
          "scripts/analyze_context_keep3.py","tests/test_history_recall_v2.py"]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dataset(pilot=False):
    return unified.read((ROOT/"data/ch07/context-keep3/20261027-pilot-v1" if pilot else PARENT)/"dataset.json")

def effective_config():
    return {"environment":base.config(),"overrides":COMMON,"thinking":base.OFF,"failover":False,
        "python_hash_seed":os.environ.get("PYTHONHASHSEED","unset"),"selector":VERSION,
        "recall":"all visible prior Q&A, user-text BM25 top3, previous+next units, dedup only message IDs already visible to the actual node; no entity expansion",
        "inputs":"unchanged legacy coref/intent last6 messages, answer original3000 window, query rewriting query-only; same summary prompt30/keep3",
        "summary_control":"reuse previous frozen summary arm; no new summary-only model calls",
        "parent_dataset_sha256":sha(PARENT/"dataset.json"),"parent_analysis_sha256":sha(PARENT/"ANALYSIS.json"),
        "task_reference_analysis_sha256":sha(TASK_REFERENCE/"ANALYSIS.json")}

def replay_view(state,arm,function,max_turns=6,tool_schemas=None):
    if arm not in ARMS or function not in {"history","answer"}:raise ValueError("unknown view")
    with ExitStack() as stack:
        for key,val in {"context_policy":"legacy","context_window_max_tokens":3000,
                        "context_legacy_recall_mode":"all_visible_neighbors","context_legacy_recall_enabled":False}.items():
            stack.enter_context(patch.object(base.settings,key,val))
        def view():
            if function=="history":return {"history":previous.ORIGINAL_HISTORY(state,max_turns),"messages":[]}
            messages=previous.ORIGINAL_AGENT(state,tool_schemas=tool_schemas)
            return {"history":"\n".join(str(m.content) for m in messages),"messages":[message_to_dict(m) for m in messages]}
        unchanged=view()
        if function=="history":
            window=memory.build_window(state.get("messages",[]),state.get("summary_upto_msg_id") or 0,3000)
            visible=persistent_ids(window[:-1][-max_turns:])
        else:visible=persistent_ids(messages_from_dict(unchanged["messages"]))
        supplement=dialogue_supplement(state,visible_message_ids=visible)
        with patch.object(base.settings,"context_legacy_recall_enabled",True):actual=view()
        return {"packet":actual,"base_packet":unchanged,"supplement":supplement,"visible_message_ids":visible}

@contextmanager
def binding():
    with ExitStack() as stack:
        for key,val in {"ARMS":ARMS,"LABELS":LABELS,"COMMON":COMMON,"dataset":dataset,
                        "effective_config":effective_config,"replay_view":replay_view,"audit":audit,
                        "EXTRA_SOURCES":list(dict.fromkeys(previous.EXTRA_SOURCES+EXTRAS))}.items():
            stack.enter_context(patch.object(previous,key,val))
        with previous.binding():yield

def prepare(out,pilot=False):
    control=unified.read(PARENT/"manifest.json")["config"]["environment"]
    if any(base.config()[k]!=control[k] for k in ("chat_model","chat_base_url")):
        raise ValueError("model or endpoint differs from controls")
    with binding():manifest=previous.prepare(out,pilot=pilot,arms=ARMS)
    manifest.update(version="all-visible-neighbors-node-dedup-v1",labels=LABELS,
        comparison="Two new arms: same node inputs and batch window; summary on/off. Summary-only control reused, not rerun. Separate single runs, not identical generated histories.",
        recall=effective_config()["recall"],summary_control={"output":str(PARENT),"arm":"summary","rerun":False},
        token_scope="all actual node calls, summary initialization and maintenance, and recorded retries; scripted history generation and real RAG/business costs excluded")
    unified.write(Path(out)/"manifest.json",manifest)
    return manifest

def audit(out):
    out=Path(out).resolve()
    result=previous.ORIGINAL_AUDIT(out)
    result['labels']=LABELS
    result['limits']=["Two newly run arms, reused frozen summary-only and other historical references.",
        "Same frozen synthetic inputs and rule scorer, independently generated outputs; no semantic reruns.",
        "All-visible top3 plus previous/next units, ID dedup against each node's actual original window, no entity expansion.",
        "Unknown usage is an unknown cost lower bound, not zero."]
    unified.write(out/"audited_report.json",result)
    return result

def finalize(out):
    from scripts.audit_context_recall_v2 import audit as independent
    from scripts import analyze_context_keep3 as analyzer
    out=Path(out).resolve();generic=audit(out);detail=independent(out)
    with patch.object(analyzer,'_write_report',lambda *args:None):analyzer.analyze(out)
    child=unified.read(out/"ANALYSIS.json");parent=unified.read(PARENT/"ANALYSIS.json")
    manifest=unified.read(out/"manifest.json")
    config=manifest['config']
    if sha(PARENT/'dataset.json')!=config['parent_dataset_sha256'] or sha(PARENT/'ANALYSIS.json')!=config['parent_analysis_sha256']:
        raise ValueError("frozen control changed")
    formal=not manifest['pilot']
    if formal and unified.read(out/'dataset.json')!=unified.read(PARENT/'dataset.json'):
        raise ValueError("cases or gold changed")
    groups={"summary":deepcopy(parent['groups']['summary']),**child['groups']} if formal else child['groups']
    for arm,g in groups.items():g['label']=LABELS[arm]
    refs={}
    if formal:
        if sha(TASK_REFERENCE/'ANALYSIS.json')!=config['task_reference_analysis_sha256']:
            raise ValueError("task reference changed")
        task=unified.read(TASK_REFERENCE/'ANALYSIS.json')
        refs={"previous_hybrid":parent['groups']['hybrid'],"previous_complete_retrieval":task['groups']['task'],"full":parent['groups']['full']}
    comparisons={}
    for arm,g in groups.items():
        if arm=='hybrid':continue
        a,b=groups.get('hybrid',{}).get('overall',{}),g['overall']
        if not a:continue
        ca,cb=a['cost'],b['cost']
        comparisons[arm]={"task_delta":a['task']['pass']-b['task']['pass'],"episode_delta":a['episodes']['pass']-b['episodes']['pass'],
            "token_change_percent":100*(ca['total_tokens']/cb['total_tokens']-1) if ca['cost_complete'] and cb['cost_complete'] and cb['total_tokens'] else None}
    result={"formal":formal,"new_arms":list(ARMS),"groups":groups,"historical_references":refs,"hybrid_vs":comparisons,
        "same_cases_and_gold":True,"summary_control_reused":formal,"independent_passed":detail['passed'],
        "source_audit":detail,"generic_audit":generic,"limits":manifest['comparison']}
    unified.write(out/'COMPARISON.json',result)
    rate=lambda m:f"{m['pass']}/{m['denominator']}（{m['rate']:.1%}）" if m.get('rate') is not None else f"{m['pass']}/{m['denominator']}"
    cost=lambda c,k:('≥' if not c['cost_complete'] else '')+f"{c[k]:,}"
    lines=['# 全可见回查、前后邻轮与节点去重实验','',f"独立审计通过：{detail['passed']}。",'',
        '正式只新跑无摘要回查组和摘要混合组；摘要＋滑窗读取冻结对照，没有重复模型调用。' if formal else '独立预检，不计入正式结果。','',
        '摘要规则30条消息触发、维护后留3轮；coref/intent最近6条消息，回答原3000估算Token窗口，查询改写仍不加历史。新回查对全部可见历史做BM25，取Top3及前后邻轮；按节点已有原文的消息ID去重，不按文本相似删除，不增加模型规划。','',
        '|方法|任务通过|整段三问全过|全部输入|全部输出|全部总Token|摘要调用/总Token|','|---|---:|---:|---:|---:|---:|---:|']
    for g in groups.values():
        o=g['overall'];c=o['cost'];s=o['summary_cost']
        lines.append(f"|{g['label']}|{rate(o['task'])}|{rate(o['episodes'])}|{cost(c,'input_tokens')}|{cost(c,'output_tokens')}|{cost(c,'total_tokens')}|{s['model_calls']} / {cost(s,'total_tokens')}|")
    lines+=['','|方法|更正任务|限制目标查询|歧义澄清|','|---|---:|---:|---:|']
    for g in groups.values():lines.append('|'+g['label']+'|'+'|'.join(rate(g['metrics'][k]) for k in ('correction_task','restriction_target_query','clarification_task'))+'|')
    lines+=['','## 分层结果','','|方法|子集|任务通过|整段全过|总Token|','|---|---|---:|---:|---:|']
    for g in groups.values():
        for field in ('stratum','cohort'):
            for name,v in g[field].items():lines.append(f"|{g['label']}|{name}|{rate(v['task'])}|{rate(v['episodes'])}|{cost(v['cost'],'total_tokens')}|")
    lines+=['','## 已有系统参照（冻结结果，非本次重跑）','','|方法|任务通过|整段全过|总Token|','|---|---:|---:|---:|']
    for g in refs.values():
        o=g['overall'];lines.append(f"|{g['label']}|{rate(o['task'])}|{rate(o['episodes'])}|{cost(o['cost'],'total_tokens')}|")
    lines+=['','Token包含本工作负载内所有记录调用、初始化/维护及重试；不含预置历史生成与真实检索/业务费用。≥为未知usage下界。无摘要组按同一节奏本地推进原文边界，不调用摘要。Top3邻轮最多9个单元，去重可能只补回一个单元中节点缺少的消息。','',
        '这是同一内部合成数据与评分的单次补充对照，前轮实际回答独立生成，有错误传播；控制结果来自先前运行，不能当作同时多重复实验。规则通过不等于完整语义或线上准确率。','',
        'AUDIT_DETAIL.json核验来源、节点已有原文去重、选择器精确回放及维护；ANALYSIS.json保留全部失败。']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {"complete":generic['complete'],"audit_passed":detail['passed'],"issues":detail['issues'],
            "groups":{k:{'task':g['overall']['task'],'episodes':g['overall']['episodes'],'cost':g['overall']['cost']} for k,g in groups.items()}}

def main():
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true')
    m=p.add_mutually_exclusive_group();m.add_argument('--run',action='store_true');m.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.run:
        with binding():r=asyncio.run(unified.run(a.output))
    elif a.audit:r=finalize(a.output)
    else:
        v=prepare(a.output,a.pilot);r={'frozen':True,'output':str(a.output),'sources':len(v['sources']),'dataset_digest':v['dataset_sha256']}
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
