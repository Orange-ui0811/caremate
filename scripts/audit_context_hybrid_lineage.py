"""Independent raw-source/summary lineage audit, no model calls or rescoring."""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import messages_from_dict, message_to_dict

from scripts import eval_context_unified as unified


def audit(root):
    root=Path(root).resolve()
    if (root/"writer.lock").exists():raise ValueError("writer active")
    read=unified.read
    data=read(root/"dataset.json")
    cases={c["id"]:c for c in data["cases"]}
    issues=[]; count=0; legacy_replays=0; summary_checks=0
    for path in (root/"projections").glob("*.json"):
        for entry in read(path):
            coord=entry["coords"];arm,cid,n=coord["arm"],coord["case_id"],coord["turn"]
            inp=entry["assembler_input"]
            if arm=="legacy":
                state=deepcopy(inp["state"])
                state["messages"]=messages_from_dict(state.get("messages",[]))
                with patch.object(unified.base.settings,"context_policy","legacy"),patch.object(
                        unified.base.settings,"context_window_max_tokens",inp["context_window_max_tokens"]):
                    if entry["function"]=="history":
                        actual=unified.base.nodes._history_text(state,inp["max_turns"],stage=entry["stage"])
                        if actual!=entry["packet"]["history"]:issues.append("legacy history replay "+path.name)
                    else:
                        actual=unified.base.nodes._agent_messages(state,tool_schemas=inp["tool_schemas"])
                        if [message_to_dict(m) for m in actual]!=entry["packet"]["messages"]:
                            issues.append("legacy answer replay "+path.name)
                legacy_replays+=1
                continue
            seed=read(root/"seeds"/f"{arm}-{cid}.json")
            previous=read(root/"turns"/f"{arm}-{cid}-{n-1}.json") if n>1 else None
            prior=deepcopy(previous["actual_dialog_rows"] if previous else seed["rows"])
            snapshot=inp["snapshot"];rows=snapshot["rows"]
            normalized=lambda xs:[{k:r[k] for k in ("id","role","content")} for r in xs]
            if normalized(rows[:-1])!=normalized(prior):issues.append("prior conversation mismatch "+path.name)
            if rows[-1]["content"]!=cases[cid]["steps"][n-1]["query"]:
                issues.append("active original query mismatch "+path.name)
            m=entry["packet"]["manifest"]
            if arm in {"summary6","hybrid6"}:
                summary=previous["summary"] if previous else seed["summary"]
                upto=previous["summary_upto"] if previous else seed["summary_upto"]
                if (inp["summary"],inp["summary_upto"])!=(summary,upto):
                    issues.append("summary not from previous visible turn "+path.name)
                expected_recent=[r["id"] for r in prior if r["id"]>upto]
                if m["recent_message_ids"]!=expected_recent:issues.append("recent raw history omitted "+path.name)
                if any(i>upto for i in m["old_message_ids"]):issues.append("old retrieval crossed summary boundary "+path.name)
                if arm=="summary6" and m["old_message_ids"]:issues.append("control retrieved old history "+path.name)
                if m["old_units"]>6 or m["budget"] is not None:issues.append("simple contract mismatch "+path.name)
            selected=m["selected_message_ids"]
            if len(selected)!=len(set(selected)) or selected!=sorted(selected):issues.append("source duplication/order "+path.name)
            if not set(selected)<=set(r["id"] for r in prior):issues.append("unavailable source selected "+path.name)
            count+=1
    for path in (root/"maintenance").glob("*.json"):
        for m in read(path):
            if not m["triggered"]:continue
            calls=[read(root/"calls"/(i+".json")) for i in m["call_ids"]]
            if not calls or any(c["purpose"]!="conversation_summary" for c in calls):
                issues.append("maintenance purpose mismatch "+path.name)
            coord=m["coords"];arm,cid,n=coord["arm"],coord["case_id"],coord["turn"]
            if arm not in {"summary6","hybrid6"}:continue
            rows=(read(root/"turns"/f"{arm}-{cid}-{n}.json")["actual_dialog_rows"] if n else
                  read(root/"seeds"/f"{arm}-{cid}.json")["rows"])
            visible=[r for r in rows if r["id"]<=m["row_watermark"]]
            # Reconstruct production human prompt from source boundaries.
            users=[i for i,r in enumerate(visible) if r["role"]=="user"]
            boundary=visible[users[-6]-1]["id"]
            segment=[r for r in visible if m["previous_summary_upto"]<r["id"]<=boundary]
            dialog="\n".join(('用户' if r['role']=='user' else '客服')+':'+r['content'] for r in segment if r['content'])
            expected="旧摘要:"+(m["previous_summary"] or "(无)")+"\n\n需要并入摘要的对话:\n"+dialog
            wire_messages=(calls[0].get("wire_payload") or {}).get("messages",[])
            if not any(msg.get("content")==expected for msg in wire_messages):
                issues.append("summary wire omitted/changed original segment "+path.name)
            summary_checks+=1
    result={"passed":not issues,"issues":issues,"source_views":count,
            "legacy_replays":legacy_replays,"summary_wire_checks":summary_checks,
            "scope":"Original source prefix, no future or gold in assembler, full recent raw inclusion, previous-turn summary, exact maintenance prompt, and legacy context replay."}
    unified.write(root/"LINEAGE_AUDIT.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    result=audit(p.parse_args().output);print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["passed"]:raise SystemExit(1)
