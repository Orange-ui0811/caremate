"""Audit original routing and persisted context; no production policy changes."""
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.core import memory
from app.core.llm import get_chat_model
from app.core.runtime_settings import get_evidence_confidence_threshold
from app.db import base, repository
from app.graph import nodes
from app.tools import registry

OUTPUT = Path("data/backend-investigation-20261011/root-cause-results.json")


async def inspect(saver, cid):
    checkpoint = await saver.aget_tuple({"configurable": {"thread_id": str(cid)}})
    state = checkpoint.checkpoint["channel_values"]
    rows = await repository.list_dialog_messages(cid)
    return state, {
        "intent": state.get("intent"), "route": state.get("route"),
        "trace": state.get("trace"),
        "model_history": [{"role": m.type, "content": m.content,
                           "tools": getattr(m, "tool_calls", [])} for m in state.get("messages", [])],
        "display_history": [{"role": r.role, "content": r.content} for r in rows],
    }


async def main():
    report = {"production_rules": "original, restored byte-for-byte",
              "knowledge_gate_threshold": get_evidence_confidence_threshold(),
              "scenarios": {}, "context_comparison": {}}
    def save():
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        async with AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path) as saver:
            async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=120) as client:
                scenarios = {
                    "product_price_then_policy": ["玩具多少钱", "退货政策是什么"],
                    "recall_then_policy": ["刚刚我们聊了什么", "退货政策是什么"],
                    "policy_as_first_question": ["退货政策是什么"],
                    "missing_knowledge_then_policy": ["猫粮怎么选", "退货政策是什么"],
                }
                for name, questions in scenarios.items():
                    cid = None
                    uid = f"root-audit-{uuid4().hex[:12]}"
                    turns = []
                    for question in questions:
                        response = await client.post("/api/chat", json={"user_id": uid,
                            "message": question, "conversation_id": cid})
                        response.raise_for_status()
                        events = [json.loads(line[6:]) for line in response.text.splitlines()
                                  if line.startswith("data: ") and line[6:] != "[DONE]"]
                        cid = next(e["conversation_id"] for e in events if e.get("event") == "done")
                        _, diagnostic = await inspect(saver, cid)
                        turn = {"question": question,
                                "answer": "".join(e.get("delta", "") for e in events),
                                "tools": [e["name"] for e in events if e.get("event") == "tool"],
                                **diagnostic}
                        turns.append(turn)
                        report["scenarios"][name] = {"conversation_id": cid, "turns": turns}
                        save()
                        print(json.dumps({"scenario": name, "cid": cid, "question": question,
                            "answer_start": turn["answer"][:110], "tools": turn["tools"],
                            "trace": turn["trace"],
                            "checkpoint_roles": [m["role"] for m in turn["model_history"]],
                            "db_roles": [m["role"] for m in turn["display_history"]]}, ensure_ascii=False), flush=True)
            # Replay the actual original user's model context locally. Only read state;
            # two model calls request no business operation and execute no tools.
            state, diagnostic = await inspect(saver, 4)
            last_human = max(i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage))
            messages = state["messages"][:last_human + 1]
            replies = {}
            user_id = None
            for row in await repository.list_dialog_messages(4):
                if row.role == "user":
                    user_id = row.id
                elif row.role == "assistant" and row.content:
                    replies[user_id] = row
            completed = []
            missing = []
            for i, message in enumerate(messages):
                completed.append(message)
                if not isinstance(message, HumanMessage) or i == len(messages) - 1:
                    continue
                end = next((j for j in range(i + 1, len(messages)) if isinstance(messages[j], HumanMessage)), len(messages))
                row = replies.get(memory._db_msg_id(message))
                if row and not any(isinstance(m, AIMessage) for m in messages[i + 1:end]):
                    completed.append(AIMessage(row.content, id=f"db-{row.id}"))
                    missing.append({"question": message.content, "actual_reply": row.content})
            registry.scan_builtin()
            specs = await registry.get_all_specs()
            for name, history in (("original_missing_replies", messages), ("actual_replies_restored_in_memory_only", completed)):
                model = get_chat_model(streaming=True, temperature=0).bind_tools([s.tool for s in specs])
                ai = await model.ainvoke(nodes._agent_messages({**state, "messages": history, "steps": 0}))
                report["context_comparison"][name] = {"conversation_id": 4,
                    "production_state_modified": False, "reply": ai.content, "requested_tools": ai.tool_calls,
                    "context_roles": [m.type for m in history]}
                save()
                print(json.dumps({"comparison": name, "reply": ai.content, "tools": ai.tool_calls}, ensure_ascii=False), flush=True)
            report["context_comparison"]["missing_original_replies"] = missing
            save()
    finally:
        await base.engine.dispose()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
