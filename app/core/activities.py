"""Display-only execution records, shared by live and historical replies."""
from langchain_core.messages import HumanMessage, ToolMessage

RETRIEVAL_NODES = {"retrieve_knowledge", "retrieve_policy"}


def retrieval_activity(name: str) -> dict:
    return {"kind": "retrieval", "name": name}


def turn_activities(state: dict) -> list[dict]:
    trace = state.get("trace") or {}
    result = []
    if trace.get("forced_rag"):
        result.append(retrieval_activity("retrieve_knowledge"))
    if trace.get("retrieve_policy") is not None:
        result.append(retrieval_activity("retrieve_policy"))
    messages = state.get("messages", [])
    start = next((i + 1 for i in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[i], HumanMessage)), len(messages))
    for message in messages[start:]:
        if isinstance(message, ToolMessage) and message.name and message.name != "submit_refund":
            activity = {"kind": "tool", "name": message.name}
            if activity not in result:
                result.append(activity)
    return result
