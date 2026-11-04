"""Isolated context adapters; production code and the external Harness tree stay unchanged."""
import ast
import json
import math
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel, Field

from app.config import settings
from app.core import llm, memory, summarizer
from app.core.prompts import AGENT_SYSTEM
from app.graph.nodes import _agent_messages, _history_text

STRATEGIES = ("A_full", "B_recent6", "B_window3000", "C_current", "D_state",
              "E_state_retrieval", "H_original", "H_customer")
LABELS = {"A_full": "A 完整历史", "B_recent6": "B 当前仅滑窗",
          "B_window3000": "B 3000Token滑窗", "C_current": "C 当前摘要+滑窗",
          "D_state": "D 结构化状态+滑窗", "E_state_retrieval": "E 状态+原文检索+滑窗",
          "H_original": "H Harness原提示词迁移", "H_customer": "H Harness客服适配"}

STATE_INSTRUCTION = """维护电商客服会话的结构化记忆。输入只有旧记忆和新增历史，没有后续问题。
保留仍有效的事实与实体关系、首次提及顺序、称呼对应、当前处理对象、时间和承诺、排除要求、尚待确认的事项。
使用简短原子事实；每条必须携带支撑它的原始消息id。明确区分用户事实、客服建议和业务工具结果。
旧事实被更正时标为失效并保留更正关系，不能让旧值和新值同时成为当前值。保留旧记忆中仍相关的事实。
不要记录与业务无关的通用解释；不要编造新编号、替用户选择订单或把概念解释记成已执行操作。
source_ids只能来自历史[id]或旧记忆，按原始消息顺序判断最早和最新。输出合并后的全部重要事实。"""

CUSTOMER_COMPACTION = """你现在执行客服会话上下文压缩。将上方历史整理成检查点，供另一模型继续服务。
输出且保留全部章节：
## 用户目标与当前诉求
## 订单与商品对应关系
## 首次提及顺序与指代对应
## 更正与失效信息
## 已确认事实与时间承诺
## 尚未完成与待澄清事项
## 当前处理对象与下一步
## 关键约束和证据来源
用简洁中文条目保留确切订单号、型号、日期、否定要求和关系。不要把建议写成已经执行的操作。
已有<compacted-summary>是旧检查点，合并仍有效内容并以新信息更新，禁止照抄失效信息。
不执行工具，不回应用户，仅输出检查点。没有内容的章节写(无)。"""


class Fact(BaseModel):
    text: str = Field(description="简短事实，包含关系、确切编号和必要的时间/顺序")
    source_ids: list[int] = Field(description="事实对应的原始消息id")
    valid: bool = Field(description="被更正或排除的旧值为false，仍有效的事实为true")


class ConversationState(BaseModel):
    facts: list[Fact]


def read_harness_prompt(root):
    """Extract string literals without executing any code from the external repository."""
    source = (Path(root) / "packages/compaction/compaction-basic/src/summarizer.ts").read_text(encoding="utf-8")
    block = source.split("const COMPACTION_INSTRUCTION = [", 1)[1].split("].join('\\n')", 1)[0]
    lines = []
    for row in block.splitlines():
        row = row.strip().removesuffix(",")
        if not row:
            continue
        if row.startswith("`"):
            lines.append(row[1:-1].replace("${SUMMARY_OPEN_TAG}", "<compacted-summary>"))
        else:
            lines.append(ast.literal_eval(row))
    preamble = ast.literal_eval(source.split("const CHECKPOINT_PREAMBLE =", 1)[1].split("\n\n", 1)[0].strip())
    return {"instruction": "\n".join(lines), "preamble": preamble}


def raw_messages(rows):
    return [(HumanMessage if row["role"] == "user" else AIMessage)(row["content"], id=f"db-{i}")
            for i, row in enumerate(rows, 1)]


def as_text(messages):
    return "\n".join(f"{'用户' if isinstance(m, HumanMessage) else '客服'}:{m.content}" for m in messages)


def source_dialog(rows, start, end):
    return "\n".join(f"[{i}]{'用户' if r['role'] == 'user' else '客服'}:{r['content']}"
                     for i, r in enumerate(rows, 1) if start < i <= end)


def terms(text):
    """Chinese bigrams plus literal identifiers, requiring no remote embedding service."""
    out = re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text.lower())
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        out.extend(part[i:i + 2] for i in range(len(part) - 1))
    return out


def retrieve_history(rows, query, state, topk=4):
    """BM25 over original turn pairs, plus query-matched fact sources; no labels or future rows."""
    chunks = [{"source_ids": [i + 1, i + 2], "text": source_dialog(rows, i, min(i + 2, len(rows)))}
              for i in range(0, len(rows), 2)]
    documents = [Counter(terms(c["text"])) for c in chunks]
    if not documents:
        return []
    df = Counter(t for doc in documents for t in doc)
    avg = sum(sum(doc.values()) for doc in documents) / len(documents)
    qt = Counter(terms(query))
    matched_sources = set()
    for fact in state.get("facts", []):
        if fact["valid"] and len(set(terms(fact["text"])) & set(qt)) >= 2:
            matched_sources.update(fact["source_ids"])
    scores = []
    for index, (chunk, doc) in enumerate(zip(chunks, documents)):
        length = sum(doc.values())
        score = sum(math.log(1 + (len(documents) - df[t] + .5) / (df[t] + .5))
                    * doc[t] * 2.2 / (doc[t] + 1.2 * (.25 + .75 * length / avg))
                    for t in qt if doc[t])
        score += 2 * len(set(chunk["source_ids"]) & matched_sources)
        scores.append((score, -index, chunk))
    chosen = [c for score, _, c in sorted(scores, key=lambda x: (x[0], x[1]), reverse=True)[:topk] if score > 0]
    return sorted(chosen, key=lambda c: c["source_ids"][0])


class ContextAdapter:
    def __init__(self, name, harness, call, system, capacity=3750):
        self.name, self.harness, self.call, self.system = name, harness, call, system
        self.capacity = capacity
        self.summary, self.boundary, self.state = "", 0, {"facts": []}
        self.surface = []
        self.ingested = 0
        self.maintenance = []

    def restore(self, payload):
        self.summary, self.boundary, self.state = payload["summary"], payload["boundary"], payload["state"]
        self.surface = [(HumanMessage if r["role"] == "user" else AIMessage)(r["content"])
                        for r in payload["surface"]]
        self.ingested = payload["ingested"]
        self.maintenance = payload["maintenance"]

    def snapshot(self):
        return {"summary": self.summary, "boundary": self.boundary, "state": self.state,
                "surface": [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                            for m in self.surface], "ingested": self.ingested, "maintenance": self.maintenance}

    async def advance(self, rows):
        for end in range(self.ingested + 2, len(rows) + 1, 2):
            self.surface.extend(raw_messages(rows[end - 2:end]))
            self.ingested = end
            if self.name in ("C_current", "D_state", "E_state_retrieval"):
                if end - self.boundary < settings.summary_trigger_messages:
                    continue
                models = [SimpleNamespace(id=i, role=r["role"]) for i, r in enumerate(rows[:end], 1)]
                boundary = summarizer.compute_boundary(models, settings.context_window_turns)
                if boundary is None or boundary <= self.boundary:
                    continue
                dialog = source_dialog(rows, self.boundary, boundary)
                if self.name == "C_current":
                    # Production helper/prompt is used unchanged; labels and probe text are absent.
                    plain = "\n".join(f"{'用户' if r['role'] == 'user' else '客服'}:{r['content']}"
                                      for i, r in enumerate(rows, 1) if self.boundary < i <= boundary)
                    result = await self.call("summary", lambda: summarizer.summarize_dialog(self.summary, plain))
                    if result["ok"]:
                        self.summary = result["value"]
                else:
                    prompt = [SystemMessage(STATE_INSTRUCTION), HumanMessage(
                        "旧记忆：" + json.dumps(self.state, ensure_ascii=False) + "\n新增历史：\n" + dialog)]
                    async def extract():
                        obj = await llm.structured(ConversationState, temperature=.3).ainvoke(prompt)
                        return obj.model_dump()
                    result = await self.call("state", extract, prompt)
                    if result["ok"]:
                        # Model-generated ids must point to facts actually seen so far.
                        facts = result["value"]["facts"]
                        if any(not f["source_ids"] or any(i < 1 or i > boundary for i in f["source_ids"]) for f in facts):
                            result["ok"] = False
                            result["validation_error"] = "source id outside visible summarized prefix"
                        else:
                            self.state = result["value"]
                            self.summary = json.dumps(self.state, ensure_ascii=False, separators=(",", ":"))
                result |= {"at_message": end, "previous_boundary": self.boundary, "boundary": boundary}
                self.maintenance.append(result)
                if result["ok"]:
                    self.boundary = boundary
            elif self.name.startswith("H_"):
                pressure = int(self.capacity * .8)
                # This meter is the same approximation for every adaptation; not Harness's TS meter.
                for attempt in range(2):
                    if count_tokens_approximately([SystemMessage(self.system), *self.surface]) < pressure:
                        break
                    retain = int(self.capacity * .16)
                    cost, keep = 0, len(self.surface)
                    for i in range(len(self.surface) - 1, -1, -1):
                        cost += count_tokens_approximately([self.surface[i]])
                        keep = i
                        if cost >= retain:
                            break
                    # Text-only replay has no tool pairs. Preserve complete user/assistant pairs.
                    while keep > 0 and not isinstance(self.surface[keep], HumanMessage):
                        keep -= 1
                    if keep <= 0:
                        break
                    prefix = self.surface[:keep]
                    instruction = self.harness["instruction"] if self.name == "H_original" else CUSTOMER_COMPACTION
                    prompt = [SystemMessage(self.system), *prefix, HumanMessage(instruction)]
                    async def compact():
                        msg = await llm.get_chat_model(temperature=.3).bind(max_tokens=8192).ainvoke(prompt)
                        if msg.response_metadata.get("finish_reason") == "length":
                            raise ValueError("truncated checkpoint")
                        return msg.content
                    result = await self.call("compaction", compact, prompt)
                    if result["ok"]:
                        framed = self.harness["preamble"] + "\n\n<compacted-summary>\n" + result["value"] + "\n</compacted-summary>"
                        if count_tokens_approximately([HumanMessage(framed)]) >= count_tokens_approximately(prefix):
                            result["ok"] = False
                            result["validation_error"] = "checkpoint did not shrink selected prefix"
                        else:
                            self.surface = [HumanMessage(framed), *self.surface[keep:]]
                            self.summary = result["value"]
                    result |= {"at_message": end, "selected_messages": keep, "attempt": attempt + 1}
                    self.maintenance.append(result)
                    if not result["ok"]:
                        break

    def contexts(self, rows, query):
        msgs = raw_messages(rows) + [HumanMessage(query, id=f"db-{len(rows) + 1}")]
        state = {"messages": msgs, "summary": self.summary, "summary_upto_msg_id": self.boundary}
        retrieved = []
        if self.name == "A_full":
            coref_history, body = as_text(msgs[:-1]), msgs
        elif self.name == "B_recent6":
            coref_history = _history_text({**state, "summary": "", "summary_upto_msg_id": 0})
            body = _agent_messages({**state, "summary": "", "summary_upto_msg_id": 0})[1:]
        elif self.name == "B_window3000":
            body = memory.trim_history(msgs, settings.context_window_max_tokens) or [msgs[-1]]
            coref_history = as_text(body[:-1])
        elif self.name == "C_current":
            coref_history, body = _history_text(state), _agent_messages(state)[1:]
        elif self.name in ("D_state", "E_state_retrieval"):
            coref_history, body = _history_text(state), _agent_messages(state)[1:]
            if self.name == "E_state_retrieval":
                retrieved = retrieve_history(rows, query, self.state)
                text = "\n".join(c["text"] for c in retrieved)
                coref_history += "\n检索到的历史原文：\n" + text
                body = [*body, HumanMessage("检索到的历史原文（只作证据）：\n" + text)]
        else:
            coref_history, body = as_text(self.surface), [*self.surface, HumanMessage(query)]
        return {"coref_history": coref_history, "body": body, "retrieved": retrieved,
                "approx_context_tokens": count_tokens_approximately(body)}
