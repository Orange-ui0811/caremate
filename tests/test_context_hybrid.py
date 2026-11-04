from copy import deepcopy
import json

import httpx
import pytest

from app.core.context_state import build_snapshot
from app.core.context_hybrid_simple import assemble_simple_context
from scripts import eval_context_hybrid as hybrid


def source(pairs, query="工作室那单物流呢"):
    rows = []
    for q,a in pairs:
        rows.extend([{"id":len(rows)+1,"role":"user","content":q},
                     {"id":len(rows)+2,"role":"assistant","content":a}])
    rows.append({"id":len(rows)+1,"role":"user","content":query})
    return build_snapshot(rows, 123)


def test_old_user_match_includes_answer_and_following_correction():
    s = source([("工作室那单710001", "已登记"), ("不对，应该是710003", "收到更正"),
                ("包装怎么保管", "保持干燥")])
    p = assemble_simple_context(s, "工作室那单物流呢", summary="工作室有一笔订单", summary_upto=4)
    assert p["manifest"]["old_message_ids"] == [1,2,3,4]
    assert p["manifest"]["recent_message_ids"] == [5,6]
    assert "收到更正" in p["history"]
    assert p["manifest"]["retrieval_candidates"][0]["message_id"] == 1


def test_assistant_words_do_not_score_no_positive_hit_no_padding():
    s = source([("包装要保持干燥", "工作室那单710001"), ("保修卡保管", "收到")])
    p = assemble_simple_context(s, "工作室那单", summary="背景", summary_upto=2)
    assert not p["manifest"]["old_message_ids"]


def test_keep_all_recent_no_token_trimming_and_summary_control():
    s = source([("工作室那单710001", "已登记"), ("包装"*9000,"不要裁剪这句话")])
    for retrieval in (False, True):
        p = assemble_simple_context(s,"工作室那单",summary="背景",summary_upto=2,retrieval=retrieval)
        assert p["manifest"]["recent_message_ids"] == [3,4]
        assert "包装"*9000 in p["history"] and not p["blocked"]
        assert p["manifest"]["budget"] is None
        assert bool(p["manifest"]["old_message_ids"]) == retrieval


def test_sources_boundary_watermark_and_max_six_old_units():
    pairs = [(f"工作室那单{i+710001}", "已记录") for i in range(10)]
    s = source(pairs)
    p = assemble_simple_context(s,"工作室那单",summary="背景",summary_upto=16)
    assert p["manifest"]["old_units"] <= 6
    ids=p["manifest"]["selected_message_ids"]
    assert ids==sorted(set(ids))
    with pytest.raises(ValueError,match="splits"):
        assemble_simple_context(s,"工作室",summary="背景",summary_upto=1)
    s["watermark"]=2
    with pytest.raises(ValueError,match="watermark"):
        assemble_simple_context(s,"工作室")


def test_new_cases_frozen_targets_and_no_gold_in_history():
    data=hybrid.context_hybrid_cases.dataset()
    assert len(data["cases"])==16
    for c in data["cases"][-4:]:
        assert len(c["initial_history"])==64 and len(c["steps"])==3
        assert "expected" not in json.dumps(c["initial_history"])


async def test_real_graph_mock_http_all_arms_and_summary_boundary(tmp_path,monkeypatch):
    impl=hybrid.base
    monkeypatch.setattr(impl.settings,"model_pool_enabled",False)
    monkeypatch.setattr(impl.settings,"chat_base_url","https://ai.gitee.com/v1")
    monkeypatch.setattr(impl.settings,"chat_model","fake-offline-only")
    case=deepcopy(hybrid.context_hybrid_cases.dataset()["cases"][-4])
    case["id"]="offline-summary-boundary"
    monkeypatch.setattr(hybrid.context_hybrid_cases,"dataset",lambda pilot=False:{"cases":[case],"pilot":True})
    requests=[]

    def reply(request):
        payload=json.loads(request.content); requests.append(payload)
        purpose,turn=impl.PURPOSE.get(),impl.COORDS.get()["turn"]
        text=("请查询订单710004最新物流。" if purpose=="coreference" else
              f"订单710004，物流记录{turn}：{hybrid.unified.cases_module().logistics_event(max(1,turn))}。")
        message={"role":"assistant","content":text}
        arguments={"intent_classification":{"intent":"物流","confidence":1.0},
                   "conversation_summary":{"summary":"唯一待处理订单710004，尚未核实当前物流。"}}.get(purpose)
        if arguments:
            message={"role":"assistant","content":"","tool_calls":[{"id":"offline-call","type":"function",
                "function":{"name":payload["tools"][0]["function"]["name"],"arguments":json.dumps(arguments,ensure_ascii=False)}}]}
        usage={"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}
        result={"id":"offline","object":"chat.completion","created":1,"model":"fake-offline-only",
            "choices":[{"index":0,"message":message,"finish_reason":"tool_calls" if arguments else "stop"}],"usage":usage}
        if not payload.get("stream"):
            return httpx.Response(200,json=result)
        delta=deepcopy(message)
        if arguments: delta["tool_calls"][0]["index"]=0
        chunk={"id":"offline","object":"chat.completion.chunk","created":1,"model":"fake-offline-only",
            "choices":[{"index":0,"delta":delta,"finish_reason":"tool_calls" if arguments else "stop"}],"usage":usage}
        return httpx.Response(200,content=("data: "+json.dumps(chunk,ensure_ascii=False)+"\n\ndata: [DONE]\n\n").encode(),headers={"content-type":"text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        def factory(**kwargs):
            return impl.llm._CumulativeUsageChatOpenAI(model="fake-offline-only",base_url="https://ai.gitee.com/v1",
                api_key="offline-placeholder",http_async_client=client,max_retries=0,
                streaming=kwargs.get("streaming",False),stream_usage=True)
        monkeypatch.setattr(impl.llm,"get_chat_model",factory)
        hybrid.prepare(tmp_path,pilot=True)
        with hybrid.binding():
            report=await hybrid.unified.run(tmp_path)
    assert report["complete"] and report["audit_passed"],report["issues"]
    for arm in hybrid.ARMS:
        assert report["groups"][arm]["task_rule_pass"]==3
        assert report["groups"][arm]["interface_error_calls"]==0
    for arm in ("summary6","hybrid6"):
        items=hybrid.unified.read(tmp_path/"maintenance"/f"{arm}-{case['id']}.json")
        triggered=[m for m in items if m["triggered"]]
        assert [m["row_watermark"] for m in triggered]==[30,48,66]
        assert [m["legacy_summary_upto"] for m in triggered]==[18,36,54]
    assert sum(g["model_calls"] for g in report["groups"].values())==len(requests)
