"""知识库录入 API:切块预览是 dry-run、录入按指纹查重、参数白名单挡住乱传。

重点锁两条容易回归的口径:
  1. 预览不写库——按了预览再看库存,一块都不该多;
  2. 同一份正文重复录入全跳过,而同一节切出的多块不许被当成重复误杀。
"""
import pytest
from datetime import datetime, timedelta
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import kb as kb_api
from app.db import repository

MD = """# 会员权益

## 运费与包邮

满 99 元包邮,未满收 10 元运费。偏远地区运费 20 元,不参与包邮。

## 发货时效

现货 48 小时内发货,预售按商品页标注的时间发。
"""


async def _noop() -> None:
    """向量化的替身:采纳这条路要验的是入库闸,不是嵌入模型通不通。"""
    return None


async def _offline_milvus() -> dict:
    """Milvus 探活的替身:测试不连真集合,免得给本地库留下测试痕迹。"""
    return {"online": False, "count": None, "detail": "stub"}


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(kb_api.router)
    return a


@pytest.fixture
async def client(app, db_session_factory):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def test_preview_is_dry_run(client):
    resp = await client.post("/api/kb/preview", json={"text": MD, "content_type": "policy"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2                       # 两个二级标题 → 两节两块
    assert body["features"]["sections"] == 2
    assert body["chunks"][0]["section_path"] == "会员权益 / 运费与包邮"
    assert body["chunks"][0]["is_key_clause"] is True   # 命中「运费/包邮」
    assert body["duplicates"] == 0 and body["dedup_known"] is True
    assert await repository.knowledge_stats() == {          # 预览不写库
        "total": 0, "pending": 0, "done": 0, "by_content_type": {}, "key_clause": 0}


async def test_preview_rejects_bad_input(client):
    assert (await client.post("/api/kb/preview", json={"text": "   "})).status_code == 400
    assert (await client.post(
        "/api/kb/preview", json={"text": MD, "content_type": "乱写"})).status_code == 400
    # file 只认建库材料清单里的文件名,路径穿越连不上门
    resp = await client.post("/api/kb/preview", json={"file": "../../etc/passwd"})
    assert resp.status_code == 400


async def test_ingest_then_reingest_is_idempotent(client):
    first = await client.post("/api/kb/ingest",
                             json={"text": MD, "content_type": "policy", "vectorize": False})
    assert first.status_code == 200
    body = first.json()
    assert body["inserted"] == 2 and body["skipped"] == 0
    assert body["vectorized"] is None               # 没勾向量化就只落 pending
    stats = await repository.knowledge_stats()
    assert stats["total"] == 2 and stats["pending"] == 2 and stats["done"] == 0

    again = await client.post("/api/kb/ingest",
                              json={"text": MD, "content_type": "policy", "vectorize": False})
    assert again.json()["inserted"] == 0 and again.json()["skipped"] == 2
    assert (await repository.knowledge_stats())["total"] == 2


async def test_ingest_keeps_multiple_pieces_of_one_section(client):
    """同一节切出的多块共用节标题,只按问法查重会把它们误杀成重复——这里锁住不许。"""
    rows = "\n".join(f"| 型号 {i} | 参数 {i} | 备注 {i} |" for i in range(24))
    md = ("# 商品规格\n\n## 猫爬架\n\n| 型号 | 参数 | 备注 |\n|---|---|---|\n" + rows + "\n")
    resp = await client.post("/api/kb/ingest",
                             json={"text": md, "content_type": "spec", "vectorize": False})
    body = resp.json()
    assert body["chunks"] > 1                        # 大表格按行拆成多块
    assert body["inserted"] == body["chunks"] and body["skipped"] == 0
    pairs = await repository.list_chunk_pairs()
    assert len({q for q, _ in pairs}) == 1           # 问法全是「猫爬架」
    assert len(pairs) == body["chunks"]              # 但每块都留下了


async def test_vectorize_endpoint_uses_dualwrite(client, monkeypatch):
    """向量化按钮与 make kb-vectorize 调同一个函数;这里只验证接线,不真跑嵌入。"""
    called = {}

    async def fake(*a, **kw):
        called["hit"] = True
        return 7
    monkeypatch.setattr("app.kb.dualwrite.vectorize_pending", fake)
    monkeypatch.setattr(kb_api, "milvus_state", _offline_milvus)
    resp = await client.post("/api/kb/vectorize")
    assert resp.status_code == 200 and resp.json()["vectorized"] == 7 and called["hit"]


async def test_search_strategy_whitelist(client):
    assert (await client.post("/api/kb/search", json={"q": " "})).status_code == 400
    resp = await client.post("/api/kb/search", json={"q": "邮费是多少", "strategy": "玄学"})
    assert resp.status_code == 400


async def test_search_returns_hits(client, monkeypatch):
    async def fake(query, strategy="hybrid_rerank", top_k=None, **kw):
        assert (query, strategy, top_k) == ("邮费是多少", "vector", 3)
        return [{"id": 1, "question": "运费与包邮", "answer": "满 99 元包邮", "score": 0.83,
                 "section_path": "会员权益 / 运费与包邮", "content_type": "policy",
                 "category": "会员权益"}]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake)
    resp = await client.post("/api/kb/search",
                             json={"q": "邮费是多少", "strategy": "vector", "top_k": 3})
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert len(hits) == 1 and hits[0]["question"] == "运费与包邮"


async def test_overview_survives_milvus_offline(client, monkeypatch):
    """Milvus 没起不该让整页塌:双写一致回 None(读不到),材料清单与作业状态照样给。"""
    monkeypatch.setattr(kb_api, "milvus_state", _offline_milvus)
    body = (await client.get("/api/kb/overview")).json()
    assert body["milvus"]["online"] is False
    assert body["consistent"] is None                # 读不到 ≠ 对不上
    assert body["db_error"] is None
    assert [s["file"] for s in body["sources"]]      # 材料清单是本地文件,不依赖任何服务
    assert {j["name"] for j in body["jobs"]} == set(kb_api.KB_JOBS)


@pytest.fixture
async def browse_rows(client):
    ids = []
    for i in range(25):
        ids.append(await repository.insert_knowledge_chunk(
            "商品" if i < 24 else "飞轮沉淀", f"商品问法{i}", f"正文{i}",
            section_path=f"商品 / 章节{i}", content_type="faq"))
    long_answer = "完整知识\n" + "详细说明。" * 150 + "\n全文末尾标记"
    policy = await repository.insert_knowledge_chunk(
        "售后", "退货说明\n折扣100%\n型号A_B", long_answer,
        section_path="售后 / 退货条件", content_type="policy", is_key_clause=1)
    await repository.mark_chunk_vectorized(policy, str(policy))
    legacy = await repository.insert_knowledge_chunk("旧资料", "未标注类型的知识", "旧正文")
    return ids, policy, legacy, long_answer


async def test_chunks_pagination_reaches_old_content_and_clamps_missing_page(client, browse_rows):
    ids, policy, legacy, _ = browse_rows
    first = (await client.get("/api/kb/chunks")).json()
    second = (await client.get("/api/kb/chunks", params={"page": 2})).json()
    assert first["total"] == second["total"] == 27
    assert len(first["items"]) == 20 and len(second["items"]) == 7
    all_ids = [r["id"] for r in first["items"] + second["items"]]
    assert all_ids == sorted(ids + [policy, legacy], reverse=True)
    assert len(set(all_ids)) == 27
    beyond = (await client.get("/api/kb/chunks", params={"page": 999})).json()
    assert beyond["page"] == 2 and beyond["items"] == second["items"]


@pytest.mark.parametrize("q", ["全文末尾标记", "退货说明", "售后", "退货条件", "100%", "A_B"])
async def test_chunks_keyword_searches_full_text_and_metadata(client, browse_rows, q):
    _, policy, _, _ = browse_rows
    body = (await client.get("/api/kb/chunks", params={"q": " " + q + " "})).json()
    assert body["total"] == 1 and body["items"][0]["id"] == policy


async def test_chunks_filters_and_count_use_same_conditions(client, browse_rows):
    _, policy, _, _ = browse_rows
    filters = {"category": "售后", "content_type": "policy", "vectorize_status": "done",
               "is_key_clause": "true"}
    body = (await client.get("/api/kb/chunks", params=filters)).json()
    assert body["total"] == 1 and body["items"][0]["id"] == policy
    pending = (await client.get("/api/kb/chunks", params={"vectorize_status": "pending",
                                                        "is_key_clause": "false", "page_size": 100})).json()
    assert pending["total"] == 26 and len(pending["items"]) == 26
    assert all(r["status"] == "pending" and not r["is_key_clause"] for r in pending["items"])
    empty = (await client.get("/api/kb/chunks", params=filters | {"q": "不存在"})).json()
    assert empty["total"] == 0 and empty["items"] == [] and empty["page"] == 1


async def test_chunks_literal_wildcards_and_categories(client, browse_rows):
    _, policy, _, _ = browse_rows
    for q in ["%", "_"]:
        body = (await client.get("/api/kb/chunks", params={"q": q})).json()
        assert body["total"] == 1 and body["items"][0]["id"] == policy
    categories = (await client.get("/api/kb/categories")).json()["items"]
    assert set(categories) == {"商品", "飞轮沉淀", "售后", "旧资料"}
    assert len(categories) == 4


async def test_detail_returns_complete_answer_neighbors_and_aware_times(client, browse_rows):
    ids, policy, _, long_answer = browse_rows
    await repository.set_chunk_neighbors(policy, ids[0], ids[1])
    listing = (await client.get("/api/kb/chunks", params={"category": "售后"})).json()["items"][0]
    assert len(listing["answer_preview"]) == 200
    assert listing["answer_length"] == len(long_answer)
    assert "answer" not in listing
    detail = (await client.get(f"/api/kb/chunks/{policy}")).json()
    assert detail["answer"] == long_answer and detail["answer"].endswith("全文末尾标记")
    assert detail["prev_chunk_id"] == ids[0] and detail["next_chunk_id"] == ids[1]
    assert detail["vector_id"] == str(policy)
    for field in ["created_at", "updated_at"]:
        assert datetime.fromisoformat(detail[field]).utcoffset() == timedelta(0)
    assert (await client.get("/api/kb/chunks/99999999")).status_code == 404


@pytest.mark.parametrize("params", [{"page": 0}, {"page_size": 0}, {"page_size": 101},
    {"q": "长" * 201}, {"content_type": "未知"}, {"vectorize_status": "未知"}, {"is_key_clause": "未知"}])
async def test_chunks_reject_invalid_parameters(client, params):
    assert (await client.get("/api/kb/chunks", params=params)).status_code == 422


async def test_content_apis_report_database_failure(client, monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("test database unavailable")
    for name in ["browse_knowledge_chunks", "knowledge_categories", "get_knowledge_chunk"]:
        monkeypatch.setattr(repository, name, fail)
    for path in ["/api/kb/chunks", "/api/kb/categories", "/api/kb/chunks/1"]:
        response = await client.get(path)
        assert response.status_code == 503 and "暂时无法读取" in response.json()["detail"]


async def test_录入页给得出补丁式重嵌这条作业(client):
    """改了 data/kb/*.md 之后要能在页面上按 kb-repatch,不必回终端。

    这条作业和 kb-reset 的区别是页面上唯一容易点错的地方:reset 会清掉飞轮写回的块,
    repatch 只把正文变了的块原地改掉(id 不变,Milvus 同 id upsert 覆盖)。
    """
    body = (await client.get("/api/kb/overview")).json()
    jobs = {j["name"]: j for j in body["jobs"]}
    assert "kb-repatch" in jobs
    assert jobs["kb-repatch"]["cmd"] == "make kb-repatch"      # 页面按的就是终端那条命令
    assert not jobs["kb-repatch"].get("heavy")                 # 秒级,不该弹二次确认
    assert jobs["kb-reset"]["heavy"] is True                   # 清库才是重活


async def test_挖出来的条目要人点采纳才进知识库(client, monkeypatch):
    """挖知识的入库口只有这一个,而且只认待审状态。

    kb-mine 自己不写库——模型归纳出来的问答对质量参差,得人过一眼。所以这条断言的是
    闸本身:待审时库里没有,采纳后才有。
    """
    sid = await repository.insert_staging("mine-t", "conv:1", "运费怎么算", "满99包邮")
    await repository.set_staging_status([sid], "kept")
    assert (await repository.knowledge_stats())["total"] == 0    # 待审阶段库里一条没有

    monkeypatch.setattr(kb_api.dualwrite, "vectorize_pending", lambda: _noop())
    resp = await client.post("/api/kb/staging/approve", json={"ids": [sid]})
    assert resp.status_code == 200 and resp.json()["approved"] == 1
    assert (await repository.knowledge_stats())["total"] == 1    # 采纳之后才落库


async def test_同一条重复采纳会被挡掉(client, monkeypatch):
    """页面上手快点两下、或者拿已弃用的行来入库,都不该造出重复知识。"""
    monkeypatch.setattr(kb_api.dualwrite, "vectorize_pending", lambda: _noop())
    sid = await repository.insert_staging("mine-t", "conv:1", "运费怎么算", "满99包邮")
    await repository.set_staging_status([sid], "kept")

    assert (await client.post("/api/kb/staging/approve", json={"ids": [sid]})).status_code == 200
    again = await client.post("/api/kb/staging/approve", json={"ids": [sid]})
    assert again.status_code == 409                               # 已经不是待审了
    assert (await repository.knowledge_stats())["total"] == 1     # 库里仍是一条


async def test_弃用只记结论不入库(client):
    sid = await repository.insert_staging("mine-t", "conv:1", "老板微信多少", "不便透露")
    await repository.set_staging_status([sid], "kept")

    resp = await client.post("/api/kb/staging/reject", json={"ids": [sid]})
    assert resp.status_code == 200 and resp.json()["rejected"] == 1
    assert (await repository.knowledge_stats())["total"] == 0
    # 行留着不删:batch_no / source_ref 是溯源用的,将来要能回答「这条当初谁挖的、为啥没要」
    assert [r.status for r in await repository.list_staging_by_ids([sid])] == ["rejected"]
