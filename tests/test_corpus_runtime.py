import asyncio
from scripts.ch10 import build_corpus as corpus
from scripts.ch10 import build_dataset as dataset


async def test_simulation_timeout_is_bounded_and_cancellable(monkeypatch):
    cancelled = []

    class Model:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)

    monkeypatch.setattr(corpus, "get_chat_model", Model)
    monkeypatch.setattr(corpus, "CALL_TIMEOUT", 0.01)
    assert await corpus.simulate("物流", 1) == []
    assert len(cancelled) == 5


async def test_each_successful_batch_is_checkpointed(monkeypatch):
    class Model:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt):
            return corpus._SimBatch(items=[corpus._SimItem(text="快递到哪了", labels=["物流"])])

    monkeypatch.setattr(corpus, "get_chat_model", Model)
    checkpoints = []
    rows = await corpus.simulate("物流", 2, on_batch=lambda items: checkpoints.append(list(items)))
    assert len(rows) == 2
    assert [len(items) for items in checkpoints] == [1, 2]


async def test_augmentation_timeout_drops_only_that_variant(monkeypatch):
    class Model:
        async def ainvoke(self, prompt):
            if "挂起" in prompt:
                await asyncio.sleep(10)
            return type("Response", (), {"content": "请问快递到哪了"})()

    monkeypatch.setattr(dataset, "get_chat_model", Model)
    monkeypatch.setattr(dataset, "CALL_TIMEOUT", 0.01)
    inputs = [{"text": "挂起的改写", "labels": ["物流"]},
              {"text": "快递到哪了", "labels": ["物流"]}]
    output = await dataset.augment(inputs)
    assert output == [{"text": "请问快递到哪了", "labels": ["物流"], "origin": "augmented"}]
    assert len(inputs) == 2


async def test_synthetic_only_augmentation_never_sends_pool_text(monkeypatch):
    sent = []

    class Model:
        async def ainvoke(self, prompt):
            sent.append(prompt)
            return type("Response", (), {"content": "模拟问题的变体"})()

    monkeypatch.setattr(dataset, "get_chat_model", Model)
    inputs = [{"text": "真实问题[手机号]", "labels": ["其他"], "origin": "pool"},
              {"text": "公开模拟问题", "labels": ["其他"], "origin": "simulated"}]
    output = await dataset.augment(inputs, synthetic_only=True)
    assert len(sent) == 1 and "公开模拟问题" in sent[0]
    assert "真实问题" not in sent[0]
    assert output[0]["text"] == "模拟问题的变体"
    assert inputs[0]["origin"] == "pool"
