"""Offline checks for restoring the exact previous selector on frozen cases."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from scripts import eval_context_full_retrieval as exp


def test_prepare_uses_identical_current_data_and_previous_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(exp.unified.base.llm,"get_chat_model",lambda **kw:pytest.fail("model call while preparing"))
    manifest=exp.prepare(tmp_path)
    assert exp.unified.read(tmp_path/"dataset.json")==exp.unified.read(exp.PARENT/"dataset.json")
    assert manifest["dataset_sha256"]==exp.unified.read(exp.PARENT/"manifest.json")["dataset_sha256"]
    assert manifest["arms"]==["task"]
    assert all(v['current']==v['previous'] for v in manifest['source_lineage']['identity_sources'].values())
    with pytest.raises(ValueError,match="immutable"):
        exp.prepare(tmp_path)


def test_selector_searches_unretired_history_and_retains_original_relation():
    from app.core.context_task import assemble_task_context
    from app.core.context_state import build_snapshot
    rows=[{'role':'user','content':'工作室的订单是710002，妈妈的是710006。'},
          {'role':'assistant','content':'已登记两单用途。'}]
    for i in range(8):
        rows.extend([{'role':'user','content':f'包装资料第{i}页'},
                     {'role':'assistant','content':'请保留原始说明书。'}])
    query='工作室那单到哪了？'
    rows.append({'role':'user','content':query})
    rows=[{'id':i+1,**r} for i,r in enumerate(rows)]
    before=deepcopy(rows)
    packet=assemble_task_context(build_snapshot(rows,900001),query,'coref',policy='task')
    assert not packet['blocked'] and rows==before
    assert 1 in packet['manifest']['selected_message_ids']
    assert '工作室的订单是710002' in packet['history']
    assert not packet['summary_text']


async def test_restored_workflow_with_mock_http_and_frozen_independent_replay(tmp_path, monkeypatch):
    # Exercise the established HTTP mock through the real graph under the new
    # wrapper. It covers input capture, three turns, tool freshness and costs.
    from tests.test_context_unified import test_actual_workflow_all_five_arms_with_mock_http
    from scripts.audit_context_unified import audit
    original_prepare=exp.unified.prepare
    with exp.binding(), patch.object(exp.unified,'prepare',
            lambda output,**kw:original_prepare(output,arms=exp.ARMS,**kw)):
        await test_actual_workflow_all_five_arms_with_mock_http(tmp_path,monkeypatch)
    independent=audit(tmp_path)
    assert independent['audit_ok'] and independent['complete'],independent['issues']
    result=exp.finalize(tmp_path)
    assert result['complete'] and result['independent_passed'],result['issues']
    comparison=exp.unified.read(tmp_path/'COMPARISON.json')
    assert not comparison['formal']
    assert comparison['groups']['task']['overall']['task']['pass']==3
    assert comparison['groups']['task']['overall']['summary_cost']['model_calls']==0
