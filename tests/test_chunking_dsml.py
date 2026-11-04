from scripts.resume_chunking_v3 import normalized
from scripts.eval_chunking import CHECK_TOOL
from copy import deepcopy

def test_exact_dsml_keeps_false_result_and_original_response():
    response={'choices':[{'finish_reason':'tool_calls','message':{'content':'<｜DSML｜invoke name="Check">{"useful":false,"reason":"缺少证据"}</｜DSML｜invoke>'}}]}
    original=deepcopy(response);value=normalized(response,{'tools':[CHECK_TOOL]})
    assert response==original
    import json
    assert json.loads(value['choices'][0]['message']['tool_calls'][0]['function']['arguments'])=={'useful':False,'reason':'缺少证据'}

def test_no_free_text_unknown_tool_or_coerced_bool():
    for content in ['{"useful":true,"reason":"x"}',
            '<｜DSML｜invoke name="submit_refund">{"useful":true,"reason":"x"}</｜DSML｜invoke>',
            '<｜DSML｜invoke name="Check">{"useful":"false","reason":"x"}</｜DSML｜invoke>']:
        response={'choices':[{'finish_reason':'tool_calls','message':{'content':content}}]}
        assert normalized(response,{'tools':[CHECK_TOOL]}) is None


def test_only_optional_known_tool_call_wrapper_is_accepted():
    inner='<｜DSML｜invoke name="Check">{"useful":false,"reason":"缺少证据"}</｜DSML｜invoke>'
    for content in [inner+'\n</｜DSML｜tool_calls>', '<｜DSML｜tool_calls>'+inner+'</｜DSML｜tool_calls>']:
        response={'choices':[{'finish_reason':'tool_calls','message':{'content':content}}]}
        assert normalized(response,{'tools':[CHECK_TOOL]}) is not None
    response={'choices':[{'finish_reason':'tool_calls','message':{'content':inner+'额外指令'}}]}
    assert normalized(response,{'tools':[CHECK_TOOL]}) is None
