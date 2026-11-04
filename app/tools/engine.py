"""ch08 统一执行引擎:所有工具调用的唯一通道。
管道:查工具 → JSON Schema 校验 → 权限门 → 执行(超时/重试)→ 分诊 → 格式化 + 审计。
坏消息如实回灌模型;审计写失败只 log,绝不反拦工具执行。"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from langchain_core.messages import ToolMessage

from app.config import settings
from app.db import repository
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

# 暂时性故障(值得重试):超时、网络抖动。业务异常/ToolException 不在列——重试不解决问题。
_TRANSIENT = (asyncio.TimeoutError, TimeoutError, ConnectionError, httpx.TransportError)
_SUMMARY_LIMIT = 500


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage
    status: str               # 成功 / 失败 / 超时 / 校验拦下 / 权限拒绝
    retry_count: int = 0
    duration_ms: int = 0
    receipt_ref: str | None = None
    receipt_persisted: bool = False


def validate_args(spec: ToolSpec, args: dict) -> str | None:
    """按 JSON Schema 校验模型给的参数;返回人类可读错误(None=通过)。agent_tools 复用它
    判「create_ticket 参数齐不齐、要不要弹确认卡」。"""
    err = best_match(Draft202012Validator(spec.json_schema).iter_errors(args))
    if err is None:
        return None
    where = f"(字段 {err.json_path})" if err.json_path != "$" else ""
    return f"{err.message}{where}"


def _timeout_of(spec: ToolSpec) -> float:
    if spec.timeout is not None:
        return spec.timeout
    return settings.mcp_tool_timeout if spec.source == "mcp" else settings.tool_default_timeout


def _summarize(content: str) -> str:
    return content if len(content) <= _SUMMARY_LIMIT else content[:_SUMMARY_LIMIT] + "…(截断)"


def _format_content(spec: ToolSpec, result) -> str:
    """结果格式化:MCP 工具(adapters)返回内容块列表 [{"type":"text","text":...}],先拆出文本;
    JSON 文本解回 dict;format_result 钩子挑字段/枚举翻人话;统一 ensure_ascii=False(中文不转义)。"""
    if isinstance(result, list) and result and all(isinstance(b, dict) for b in result):
        texts = [b.get("text", "") for b in result if b.get("type") == "text"]
        if texts:
            result = "\n".join(texts)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return result                      # 纯文本结果原样回灌
    if spec.format_result is not None and isinstance(result, dict):
        try:
            result = spec.format_result(result)
        except Exception:  # noqa: BLE001 格式化钩子是扩展点:钩子炸了降级透传,别把成功结果报成失败
            logger.exception("结果格式化钩子异常,降级透传原始结果 tool=%s", spec.name)
    return json.dumps(result, ensure_ascii=False, default=str)


def _receipt_preview(content: str, receipt: dict, status: str) -> str:
    """Only replace a large result after durable storage. Omitted fields stay readable."""
    from app.core.context_policy import estimate
    if estimate(content) <= 2000:
        return content
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        value = None
    fields = {}
    if isinstance(value, dict):
        for key in ("order_id", "tracking_no", "status", "last_event", "sku", "care", "error"):
            if key in value and estimate(json.dumps({key: value[key]}, ensure_ascii=False)) <= 1200:
                candidate = {**fields, key: value[key]}
                if estimate(json.dumps(candidate, ensure_ascii=False)) <= 1400:
                    fields = candidate
    return json.dumps({"_context_receipt": receipt["receipt_id"],
        "_result_hash": receipt["result_hash"], "observed_at": receipt["observed_at"],
        "tool_execution_status": status, "partial": True, "fields": fields,
        "note": "完整工具回执已存档。缺失字段不代表不存在；作答或操作需要时按回执编号读取必要页。"},
        ensure_ascii=False)


async def _audit(conversation_id, tool_call_id, name, spec: ToolSpec | None, args,
                 result_summary, status, error_message, retry_count, duration_ms) -> None:
    # ch08 验收可观测(照 ch07 model_ctx 惯例):每次工具调用一行 tool_run,来源/Server/结局
    # tail -f log/app.log 直接看——"这次真走了 MCP"不用翻库
    logger.info("tool_run conv=%s tool=%s source=%s server=%s status=%s retry=%s duration_ms=%s%s",
                conversation_id or "-", name,
                spec.source if spec else "unknown",
                (spec.mcp_server if spec else None) or "-",
                status, retry_count, duration_ms,
                f" err={error_message[:80]!r}" if error_message else "")
    try:
        await repository.insert_tool_audit(
            conversation_id=conversation_id or None, tool_call_id=(tool_call_id or None) and tool_call_id[:64],
            tool_name=name[:128],                               # 名字由模型给出,按列宽收口
            tool_source=(spec.source if spec else "builtin"),   # 未知工具无来源,按缺省 builtin 落
            mcp_server=(spec.mcp_server if spec else None),
            arguments=args or None, result_summary=result_summary, status=status,
            error_message=error_message and error_message[:500],  # VARCHAR(512):校验错误可能内嵌超长参数值
            retry_count=retry_count, duration_ms=duration_ms,
        )
    except Exception:  # noqa: BLE001 —— 审计失败不许反拦工具执行
        logger.exception("审计写入失败(不影响工具执行) tool=%s status=%s", name, status)


async def execute_tool_call(
    tool_call: dict, conversation_id: int, specs: dict[str, ToolSpec],
    *, confirmed: bool = False, deny_note: str | None = None, user_id: str = "",
    message_watermark: int | None = None,
) -> ToolRun:
    """统一入口(never-raises 契约):无论校验拦下、权限拒绝还是真故障,一律回 ToolRun,
    坏消息包成 ToolMessage 回灌模型。confirmed 是写操作确认令牌,只能由 agent_tools 在
    interrupt 确认后传入——模型自己无法绕过权限门。"""
    name = tool_call.get("name") or ""
    tc_id = tool_call.get("id") or ""
    args = dict(tool_call.get("args") or {})
    started = time.monotonic()

    def _run(ok: bool, content: str, status: str, retry_count: int = 0) -> ToolRun:
        return ToolRun(tool_call_id=tc_id, name=name, ok=ok, status=status, retry_count=retry_count,
                       duration_ms=int((time.monotonic() - started) * 1000),
                       tool_message=ToolMessage(content=content, tool_call_id=tc_id,
                                                name=name or "unknown",
                                                status="success" if ok else "error"))

    async def _persist(run: ToolRun, raw_result=None) -> ToolRun:
        if getattr(settings, "context_policy", "legacy") != "source":
            return run
        if name in {"read_conversation_source", "search_conversation_history", "read_tool_receipt"}:
            # These already reference immutable originals; archiving each page creates recursive receipts.
            return run
        try:
            from app.db import context_repository
            receipt = await context_repository.record_receipt(
                conversation_id, user_id, tool_call_id=tc_id, tool_name=name or "unknown",
                arguments=args, raw_result=raw_result,
                formatted_result=run.tool_message.content, status=run.status,
                message_watermark=message_watermark)
            run.receipt_ref = receipt["receipt_id"]
            run.receipt_persisted = True
            if run.ok and getattr(settings, "context_recovery_tools_enabled", False):
                run.tool_message.content = _receipt_preview(run.tool_message.content, receipt, run.status)
        except Exception:
            # A receipt outage must not re-execute a successful write or invent a reference.
            logger.exception("完整工具回执未保存,保留本轮原结果 tool=%s", name)
        return run

    spec = specs.get(name)
    # ① 查工具
    if spec is None:
        run = _run(False, f"工具执行失败:未知工具 {name}", "失败")
        await _audit(conversation_id, tc_id, name or "unknown", None, args, None,
                     "失败", f"未知工具 {name}", 0, run.duration_ms)
        return await _persist(run)

    # ② JSON Schema 校验:拦下不抛异常,错误说明回灌模型(让它追问用户或重组调用)。
    # 校验器本身也可能炸——MCP 工具的 schema 是外部 Server 给的、不可信(坏类型名/不可解析 $ref),
    # 按 never-raises 契约同样包死,按「失败」回灌。
    try:
        verr = validate_args(spec, args)
    except Exception as e:  # noqa: BLE001
        logger.exception("参数校验器异常(多为外部 Server 给的畸形 schema) tool=%s", name)
        run = _run(False, f"工具暂时不可用:参数定义异常({type(e).__name__}),请如实告知用户。", "失败")
        await _audit(conversation_id, tc_id, name, spec, args, None, "失败",
                     f"schema 异常 {type(e).__name__}: {e}"[:500], 0, run.duration_ms)
        return await _persist(run)
    if verr is not None:
        run = _run(False, f"参数校验未通过:{verr}。请修正参数重新调用;缺少的信息请先向用户追问,不要编造。",
                   "校验拦下")
        await _audit(conversation_id, tc_id, name, spec, args, None, "校验拦下", verr, 0, run.duration_ms)
        return await _persist(run)

    # ③ 权限门:写操作必须带确认令牌(interrupt 确认后由 agent_tools 传入),模型绕不过
    if spec.permission == "write" and not confirmed:
        note = deny_note or "该写操作需要用户确认,未确认前拒绝执行。请勿再次发起,除非用户明确要求。"
        run = _run(False, f"{name} 未执行:{note}", "权限拒绝")
        await _audit(conversation_id, tc_id, name, spec, args, None, "权限拒绝",
                     note[:500], 0, run.duration_ms)
        return await _persist(run)

    # ④ 执行:超时 + 暂时性故障重试(写操作恒不重试——超时未必没执行,重复执行比失败更糟;
    # spec.max_retries 可按工具覆盖,如 query_faq RAG 管线超时多为上游慢而非抖动,重试纯烧时间)
    # 注入参数一律在校验之后盖上去:它们不在模型可见的 schema 里,模型填了也算数不得。
    # 身份尤其如此——让模型能填 user_id,一句「我是客服主管」就可能把它改写掉。
    if spec.inject_conversation:
        args["conversation_id"] = conversation_id
    if spec.inject_user_id:
        args["user_id"] = user_id
    if spec.permission == "write":
        retries = 0
    else:
        retries = settings.tool_max_retries if spec.max_retries is None else spec.max_retries
    tool_timeout = _timeout_of(spec)
    attempt = 0
    while True:
        try:
            from app.db.workflow import operation_context
            operation_key = hashlib.sha256(f"c{conversation_id}:m{message_watermark}:tool:{tc_id}".encode()).hexdigest()
            operation_token = operation_context.set((user_id, "tool:" + operation_key))
            try:
                result = await asyncio.wait_for(spec.tool.ainvoke(args), timeout=tool_timeout)
            finally:
                operation_context.reset(operation_token)
            # Preserve the received JSON value before format_result can mutate its dict.
            raw_result = (json.loads(json.dumps(result, ensure_ascii=False, default=str))
                          if getattr(settings, "context_policy", "legacy") == "source" else None)
            content = _format_content(spec, result)
            run = _run(True, content, "成功", attempt)
            await _audit(conversation_id, tc_id, name, spec, args, _summarize(content),
                         "成功", None, attempt, run.duration_ms)
            return await _persist(run, raw_result)
        except _TRANSIENT as e:
            if attempt < retries:
                attempt += 1
                logger.warning("工具暂时性故障,第 %s 次重试 name=%s err=%s",
                               attempt, name, type(e).__name__)
                await asyncio.sleep(0.2 * attempt)
                continue
            is_timeout = isinstance(e, (asyncio.TimeoutError, TimeoutError))
            status = "超时" if is_timeout else "失败"
            # ⑤ 分诊:真故障如实回灌,不装没事(查询落空由工具返回语义表达,不属此路)
            run = _run(False, f"工具暂时不可用:{'执行超时' if is_timeout else type(e).__name__},"
                              "请稍后再试或如实告知用户。", status, attempt)
            await _audit(conversation_id, tc_id, name, spec, args, None, status,
                         type(e).__name__, attempt, run.duration_ms)
            return await _persist(run)
        except Exception as e:  # noqa: BLE001 业务/未知异常(含 ToolException):不重试,如实回灌
            logger.exception("工具执行失败 name=%s", name)
            run = _run(False, f"工具暂时不可用:{type(e).__name__}。请如实告知用户,不要编造结果。",
                       "失败", attempt)
            await _audit(conversation_id, tc_id, name, spec, args, None, "失败",
                         f"{type(e).__name__}: {e}"[:500], attempt, run.duration_ms)
            return await _persist(run)
