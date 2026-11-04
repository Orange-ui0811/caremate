from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Index, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.mysql import LONGTEXT

from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束"), server_default="进行中"
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)               # ch07 早期轮次滚动摘要
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # 摘要覆盖到的消息 id
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool"))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    activities: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ContextSnapshot(Base):
    """Derived state only; original messages remain in messages."""
    __tablename__ = "context_snapshots"

    conversation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    watermark: Mapped[int] = mapped_column(BigInteger, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ContextReceipt(Base):
    """Full tool receipt, separate from the abbreviated audit display."""
    __tablename__ = "context_receipts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    message_watermark: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    tool_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    arguments: Mapped[dict] = mapped_column(JSON, nullable=False)
    raw_result: Mapped[dict | list | str | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    formatted_result: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT(), "mysql"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    result_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ContextSummarySegment(Base):
    """Immutable background summary with the full source interval recoverable."""
    __tablename__ = "context_summary_segments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    from_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_refs: Mapped[list] = mapped_column(JSON, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ContextMaintenanceLease(Base):
    __tablename__ = "context_maintenance_leases"

    conversation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    owner: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ContextSource(Base):
    __tablename__ = "context_sources"
    __table_args__ = (Index("ix_context_sources_scope", "conversation_id", "message_id", "start"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    start: Mapped[int] = mapped_column(Integer, nullable=False)
    end: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class ContextSourceTerm(Base):
    __tablename__ = "context_source_terms"

    conversation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    term: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询", "退款"))
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理"), server_default="待处理"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[int] = mapped_column(Integer, server_default="0")
    prev_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done"), server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        # kept 是「去重保留、等人审」,不是「已入库」;入库只走 /api/kb/staging/approve
        Enum("extracted", "kept", "discarded", "approved", "rejected"),
        server_default="extracted",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ToolAuditLog(Base):
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # 不挂 FK:审计不能被引用约束拦
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp"))
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Enum("成功", "失败", "超时", "校验拦下", "权限拒绝"))
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, server_default="0")
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("conversations.id"), nullable=True)
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Enum("retrieval_low_conf", "self_check", "user_feedback"))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # ch09 落池时召回快照。none_as_null=True:Python None 存 SQL NULL 而非 JSON null
    # (默认 False 会把 None 写成 JSON 'null',IS NOT NULL 判真、语义全乱,实测踩坑)
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    matched_review_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("review_queue.id"), nullable=True)               # ch09 查重归并落点
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ReviewQueue(Base):
    """ch09 飞轮待审队列:一行 = 一个去重后的知识缺口;查重命中累加 occurrence_count 不新建行。"""
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    normalized_question: Mapped[str] = mapped_column(String(512))
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, server_default="1")
    review_status: Mapped[str] = mapped_column(Enum("待审", "通过", "驳回"), server_default="待审")
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class EvalRun(Base):
    """ch09 自动化评估流水线:一行 = 一轮评估;metrics JSON 收各指标,按时间连成趋势。"""
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    triggered_by: Mapped[str] = mapped_column(Enum("定时", "手动"), server_default="定时")
    dataset_size: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FaithCase(Base):
    """ch04 编造个案台账:一题一行,跨轮累计。

    报告每跑一次就被覆盖,个案和它的处置状态得有地方长期待着。同一题再次被判编造只更新
    这一行(seen_count+1);已解决的题重新被判出来会退回「未解决」——那是复发。
    citations 是这一轮喂给模型的 Top-K 证据全集(答案里的角标 [n] 即其序号),追溯时要能
    看到当时喂了什么、以及答案实际只引用了其中哪几条。
    """
    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), server_default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决"), server_default="未解决")
    seen_count: Mapped[int] = mapped_column(Integer, server_default="1")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TopicClassification(Base):
    """ch10 主题分类结果:旁路批量归类,一行 = 一条问题的一次归类;labels 存命中类目名数组。"""
    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("low_confidence_questions.id"))
    labels: Mapped[list] = mapped_column(JSON)
    classified_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
