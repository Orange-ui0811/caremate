from typing import Literal

from pydantic import BaseModel, Field


class CreateTicketRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=100)
    user_id: str = Field(min_length=1, max_length=64)
    conversation_id: int
    description: str = Field(min_length=1)
    ticket_type: Literal["售后", "投诉", "咨询"]


class CreateTicketResponse(BaseModel):
    ticket_no: str
    status: str = "已转人工"


class CreateRefundRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=100)
    user_id: str = Field(min_length=1, max_length=64)
    conversation_id: int
    order_id: str = Field(min_length=1)
    reason: Literal["七天无理由", "质量问题", "发错货", "不想要了", "其他"]


class CreateRefundResponse(BaseModel):
    ticket_no: str
    status: str = "退款申请已提交"


class ResumeRequest(BaseModel):
    operation_id: str | None = Field(default=None, min_length=1, max_length=160)
    cancelled: bool = Field(default=False, strict=True)
    user_id: str = Field(min_length=1, max_length=64)
    interrupt_id: str = Field(min_length=1, max_length=128)
    conversation_id: int
    order_id: str | None = Field(default=None, min_length=1)   # ch06 订单选择器点选
    confirmed: bool | None = Field(default=None, strict=True)   # 仅接受明确布尔确认/取消
