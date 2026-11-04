from pydantic import BaseModel, Field, StrictBool


class ChatRequest(BaseModel):
    multi_intent_enabled: StrictBool | None = Field(default=None, description='新会话模式：开启多意图，关闭单意图；已有会话保持原模式')
    request_id: str | None = Field(default=None, min_length=1, max_length=128,
                                   pattern=r"^[A-Za-z0-9_-]+$", description="同一请求重试时保持不变，用于完成结果回放")
    user_id: str = Field(min_length=1, description="用户标识,用于建/归属会话")
    message: str = Field(min_length=1, description="用户本轮消息")
    conversation_id: int | None = Field(
        default=None, description="续接会话时带上;为空则新建会话,由 done 帧回传新 id"
    )
