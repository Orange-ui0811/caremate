"""管理端可调整的运行配置:原子落盘,每次检索读取,重启后仍生效。"""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, FiniteFloat, ValidationError

from app.config import settings
from app.core.confidence import SCORING_METHOD

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).resolve().parents[2] / "data/runtime_settings.json"


class ConfidenceSettings(BaseModel):
    threshold: FiniteFloat = Field(ge=0, le=1, strict=True)
    updated_at: datetime
    # 无版本的旧配置属于四信号，不能直接用于最高重排分。
    scoring_method: str = "four_signal_v1"


def get_confidence_settings() -> dict:
    try:
        saved = ConfidenceSettings.model_validate_json(SETTINGS_PATH.read_text(encoding="utf-8"))
        if saved.scoring_method == SCORING_METHOD:
            return saved.model_dump(mode="json")
        logger.warning("阈值评分版本不匹配(%s -> %s),使用当前默认阈值",
                       saved.scoring_method, SCORING_METHOD)
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, ValidationError) as exc:
        logger.warning("无法读取运行配置,使用默认阈值: %s", exc)
    return {"threshold": settings.evidence_confidence_threshold, "updated_at": None,
            "scoring_method": SCORING_METHOD}


def get_evidence_confidence_threshold() -> float:
    return get_confidence_settings()["threshold"]


def save_confidence_threshold(threshold: float) -> dict:
    value = ConfidenceSettings(threshold=threshold, updated_at=datetime.now(timezone.utc),
                               scoring_method=SCORING_METHOD)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=SETTINGS_PATH.parent,
                                         prefix="runtime-settings-", suffix=".tmp", delete=False) as f:
            temp_path = Path(f.name)
            json.dump(value.model_dump(mode="json"), f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, SETTINGS_PATH)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return value.model_dump(mode="json")
