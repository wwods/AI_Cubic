"""
공통 요청/응답 스키마
Spring Boot ↔ FastAPI 사이의 JSON 계약
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── 요청 ──────────────────────────────────────

class OhlcvRow(BaseModel):
    date:   str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="종목 코드")
    ohlcv:  List[OhlcvRow] = Field(..., min_length=60)


class BacktestRequest(BaseModel):
    symbol:           str
    ohlcv:            List[OhlcvRow]
    initial_capital:  float = Field(default=10_000_000)
    transaction_cost: float = Field(default=0.002)


# ── 응답 ──────────────────────────────────────

class CellCoordinate(BaseModel):
    x:        str
    y:        str
    z:        str
    cell_num: int


class CubicDescription(BaseModel):
    """큐빅 신호 자연어 설명"""
    regime_desc:   str = Field(..., description="시장 분위기 설명")
    risk_desc:     str = Field(..., description="위험도 설명")
    momentum_desc: str = Field(..., description="모멘텀 설명")
    conclusion:    str = Field(..., description="최종 결론 한 줄")
    detail:        str = Field(..., description="상세 이유")
    summary:       str = Field(..., description="전체 요약")


class AnalyzeResponse(BaseModel):
    symbol:       str
    date:         str
    cell:         CellCoordinate
    action:       str
    action_code:  int
    cubic_score:  int
    regime_raw:   int
    risk_raw:     int
    momentum_raw: int
    description:  CubicDescription  # ← 자연어 설명


class StrategyMetrics(BaseModel):
    strategy:      str
    total_return:  float
    annual_return: float
    sharpe:        float
    mdd:           float
    win_rate:      float
    final_value:   float


class BacktestResponse(BaseModel):
    symbol:            str
    period:            str
    metrics:           List[StrategyMetrics]
    cell_distribution: dict
    signal_summary:    dict
