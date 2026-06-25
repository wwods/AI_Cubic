"""
/analyze 라우터
Spring Boot의 AiController.java가 호출하는 핵심 엔드포인트
"""

import os
import pandas as pd
import mysql.connector
from fastapi import APIRouter, HTTPException
from models.schemas import AnalyzeRequest, AnalyzeResponse, CellCoordinate, CubicDescription
from engine.cubic_engine import CubicEngine
from engine.description import generate_description
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()
engine = CubicEngine()


def _build_dataframes(ohlcv: list) -> tuple[pd.Series, pd.Series]:
    df = pd.DataFrame([row.model_dump() for row in ohlcv])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["close"], df["volume"]


def _get_cell_weights(cell_num: int) -> dict | None:
    """DB에서 최신 버전 셀 가중치 조회. 실패 시 None 반환"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "1234"),
            database=os.getenv("DB_NAME", "capstone"),
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT MAX(version) AS v FROM cell_weight_config")
        latest = cursor.fetchone()["v"]
        cursor.execute(
            "SELECT weight_buy, weight_sell FROM cell_weight_config "
            "WHERE cell_num = %s AND version = %s",
            (cell_num, latest)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row
    except Exception:
        return None


def _calc_cubic_score(cell_num: int) -> int:
    """
    PPO 가중치 기반 Cubic Score 계산 (0~100)
    신호와 점수 일치 보장:
      BUY  → weight_buy 높음  → 점수 높음
      SELL → weight_sell 높음 → 점수 낮음
      HOLD → 균등            → 점수 ~50

    DB 조회 실패 시 셀 번호 기반 폴백
    """
    weights = _get_cell_weights(cell_num)
    if weights:
        score = (weights["weight_buy"] - weights["weight_sell"] + 1) / 2 * 100
        return round(score)
    # 폴백: 셀 번호 기반
    return round((26 - cell_num) / 26 * 100)


@router.post("/signal", response_model=AnalyzeResponse)
def analyze_signal(req: AnalyzeRequest):
    if len(req.ohlcv) < 60:
        raise HTTPException(
            status_code=422,
            detail=f"OHLCV 데이터가 최소 60일 필요합니다. 현재: {len(req.ohlcv)}일"
        )

    try:
        close, volume = _build_dataframes(req.ohlcv)
        result = engine.analyze(close, volume)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

    # 자연어 설명 생성
    desc = generate_description(
        x=result["x"],
        y=result["y"],
        z=result["z"],
        action=result["action"],
    )

    # PPO 가중치 기반 Cubic Score (신호와 일치)
    cubic_score = _calc_cubic_score(result["cell_num"])

    return AnalyzeResponse(
        symbol       = req.symbol,
        date         = result["date"],
        cell         = CellCoordinate(
            x        = result["x"],
            y        = result["y"],
            z        = result["z"],
            cell_num = result["cell_num"],
        ),
        action       = result["action"],
        action_code  = result["action_code"],
        cubic_score  = cubic_score,
        regime_raw   = result["regime_raw"],
        risk_raw     = result["risk_raw"],
        momentum_raw = result["momentum_raw"],
        description  = CubicDescription(**desc),
    )