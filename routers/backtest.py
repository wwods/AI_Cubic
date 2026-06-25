"""
/backtest 라우터
성과 지표(Sharpe, MDD, 수익률) 계산 + 3전략 비교
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from models.schemas import BacktestRequest, BacktestResponse, StrategyMetrics
from engine.cubic_engine import CubicEngine

router = APIRouter()
engine = CubicEngine()


def _build_dataframes(ohlcv: list) -> tuple[pd.Series, pd.Series]:
    df = pd.DataFrame([row.model_dump() for row in ohlcv])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["close"], df["volume"]


def _backtest(close: pd.Series, signals: pd.Series,
              capital: float, cost: float) -> pd.Series:
    """간단 백테스팅 엔진 (backtest_3d_cubic.py backtest() 이식)"""
    common  = close.index.intersection(signals.index)
    close   = close.loc[common]
    signals = signals.loc[common]
    cash, shares, position = capital, 0, 0
    values = []

    for date, price in close.items():
        sig = signals.loc[date]
        if sig == 1 and position == 0 and cash > 0:
            shares   = int(cash * (1 - cost) / price)
            cash    -= shares * price * (1 + cost)
            position = 1
        elif sig == -1 and position == 1 and shares > 0:
            cash    += shares * price * (1 - cost)
            shares   = 0
            position = 0
        values.append(cash + shares * price)

    return pd.Series(values, index=close.index)


def _metrics(pv: pd.Series, name: str, capital: float) -> StrategyMetrics:
    """성과 지표 계산"""
    returns      = pv.pct_change().dropna()
    total_ret    = (pv.iloc[-1] / capital - 1) * 100
    n_years      = len(pv) / 252
    annual_ret   = ((pv.iloc[-1] / capital) ** (1 / n_years) - 1) * 100
    sharpe       = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252)
    drawdown     = (pv - pv.cummax()) / pv.cummax()
    mdd          = drawdown.min() * 100
    win_rate     = (returns > 0).mean() * 100

    return StrategyMetrics(
        strategy     = name,
        total_return = round(total_ret, 2),
        annual_return= round(annual_ret, 2),
        sharpe       = round(float(sharpe), 3),
        mdd          = round(float(mdd), 2),
        win_rate     = round(float(win_rate), 2),
        final_value  = round(float(pv.iloc[-1]), 0),
    )


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


@router.post("/run", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest):
    """
    3전략 비교 백테스팅 실행
      - 3D Cubic (이 시스템)
      - Buy & Hold (베이스라인 1)
      - RSI 단일 전략 (베이스라인 2)
    """
    if len(req.ohlcv) < 90:
        raise HTTPException(status_code=422, detail="백테스팅은 최소 90일 데이터 필요")

    try:
        close, volume = _build_dataframes(req.ohlcv)
        capital, cost = req.initial_capital, req.transaction_cost

        # ── 3D Cubic ──
        cell_df       = engine.compute_all(close, volume)
        signals_cubic = cell_df["signal"]
        pv_cubic      = _backtest(close, signals_cubic, capital, cost)

        # ── Buy & Hold ──
        shares_bah = int(capital * (1 - cost) / close.iloc[0])
        cash_bah   = capital - shares_bah * close.iloc[0] * (1 + cost)
        pv_bah     = (cash_bah + shares_bah * close).rename("total")

        # ── RSI 단일 전략 ──
        rsi         = _compute_rsi(close)
        sig_rsi     = pd.Series(0, index=close.index)
        sig_rsi[rsi < 30]  =  1
        sig_rsi[rsi > 70]  = -1
        pv_rsi = _backtest(close, sig_rsi, capital, cost)

        # ── 성과 지표 ──
        metrics = [
            _metrics(pv_cubic, "3D Cubic",      capital),
            _metrics(pv_bah,   "Buy & Hold",    capital),
            _metrics(pv_rsi,   "RSI Strategy",  capital),
        ]

        # ── 셀 분포 & 신호 요약 ──
        cell_dist = cell_df["cell"].value_counts().to_dict()
        cell_dist = {str(k): int(v) for k, v in cell_dist.items()}
        signal_summary = {
            "BUY":  int((cell_df["signal"] ==  1).sum()),
            "HOLD": int((cell_df["signal"] ==  0).sum()),
            "SELL": int((cell_df["signal"] == -1).sum()),
        }
        period = f"{req.ohlcv[0].date} ~ {req.ohlcv[-1].date}"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"백테스팅 오류: {str(e)}")

    return BacktestResponse(
        symbol           = req.symbol,
        period           = period,
        metrics          = metrics,
        cell_distribution= cell_dist,
        signal_summary   = signal_summary,
    )
