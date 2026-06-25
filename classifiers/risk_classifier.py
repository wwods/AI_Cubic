"""
Y축 분류기: GARCH(1,1) + VaR + MDD 복합 리스크
backtest_3d_cubic.py의 compute_y_axis() 그대로 이식
"""

import numpy as np
import pandas as pd
from arch import arch_model


RISK_LABEL = {0: "Low", 1: "Mid", 2: "High"}
RISK_KR    = {0: "저위험", 1: "중위험", 2: "고위험"}

# 가중치 (논문 3.3절 기준)
W_GARCH = 0.40
W_VAR   = 0.35
W_MDD   = 0.25
MIN_OBS = 60   # GARCH 최소 관측치


def compute_risk(close: pd.Series, percentile_window: int = 30) -> pd.Series:
    """
    복합 리스크 점수 계산 후 3단계 분류

    구성:
      GARCH(1,1) 예측 변동성  40%
      1일 95% VaR             35%
      5일 MDD                 25%

    퍼센타일 기준: 30일 롤링 윈도우로 균등 3분할
    
    Returns:
        Series[int]: 0=Low, 1=Mid, 2=High
    """
    returns = close.pct_change().dropna()

    # ── GARCH(1,1) 예측 변동성 ──
    garch_vol = pd.Series(index=returns.index, dtype=float)
    for i in range(MIN_OBS, len(returns)):
        window_ret = returns.iloc[:i] * 100
        try:
            am  = arch_model(window_ret, vol="Garch", p=1, q=1, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            forecast = res.forecast(horizon=1)
            garch_vol.iloc[i] = np.sqrt(forecast.variance.values[-1, 0]) / 100
        except Exception:
            garch_vol.iloc[i] = returns.iloc[max(0, i - 20):i].std()

    # ── 1일 95% VaR (Historical) ──
    var_series = pd.Series(index=returns.index, dtype=float)
    for i in range(MIN_OBS, len(returns)):
        hist = returns.iloc[max(0, i - MIN_OBS):i]
        var_series.iloc[i] = abs(np.percentile(hist, 5))

    # ── 5일 MDD ──
    close_aligned = close.loc[returns.index]
    rolling_max   = close_aligned.rolling(5).max()
    rolling_min   = close_aligned.rolling(5).min()
    mdd_series    = ((rolling_min - rolling_max) / rolling_max).abs().fillna(0)

    # ── 가중 합산 ──
    composite = (
        garch_vol.fillna(0) * W_GARCH +
        var_series.fillna(0) * W_VAR +
        mdd_series.fillna(0) * W_MDD
    ).dropna()

    # ── 30일 퍼센타일 균등 3분할 ──
    risk_class = pd.Series(index=composite.index, dtype=float)
    for i in range(percentile_window, len(composite)):
        window_vals = composite.iloc[i - percentile_window:i + 1]
        pct = (composite.iloc[i] - window_vals.min()) / \
              (window_vals.max() - window_vals.min() + 1e-10)
        if pct < 0.333:
            risk_class.iloc[i] = 0   # 저위험
        elif pct < 0.667:
            risk_class.iloc[i] = 1   # 중위험
        else:
            risk_class.iloc[i] = 2   # 고위험

    return risk_class.dropna().astype(int)


def get_latest_risk(close: pd.Series) -> int:
    """마지막 날의 리스크 등급만 반환 (실시간 분류용)"""
    risk = compute_risk(close)
    return int(risk.iloc[-1])
