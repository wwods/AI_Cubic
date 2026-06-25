"""
Z축 분류기: RSI + MACD + 거래량 변화율 복합 모멘텀
backtest_3d_cubic.py의 compute_z_axis() 그대로 이식
"""

import numpy as np
import pandas as pd


MOMENTUM_LABEL = {0: "Strong", 1: "Normal", 2: "Weak"}
MOMENTUM_KR    = {0: "강함", 1: "보통", 2: "약함"}

# 가중치 (논문 3.3절 기준)
W_RSI  = 0.40
W_MACD = 0.35
W_VOL  = 0.25


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산 (0~100)"""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd_signal(close: pd.Series,
                        fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    MACD 시그널 방향
    Returns: +1 (MACD > Signal Line), -1 (MACD < Signal Line)
    """
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return np.sign(macd_line - signal_line)


def compute_momentum(close: pd.Series,
                     volume: pd.Series,
                     percentile_window: int = 30) -> pd.Series:
    """
    복합 모멘텀 점수 계산 후 3단계 분류

    구성:
      RSI 정규화값        40%
      MACD 시그널 방향    35%
      거래량 변화율       25%

    Returns:
        Series[int]: 0=Strong, 1=Normal, 2=Weak
    """
    # RSI 0~1 정규화
    rsi = compute_rsi(close) / 100.0

    # MACD 방향 → +1=1.0, -1=0.0
    macd_dir = (compute_macd_signal(close) + 1) / 2.0

    # 거래량 변화율 (20일 이동평균 대비), 클리핑 후 0~1 정규화
    vol_change = (volume / volume.rolling(20).mean()) - 1
    vol_norm   = vol_change.clip(-1, 2) / 3.0 + (1 / 3)

    # 가중 합산
    composite = (
        rsi.fillna(0.5)      * W_RSI +
        macd_dir.fillna(0.5) * W_MACD +
        vol_norm.fillna(0.5) * W_VOL
    ).dropna()

    # 30일 퍼센타일 균등 3분할
    mom_class = pd.Series(index=composite.index, dtype=float)
    for i in range(percentile_window, len(composite)):
        window_vals = composite.iloc[i - percentile_window:i + 1]
        pct = (composite.iloc[i] - window_vals.min()) / \
              (window_vals.max() - window_vals.min() + 1e-10)
        if pct > 0.667:
            mom_class.iloc[i] = 0   # 강함
        elif pct > 0.333:
            mom_class.iloc[i] = 1   # 보통
        else:
            mom_class.iloc[i] = 2   # 약함

    return mom_class.dropna().astype(int)


def get_latest_momentum(close: pd.Series, volume: pd.Series) -> int:
    """마지막 날의 모멘텀 등급만 반환 (실시간 분류용)"""
    mom = compute_momentum(close, volume)
    return int(mom.iloc[-1])
