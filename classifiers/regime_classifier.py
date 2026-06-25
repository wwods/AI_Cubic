"""
X축 분류기: HMM 기반 시장 레짐
backtest_3d_cubic.py의 compute_x_axis() 그대로 이식
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


REGIME_LABEL = {0: "Bull", 1: "Side", 2: "Bear"}
REGIME_KR    = {0: "상승", 1: "횡보", 2: "하락"}


def compute_regime(close: pd.Series, window: int = 20) -> pd.Series:
    """
    HMM으로 시장 레짐 분류
    
    흐름: Close → 20일 롤링 수익률/변동성 → GaussianHMM(3) → 레짐 매핑 → 히스테리시스
    
    Returns:
        Series[int]: 0=Bull, 1=Side, 2=Bear  (features.index 기준)
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    roll_return = log_returns.rolling(window).mean()
    roll_vol    = log_returns.rolling(window).std()

    features = pd.concat([roll_return, roll_vol], axis=1).dropna()
    features.columns = ["return", "vol"]

    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(features.values)
    raw_states = model.predict(features.values)

    # 평균 수익률 기준으로 상태 정렬 → Bull/Side/Bear 매핑
    state_means = {}
    for s in range(3):
        mask = raw_states == s
        state_means[s] = features["return"].values[mask].mean() if mask.sum() > 0 else 0.0

    sorted_states = sorted(state_means, key=state_means.get, reverse=True)
    mapping = {sorted_states[0]: 0, sorted_states[1]: 1, sorted_states[2]: 2}
    mapped = np.array([mapping[s] for s in raw_states])

    # 히스테리시스: 최소 3거래일 유지 (잦은 전환 억제)
    HOLD_MIN = 3
    stabilized = mapped.copy()
    current_state, count = stabilized[0], 1
    for i in range(1, len(stabilized)):
        if stabilized[i] != current_state:
            if count >= HOLD_MIN:
                current_state = stabilized[i]
                count = 1
            else:
                stabilized[i] = current_state
                count += 1
        else:
            count += 1

    return pd.Series(stabilized, index=features.index, name="regime")


def get_latest_regime(close: pd.Series) -> int:
    """마지막 날의 레짐만 반환 (실시간 분류용)"""
    regime = compute_regime(close)
    return int(regime.iloc[-1])
