"""
CubicEngine: X·Y·Z 세 축 조합 → 셀 좌표 + 매매 신호 결정

신호 결정 방식 (우선순위):
  1. DB cell_weight_config 테이블에서 최신 version 가중치 조회
     → weight_buy > weight_sell 이면 BUY
     → weight_sell > weight_buy 이면 SELL
     → 그 외 HOLD
  2. DB 연결 불가 시 CELL_SIGNAL_MAP (하드코딩) 폴백
"""

import pandas as pd
from classifiers.regime_classifier   import compute_regime,   REGIME_LABEL
from classifiers.risk_classifier     import compute_risk,     RISK_LABEL
from classifiers.momentum_classifier import compute_momentum, MOMENTUM_LABEL


# ── 하드코딩 폴백 신호맵 (DB 없을 때) ────────────────────────
CELL_SIGNAL_MAP: dict[tuple, int] = {
    (0, 0, 0):  1,  (0, 0, 1):  1,  (0, 0, 2):  0,
    (0, 1, 0):  1,  (0, 1, 1):  0,  (0, 1, 2):  0,
    (0, 2, 0):  0,  (0, 2, 1): -1,  (0, 2, 2): -1,
    (1, 0, 0):  0,  (1, 0, 1):  0,  (1, 0, 2):  0,
    (1, 1, 0):  0,  (1, 1, 1):  0,  (1, 1, 2): -1,
    (1, 2, 0):  0,  (1, 2, 1): -1,  (1, 2, 2): -1,
    (2, 0, 0):  0,  (2, 0, 1): -1,  (2, 0, 2): -1,
    (2, 1, 0): -1,  (2, 1, 1): -1,  (2, 1, 2): -1,
    (2, 2, 0): -1,  (2, 2, 1): -1,  (2, 2, 2): -1,
}

ACTION_LABEL = {1: "BUY", 0: "HOLD", -1: "SELL"}


def _load_weights_from_db(host="localhost", user="root",
                           password="1234", database="capstone") -> dict[int, dict] | None:
    """
    cell_weight_config에서 최신 version 가중치 로드
    실패 시 None 반환 (폴백으로 하드코딩 사용)
    """
    try:
        import mysql.connector
        conn   = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT MAX(version) AS v FROM cell_weight_config")
        latest = cursor.fetchone()["v"]
        if latest is None:
            return None

        cursor.execute(
            "SELECT cell_num, weight_buy, weight_hold, weight_sell "
            "FROM cell_weight_config WHERE version = %s",
            (latest,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {r["cell_num"]: r for r in rows}
    except Exception:
        return None


def _weights_to_signal(w: dict) -> int:
    """가중치 → 신호 변환"""
    buy  = w["weight_buy"]
    sell = w["weight_sell"]
    hold = w["weight_hold"]
    if buy >= sell and buy >= hold:
        return 1    # BUY
    elif sell >= buy and sell >= hold:
        return -1   # SELL
    return 0        # HOLD


class CubicEngine:
    """
    OHLCV DataFrame → 현재 셀 좌표 + 매매 신호 반환

    사용법:
        engine = CubicEngine()
        result = engine.analyze(close_series, volume_series)
    """

    def __init__(self, use_db: bool = True,
                 db_host: str = "localhost",
                 db_user: str = "root",
                 db_password: str = "1234",
                 db_name: str = "capstone"):
        self.use_db      = use_db
        self.db_config   = dict(host=db_host, user=db_user,
                                password=db_password, database=db_name)
        self._db_weights = None   # 캐시

    def _get_signal(self, cell_num: int, x: int, y: int, z: int) -> int:
        """DB 가중치 우선, 없으면 하드코딩 폴백"""
        if self.use_db:
            if self._db_weights is None:
                self._db_weights = _load_weights_from_db(**self.db_config)

            if self._db_weights and cell_num in self._db_weights:
                return _weights_to_signal(self._db_weights[cell_num])

        return CELL_SIGNAL_MAP.get((x, y, z), 0)

    def analyze(self, close: pd.Series, volume: pd.Series) -> dict:
        """
        실시간 분석: 마지막 날짜 기준 셀 + 신호 반환

        Returns:
            {
              "date": str,
              "regime_raw": int, "risk_raw": int, "momentum_raw": int,
              "cell_num": int, "x": str, "y": str, "z": str,
              "action": str, "action_code": int
            }
        """
        regime   = compute_regime(close)
        risk     = compute_risk(close)
        momentum = compute_momentum(close, volume)

        common = regime.index.intersection(risk.index).intersection(momentum.index)
        last   = common[-1]

        x = int(regime.loc[last])
        y = int(risk.loc[last])
        z = int(momentum.loc[last])
        cell_num    = x * 9 + y * 3 + z
        action_code = self._get_signal(cell_num, x, y, z)

        return {
            "date":          str(last.date()),
            "regime_raw":    x,
            "risk_raw":      y,
            "momentum_raw":  z,
            "cell_num":      cell_num,
            "x":             REGIME_LABEL[x],
            "y":             RISK_LABEL[y],
            "z":             MOMENTUM_LABEL[z],
            "action":        ACTION_LABEL[action_code],
            "action_code":   action_code,
        }

    def compute_all(self, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """
        백테스팅용: 전체 기간 셀 + 신호 DataFrame 반환
        """
        regime   = compute_regime(close)
        risk     = compute_risk(close)
        momentum = compute_momentum(close, volume)

        common = regime.index.intersection(risk.index).intersection(momentum.index)
        df = pd.DataFrame({
            "regime":   regime.loc[common],
            "risk":     risk.loc[common],
            "momentum": momentum.loc[common],
        }).dropna().astype(int)

        df["cell"] = df["regime"] * 9 + df["risk"] * 3 + df["momentum"]
        df["signal"] = df.apply(
            lambda r: self._get_signal(
                r["cell"], r["regime"], r["risk"], r["momentum"]
            ),
            axis=1,
        )
        return df
