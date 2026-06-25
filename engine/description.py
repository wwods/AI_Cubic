"""
description.py — 큐빅 셀 좌표를 자연어 설명으로 변환
누구나 이해할 수 있는 표현 사용
"""

# ── 축별 자연어 표현 ─────────────────────────────────────────

REGIME_TEXT = {
    "Bull": "지금 시장 전체가 오르는 분위기예요",
    "Side": "지금 시장이 뚜렷한 방향 없이 보합세예요",
    "Bear": "지금 시장 전체가 내려가는 분위기예요",
}

RISK_TEXT = {
    "Low":  "이 종목의 투자 위험은 낮은 편이에요",
    "Mid":  "이 종목의 투자 위험은 보통 수준이에요",
    "High": "이 종목의 투자 위험이 높은 편이에요",
}

MOMENTUM_TEXT = {
    "Strong": "주가가 오르려는 힘도 강해요",
    "Normal": "주가의 움직임은 평범한 수준이에요",
    "Weak":   "주가가 오르려는 힘이 약한 상태예요",
}

ACTION_CONCLUSION = {
    "BUY": {
        "label": "지금이 사기 좋은 타이밍이에요",
        "reason": {
            ("Bull", "Low",  "Strong"): "시장, 위험도, 모멘텀 모든 조건이 좋아요. 적극적으로 매수를 고려해보세요.",
            ("Bull", "Low",  "Normal"): "시장 분위기와 위험도가 좋아요. 매수를 고려해볼 만해요.",
            ("Bull", "Mid",  "Strong"): "시장이 오르는 분위기이고 모멘텀도 강해요. 위험도를 감안하더라도 매수할 만해요.",
        }
    },
    "HOLD": {
        "label": "지금은 기다려보는 게 나을 것 같아요",
        "reason": {}
    },
    "SELL": {
        "label": "지금은 파는 것이 나을 것 같아요",
        "reason": {
            ("Bull", "High", "Normal"): "시장은 오르고 있지만 이 종목의 위험이 높아요. 수익을 지키기 위해 파는 것을 권장해요.",
            ("Bull", "High", "Weak"):   "시장은 오르고 있지만 위험도가 높고 상승 힘도 약해요. 지금 파는 것이 안전해요.",
            ("Bear", "High", "Weak"):   "시장도 내려가고 위험도 높고 상승 힘도 없어요. 지금 파는 것을 강력히 권장해요.",
        }
    },
}


def generate_description(x: str, y: str, z: str, action: str) -> dict:
    """
    셀 좌표 → 자연어 설명 생성

    Returns:
        {
          "regime_desc":   str,  # 시장 분위기 설명
          "risk_desc":     str,  # 위험도 설명
          "momentum_desc": str,  # 모멘텀 설명
          "conclusion":    str,  # 최종 결론 한 줄
          "detail":        str,  # 상세 이유 설명
          "summary":       str,  # 전체 요약 (3줄)
        }
    """
    regime_desc   = REGIME_TEXT.get(x, "")
    risk_desc     = RISK_TEXT.get(y, "")
    momentum_desc = MOMENTUM_TEXT.get(z, "")

    action_info = ACTION_CONCLUSION.get(action, ACTION_CONCLUSION["HOLD"])
    conclusion  = action_info["label"]

    # 특수 케이스 상세 이유
    detail = action_info["reason"].get((x, y, z), "")

    # 기본 상세 이유 (특수 케이스 없을 때)
    if not detail:
        if action == "BUY":
            detail = "현재 시장 상황을 종합적으로 분석한 결과 매수가 유리한 시점이에요."
        elif action == "SELL":
            detail = "현재 시장 상황을 종합적으로 분석한 결과 매도가 유리한 시점이에요."
        else:
            detail = "뚜렷한 매수·매도 신호가 없어요. 상황을 좀 더 지켜보는 것을 권장해요."

    # 전체 요약 (3줄)
    summary = f"{regime_desc}. {risk_desc}. {momentum_desc}. {conclusion}."

    return {
        "regime_desc":   regime_desc,
        "risk_desc":     risk_desc,
        "momentum_desc": momentum_desc,
        "conclusion":    conclusion,
        "detail":        detail,
        "summary":       summary,
    }
