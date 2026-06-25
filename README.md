# AI Cubic — 3D 큐브 기반 주식 거래 보조 시스템

3D 큐브 구조로 시장 상황을 분류하고 PPO 강화학습으로 셀별 최적 매매 가중치를 산출하는 주식 거래 보조 시스템.  
2026 KSII 봄 학술대회 발표 논문 기반 프로젝트.

## Overview

시장을 세 가지 축으로 분류해 27개 독립 셀을 구성하고, 각 셀에 최적화된 매매 전략을 적용한다.

| 축 | 지표 | 분류 |
|---|---|---|
| X (시장 국면) | HMM 20일 롤링 | Bull / Side / Bear |
| Y (복합 위험) | GARCH 40% + VaR 35% + MDD 25% | Low / Mid / High |
| Z (모멘텀) | RSI 40% + MACD 35% + 거래량 25% | Weak / Neutral / Strong |

보상 함수: `R = Sharpe Ratio − 0.5 × MDD`

## Results

| 종목 | 전략 | Total Return | Sharpe | MDD |
|---|---|---|---|---|
| NVDA | 3D Cubic | 473.15% | 1.527 | -20.28% |
| NVDA | Buy & Hold | 1331.28% | 1.285 | -66.33% |
| 삼성전자 | 3D Cubic | 60.90% | 0.595 | -32.83% |
| 삼성전자 | Buy & Hold | 59.96% | 0.496 | -42.66% |

## Tech Stack

- Python, yfinance, hmmlearn, arch
- PPO (Reinforcement Learning)
- Backtest period: 2021–2025

## Team

- 정지훈, 백주용, 임상오
- 지도교수: 강민구 (한신대학교 소프트웨어융합학부)

## Paper

2026 KSII 봄 학술대회 — *3D 큐빅 상황 가중치 최적화 기반 주식거래 보조 시스템 설계*
