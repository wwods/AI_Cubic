"""
trading_env.py — PPO 강화학습용 커스텀 Gym 환경

상태 공간 (4차원, 논문 3.4절):
  [cell_x/2, cell_y/2, cell_z/2, position]
  - cell_x: 0=Bull, 1=Side, 2=Bear  → /2 로 0~1 정규화
  - cell_y: 0=Low,  1=Mid,  2=High  → /2 로 0~1 정규화
  - cell_z: 0=Strong,1=Normal,2=Weak → /2 로 0~1 정규화
  - position: 0=미보유, 1=보유

행동 공간 (3 이산):
  0=BUY, 1=HOLD, 2=SELL

보상 함수 (논문 3.4절):
  R = Sharpe Ratio - λ × MDD  (λ=0.5)
  - 에피소드 종료 시 최종 보상 계산
  - 중간 스텝: 일별 수익률 기반 즉시 보상
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    3D Cubic 시스템 기반 주식 거래 환경
    
    사용법:
        env = TradingEnv(cell_df, close)
        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 cell_df: pd.DataFrame,
                 close: pd.Series,
                 initial_capital: float = 10_000_000,
                 transaction_cost: float = 0.002,
                 lambda_mdd: float = 0.5):
        """
        Args:
            cell_df:          determine_cell_signals() 반환값
                              columns: [regime, risk, momentum, cell, signal]
            close:            종가 시계열
            initial_capital:  초기 자본금
            transaction_cost: 거래 수수료
            lambda_mdd:       MDD 페널티 가중치 (논문: 0.5)
        """
        super().__init__()

        # 공통 인덱스 정렬
        common = cell_df.index.intersection(close.index)
        self.cell_df    = cell_df.loc[common].reset_index(drop=True)
        self.close      = close.loc[common].reset_index(drop=True)
        self.n_steps    = len(self.cell_df)

        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.lambda_mdd       = lambda_mdd

        # 행동 공간: 0=BUY, 1=HOLD, 2=SELL
        self.action_space = spaces.Discrete(3)

        # 상태 공간: [cell_x/2, cell_y/2, cell_z/2, position]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 에피소드 상태
        self.current_step  = 0
        self.cash          = initial_capital
        self.shares        = 0
        self.position      = 0   # 0=미보유, 1=보유
        self.portfolio_values: list[float] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step     = 0
        self.cash             = self.initial_capital
        self.shares           = 0
        self.position         = 0
        self.portfolio_values = [self.initial_capital]
        return self._get_obs(), {}

    def step(self, action: int):
        """
        action: 0=BUY, 1=HOLD, 2=SELL
        """
        price = float(self.close.iloc[self.current_step])
        row   = self.cell_df.iloc[self.current_step]

        # ── 거래 실행 ──
        if action == 0 and self.position == 0 and self.cash > 0:   # BUY
            self.shares   = int(self.cash * (1 - self.transaction_cost) / price)
            self.cash    -= self.shares * price * (1 + self.transaction_cost)
            self.position = 1

        elif action == 2 and self.position == 1 and self.shares > 0:  # SELL
            self.cash    += self.shares * price * (1 - self.transaction_cost)
            self.shares   = 0
            self.position = 0

        # ── 포트폴리오 가치 ──
        total_value = self.cash + self.shares * price
        self.portfolio_values.append(total_value)

        # ── 즉시 보상: 일별 수익률 ──
        prev_value = self.portfolio_values[-2]
        step_reward = (total_value - prev_value) / prev_value

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # ── 에피소드 종료 시 최종 보상 ──
        if done:
            pv = pd.Series(self.portfolio_values)
            reward = self._compute_final_reward(pv)
        else:
            reward = float(step_reward)

        return self._get_obs(), reward, done, False, {
            "total_value": total_value,
            "position":    self.position,
        }

    def _get_obs(self) -> np.ndarray:
        """현재 상태 반환"""
        if self.current_step >= self.n_steps:
            return np.zeros(4, dtype=np.float32)

        row = self.cell_df.iloc[self.current_step]
        return np.array([
            row["regime"]   / 2.0,   # 0~1 정규화
            row["risk"]     / 2.0,
            row["momentum"] / 2.0,
            float(self.position),
        ], dtype=np.float32)

    def _compute_final_reward(self, pv: pd.Series) -> float:
        """
        R = Sharpe Ratio - λ × MDD  (논문 3.4절)
        """
        returns = pv.pct_change().dropna()

        # Sharpe Ratio (연환산)
        sharpe = float((returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252))

        # MDD
        cummax   = pv.cummax()
        drawdown = (pv - cummax) / (cummax + 1e-10)
        mdd      = float(abs(drawdown.min()))

        reward = sharpe - self.lambda_mdd * mdd
        return reward
