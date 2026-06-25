"""
ppo_trainer.py — 국내/해외 상위 20개 종목 통합 PPO 학습

국내 시가총액 상위 10개 + 해외 시가총액 상위 10개
모든 종목 데이터를 하나의 환경으로 합쳐서 학습 → 범용성 극대화

실행:
  python engine/ppo_trainer.py
  python engine/ppo_trainer.py --timesteps 300000
"""

import sys, os, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import mysql.connector
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.trading_env import TradingEnv
from dotenv import load_dotenv
load_dotenv()

# ── 종목 리스트 ──────────────────────────────────────────────

DOMESTIC_TOP10 = [
    ("005930.KS", "삼성전자"),
    ("000660.KS", "SK하이닉스"),
    ("005380.KS", "현대차"),
    ("035420.KS", "NAVER"),
    ("051910.KS", "LG화학"),
    ("006400.KS", "삼성SDI"),
    ("035720.KS", "카카오"),
    ("003550.KS", "LG"),
    ("028260.KS", "삼성물산"),
    ("068270.KS", "셀트리온"),
]

OVERSEAS_TOP10 = [
    ("AAPL",  "Apple"),
    ("MSFT",  "Microsoft"),
    ("NVDA",  "NVIDIA"),
    ("AMZN",  "Amazon"),
    ("GOOGL", "Alphabet"),
    ("META",  "Meta"),
    ("TSLA",  "Tesla"),
    ("BRK-B", "Berkshire Hathaway"),
    ("JPM",   "JPMorgan"),
    ("V",     "Visa"),
]

ALL_STOCKS = DOMESTIC_TOP10 + OVERSEAS_TOP10


# ── 데이터 수집 ──────────────────────────────────────────────

def get_data(ticker: str, name: str, start: str, end: str) -> pd.DataFrame | None:
    """yfinance로 데이터 수집. 실패 시 None 반환"""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) >= 90:
            print(f"  ✓ {name} ({ticker}): {len(df)}일")
            return df
        else:
            print(f"  ✗ {name} ({ticker}): 데이터 부족 ({len(df)}일)")
            return None
    except Exception as e:
        print(f"  ✗ {name} ({ticker}): 수집 실패 ({e})")
        return None


# ── 3축 분류 ─────────────────────────────────────────────────

def build_cell_df(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """3축 분류 → cell_df 생성"""
    try:
        from classifiers.regime_classifier   import compute_regime
        from classifiers.risk_classifier     import compute_risk
        from classifiers.momentum_classifier import compute_momentum
        from engine.cubic_engine             import CELL_SIGNAL_MAP

        regime   = compute_regime(close)
        risk     = compute_risk(close)
        momentum = compute_momentum(close, volume)

        common = regime.index.intersection(risk.index).intersection(momentum.index)
        df = pd.DataFrame({
            "regime":   regime.loc[common],
            "risk":     risk.loc[common],
            "momentum": momentum.loc[common],
        }).dropna().astype(int)

        df["cell"]   = df["regime"] * 9 + df["risk"] * 3 + df["momentum"]
        df["signal"] = df.apply(
            lambda r: CELL_SIGNAL_MAP.get((r["regime"], r["risk"], r["momentum"]), 0),
            axis=1
        )
        if len(df) < 60:
            return None
        return df
    except Exception as e:
        print(f"    분류 실패: {e}")
        return None


# ── 멀티 종목 환경 구성 ──────────────────────────────────────

def build_multi_env(stock_data: list[tuple]) -> DummyVecEnv:
    """
    여러 종목의 TradingEnv를 하나의 VecEnv로 묶기
    PPO가 동시에 여러 시장 상황을 학습
    """
    env_fns = []
    for cell_df, close in stock_data:
        # 클로저로 각 종목 환경 생성
        def make_env(c=cell_df, p=close):
            return lambda: TradingEnv(c, p)
        env_fns.append(make_env())

    return DummyVecEnv(env_fns)


# ── PPO 학습 ─────────────────────────────────────────────────

def train_ppo_multi(vec_env: DummyVecEnv, timesteps: int) -> PPO:
    """멀티 환경 PPO 학습"""
    print(f"\n[PPO] {vec_env.num_envs}개 종목 환경으로 학습 시작 (timesteps={timesteps:,})...")

    model = PPO(
        policy        = "MlpPolicy",
        env           = vec_env,
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        clip_range    = 0.2,
        verbose       = 1,
    )
    model.learn(total_timesteps=timesteps)
    print("  → 학습 완료")
    return model


# ── 셀별 가중치 추출 ─────────────────────────────────────────

def extract_cell_weights(model: PPO) -> dict[int, dict]:
    """27개 셀 각각의 행동 확률 추출"""
    print("\n[가중치] 27개 셀별 행동 확률 추출 중...")
    cell_weights = {}
    for cell_num in range(27):
        x = cell_num // 9
        y = (cell_num % 9) // 3
        z = cell_num % 3
        obs = np.array([x / 2.0, y / 2.0, z / 2.0, 0.0], dtype=np.float32)

        action_counts = {0: 0, 1: 0, 2: 0}
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=False)
            action_counts[int(action)] += 1

        total = sum(action_counts.values())
        cell_weights[cell_num] = {
            "weight_buy":  round(action_counts[0] / total, 4),
            "weight_hold": round(action_counts[1] / total, 4),
            "weight_sell": round(action_counts[2] / total, 4),
        }
    print(f"  → 완료 (27개 셀)")
    return cell_weights


# ── DB 저장 ──────────────────────────────────────────────────

def save_to_db(cell_weights: dict[int, dict]) -> int:
    host     = os.getenv("DB_HOST", "localhost")
    user     = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "1234")
    database = os.getenv("DB_NAME", "capstone")

    print("\n[DB] cell_weight_config 저장 중...")
    conn   = mysql.connector.connect(host=host, user=user, password=password, database=database)
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(version) FROM cell_weight_config")
    new_version = (cursor.fetchone()[0] or 0) + 1
    print(f"  → 새 version: {new_version}")

    sql = """
        INSERT INTO cell_weight_config
            (cell_num, version, weight_buy, weight_hold, weight_sell, source)
        VALUES (%s, %s, %s, %s, %s, 'ppo_multi')
    """
    rows = [(n, new_version, w["weight_buy"], w["weight_hold"], w["weight_sell"])
            for n, w in cell_weights.items()]
    cursor.executemany(sql, rows)
    conn.commit()
    print(f"  → {len(rows)}개 셀 저장 완료 (version={new_version})")
    cursor.close()
    conn.close()
    return new_version


def save_model(model: PPO, version: int):
    os.makedirs("models", exist_ok=True)
    path = f"models/ppo_cubic_v{version}"
    model.save(path)
    print(f"  → 모델 저장: {path}.zip")


# ── 메인 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3D Cubic 멀티 종목 PPO Trainer")
    parser.add_argument("--start",     default="2021-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--no-db",     action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  3D Cubic PPO 멀티 종목 학습")
    print(f"  국내 상위 10개 + 해외 상위 10개 = 20개 종목")
    print(f"  기간: {args.start} ~ {args.end}")
    print(f"  Timesteps: {args.timesteps:,}")
    print("=" * 60)

    # 1. 데이터 수집
    print("\n[데이터] 20개 종목 수집 중...")
    stock_data = []
    failed = []

    for ticker, name in ALL_STOCKS:
        df = get_data(ticker, name, args.start, args.end)
        if df is None:
            failed.append(name)
            continue

        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        cell_df = build_cell_df(close, volume)
        if cell_df is None:
            failed.append(name)
            continue

        stock_data.append((cell_df, close))

    print(f"\n  → 성공: {len(stock_data)}개 / 실패: {len(failed)}개")
    if failed:
        print(f"  실패 종목: {', '.join(failed)}")

    if len(stock_data) < 5:
        print("유효한 데이터가 너무 적어요. 인터넷 연결을 확인해주세요.")
        return

    # 2. 멀티 종목 환경 구성
    print(f"\n[환경] {len(stock_data)}개 종목 환경 구성 중...")
    vec_env = build_multi_env(stock_data)
    print(f"  → {vec_env.num_envs}개 병렬 환경 생성 완료")

    # 3. PPO 학습
    model = train_ppo_multi(vec_env, timesteps=args.timesteps)

    # 4. 가중치 추출
    cell_weights = extract_cell_weights(model)

    # 5. DB 저장
    if not args.no_db:
        version = save_to_db(cell_weights)
        save_model(model, version)
    else:
        print("\n[DB 저장 건너뜀]")
        for n, w in cell_weights.items():
            print(f"  셀 {n:2d}: BUY={w['weight_buy']:.3f} HOLD={w['weight_hold']:.3f} SELL={w['weight_sell']:.3f}")

    print("\n" + "=" * 60)
    print("  멀티 종목 PPO 학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()