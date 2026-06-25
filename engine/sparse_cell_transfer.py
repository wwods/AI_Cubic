"""
sparse_cell_transfer.py — 희소 셀 탐지 + 전이 학습

흐름:
  1. 최신 버전 가중치 로드
  2. 직관적으로 이상한 셀 탐지 (희소 셀)
  3. 인접 셀 가중치 평균으로 대체 (전이)
  4. DB에 version+1 로 저장

실행:
  python engine/sparse_cell_transfer.py
  python engine/sparse_cell_transfer.py --dry-run  (저장 없이 결과만 확인)
"""

import os
import argparse
import mysql.connector
from dotenv import load_dotenv
load_dotenv()


# ── 직관적 기대 신호 (논문 기반) ────────────────────────────
# (regime, risk, momentum) → 기대 신호
# 1=BUY, 0=HOLD, -1=SELL
EXPECTED_SIGNAL = {
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


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "1234"),
        database=os.getenv("DB_NAME", "capstone"),
    )


def load_latest_weights() -> tuple[int, dict]:
    """최신 버전 가중치 로드"""
    conn   = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT MAX(version) AS v FROM cell_weight_config")
    latest = cursor.fetchone()["v"]
    cursor.execute(
        "SELECT cell_num, weight_buy, weight_hold, weight_sell "
        "FROM cell_weight_config WHERE version = %s ORDER BY cell_num",
        (latest,)
    )
    weights = {r["cell_num"]: r for r in cursor.fetchall()}
    cursor.close()
    conn.close()
    print(f"[로드] version {latest} 가중치 27개 로드 완료")
    return latest, weights


def detect_sparse_cells(weights: dict) -> list[int]:
    """
    희소 셀 탐지 기준:
      1. PPO 학습 신호가 기대 신호와 반대인 경우
      2. 지배적인 가중치가 0.4 미만 (불확실한 경우)
    """
    sparse = []
    for cell_num, w in weights.items():
        x = cell_num // 9
        y = (cell_num % 9) // 3
        z = cell_num % 3
        expected = EXPECTED_SIGNAL.get((x, y, z), 0)

        # PPO가 선택한 신호
        buy, hold, sell = w["weight_buy"], w["weight_hold"], w["weight_sell"]
        max_weight = max(buy, hold, sell)
        if buy == max_weight:   ppo_signal =  1
        elif sell == max_weight: ppo_signal = -1
        else:                    ppo_signal =  0

        # 신호 반전 or 불확실 → 희소 셀
        signal_conflict = (expected != 0 and ppo_signal != expected)
        uncertain       = max_weight < 0.40

        if signal_conflict or uncertain:
            reason = []
            if signal_conflict: reason.append(f"신호반전(기대:{expected} PPO:{ppo_signal})")
            if uncertain:       reason.append(f"불확실(최대:{max_weight:.3f})")
            sparse.append(cell_num)
            print(f"  희소셀 #{cell_num:2d} ({['Bull','Side','Bear'][x]}+{['Low','Mid','High'][y]}+{['Strong','Normal','Weak'][z]}): {', '.join(reason)}")

    return sparse


def get_adjacent_cells(cell_num: int) -> list[int]:
    """6방향 인접 셀 반환 (범위 내)"""
    x = cell_num // 9
    y = (cell_num % 9) // 3
    z = cell_num % 3
    adjacent = []
    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
        nx, ny, nz = x+dx, y+dy, z+dz
        if 0 <= nx <= 2 and 0 <= ny <= 2 and 0 <= nz <= 2:
            adjacent.append(nx*9 + ny*3 + nz)
    return adjacent


def transfer_weights(weights: dict, sparse_cells: list[int]) -> dict:
    """
    희소 셀 가중치를 인접 셀 평균으로 대체
    인접 셀 중 희소 셀이 아닌 것만 사용
    """
    new_weights = {k: dict(v) for k, v in weights.items()}

    for cell_num in sparse_cells:
        adjacent = get_adjacent_cells(cell_num)
        # 희소 셀이 아닌 인접 셀만 사용
        valid = [c for c in adjacent if c not in sparse_cells]

        if not valid:
            print(f"  셀 #{cell_num:2d}: 유효한 인접 셀 없음 → 기대 신호 기반으로 설정")
            x = cell_num // 9
            y = (cell_num % 9) // 3
            z = cell_num % 3
            expected = EXPECTED_SIGNAL.get((x, y, z), 0)
            if expected == 1:
                new_weights[cell_num] = {"weight_buy": 0.60, "weight_hold": 0.30, "weight_sell": 0.10}
            elif expected == -1:
                new_weights[cell_num] = {"weight_buy": 0.10, "weight_hold": 0.30, "weight_sell": 0.60}
            else:
                new_weights[cell_num] = {"weight_buy": 0.20, "weight_hold": 0.60, "weight_sell": 0.20}
            continue

        # 인접 셀 평균
        avg_buy  = round(sum(weights[c]["weight_buy"]  for c in valid) / len(valid), 4)
        avg_hold = round(sum(weights[c]["weight_hold"] for c in valid) / len(valid), 4)
        avg_sell = round(sum(weights[c]["weight_sell"] for c in valid) / len(valid), 4)

        # 합이 1이 되도록 정규화
        total = avg_buy + avg_hold + avg_sell
        avg_buy  = round(avg_buy  / total, 4)
        avg_hold = round(avg_hold / total, 4)
        avg_sell = round(1 - avg_buy - avg_hold, 4)

        new_weights[cell_num] = {
            "weight_buy":  avg_buy,
            "weight_hold": avg_hold,
            "weight_sell": avg_sell,
        }
        print(f"  셀 #{cell_num:2d}: 인접셀 {valid} 평균 → BUY={avg_buy:.3f} HOLD={avg_hold:.3f} SELL={avg_sell:.3f}")

    return new_weights


def save_to_db(weights: dict, base_version: int) -> int:
    """새 버전으로 DB 저장"""
    new_version = base_version + 1
    conn   = get_db_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO cell_weight_config
            (cell_num, version, weight_buy, weight_hold, weight_sell, source)
        VALUES (%s, %s, %s, %s, %s, 'transfer')
    """
    rows = [(n, new_version, w["weight_buy"], w["weight_hold"], w["weight_sell"])
            for n, w in weights.items()]
    cursor.executemany(sql, rows)
    conn.commit()
    print(f"\n[DB] version {new_version} (transfer) 저장 완료 → {len(rows)}개 셀")
    cursor.close()
    conn.close()
    return new_version


def main():
    parser = argparse.ArgumentParser(description="희소 셀 탐지 + 전이 학습")
    parser.add_argument("--dry-run", action="store_true", help="DB 저장 없이 결과만 확인")
    args = parser.parse_args()

    print("=" * 60)
    print("  희소 셀 탐지 + 전이 학습")
    print("=" * 60)

    # 1. 최신 가중치 로드
    latest_version, weights = load_latest_weights()
    original_weights = {k: dict(v) for k, v in weights.items()}

    # 2. 희소 셀 수렴할 때까지 반복 전이 학습
    MAX_ITER = 5
    all_changed = set()
    new_weights = weights

    for iteration in range(1, MAX_ITER + 1):
        print(f"\n[{iteration}회차] 희소 셀 탐지 중...")
        sparse_cells = detect_sparse_cells(new_weights)
        print(f"  → {len(sparse_cells)}개 희소 셀: {sparse_cells}")

        if not sparse_cells:
            print(f"  ✓ {iteration}회차에서 수렴 완료!")
            break

        print(f"\n[{iteration}회차] 전이 학습 적용 중...")
        new_weights = transfer_weights(new_weights, sparse_cells)
        all_changed.update(sparse_cells)

        if iteration == MAX_ITER:
            print(f"\n  최대 반복({MAX_ITER}회) 도달. 남은 희소 셀 강제 교정...")
            for cell_num in sparse_cells:
                x = cell_num // 9
                y = (cell_num % 9) // 3
                z = cell_num % 3
                expected = EXPECTED_SIGNAL.get((x, y, z), 0)
                if expected == 1:
                    new_weights[cell_num] = {"weight_buy": 0.60, "weight_hold": 0.30, "weight_sell": 0.10}
                elif expected == -1:
                    new_weights[cell_num] = {"weight_buy": 0.10, "weight_hold": 0.30, "weight_sell": 0.60}
                else:
                    new_weights[cell_num] = {"weight_buy": 0.20, "weight_hold": 0.60, "weight_sell": 0.20}
                all_changed.add(cell_num)
                print(f"  셀 #{cell_num:2d} 강제 교정 → 기대신호: {expected}")

    # 3. 결과 출력
    if all_changed:
        print(f"\n[결과] 총 {len(all_changed)}개 셀 교정 완료")
        print(f"{'셀':>4} {'이전 BUY':>10} {'이전 SELL':>10} {'이후 BUY':>10} {'이후 SELL':>10}")
        print("-" * 50)
        for cell_num in sorted(all_changed):
            o = original_weights[cell_num]
            n = new_weights[cell_num]
            print(f"  #{cell_num:2d}  {o['weight_buy']:>8.3f}  {o['weight_sell']:>9.3f}  {n['weight_buy']:>9.3f}  {n['weight_sell']:>9.3f}")
    else:
        print("\n희소 셀 없음. 전이 학습 불필요.")
        return

    # 4. DB 저장
    if not args.dry_run:
        new_version = save_to_db(new_weights, latest_version)
        print(f"\n완료! 이제 CubicEngine이 version {new_version} 가중치를 사용합니다.")
    else:
        print("\n[dry-run] DB 저장 건너뜀")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()