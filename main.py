"""
3D Cubic AI Engine - FastAPI 서버
Spring Boot 백엔드에서 HTTP POST로 호출
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze, backtest
from engine.cubic_engine import CubicEngine
from dotenv import load_dotenv
import os

load_dotenv()

engine = CubicEngine(
    use_db=True,
    db_host=os.getenv("DB_HOST", "localhost"),
    db_user=os.getenv("DB_USER", "root"),
    db_password=os.getenv("DB_PASSWORD", "1234"),
    db_name=os.getenv("DB_NAME", "capstone"),
)

app = FastAPI(
    title="3D Cubic AI Engine",
    description="시장 레짐(HMM) × 복합 리스크(GARCH) × 모멘텀(RSI·MACD) 기반 매매 신호 엔진",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Spring Boot 주소로 좁혀도 됨
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router,   prefix="/analyze",   tags=["analyze"])
app.include_router(backtest.router,  prefix="/backtest",  tags=["backtest"])


@app.get("/health")
def health():
    return {"status": "ok", "engine": "3D Cubic v1.0"}
