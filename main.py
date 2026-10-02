from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# 匯入資料擷取與推論模組
from models.stablecoins.predictor import get_stablecoin_model_info, run_ensemble_risk_analysis
from data_fetchers.binance_api import fetch_live_features
from data_fetchers.crypto_api import fetch_crypto_features
from models.cryptos.cryptos_predictor import load_crypto_models, predict_latest_score

base_dir = Path(__file__).parent
checkpoint_dir = base_dir / "models" / "cryptos" / "weights"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.crypto_models = load_crypto_models(checkpoint_dir)
    print("BTC、ETH、SOL、XRP 模型載入成功")
    yield
    app.state.crypto_models.clear()


app = FastAPI(
    title="量化風控與行情預測 API - 單頁儀表板整合版",
    lifespan=lifespan,
)

# 設定 CORS 允許前端跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok"}

# ------------------------------------------------------------
# 單一總覽 API 網址（前端只要請求這個即可）
# ------------------------------------------------------------
@app.get("/api/v1/dashboard/overview")
async def get_dashboard_overview(request: Request):
    try:
        # ==================== A. 穩定幣脫鉤風險分析 ====================
        df_stable_latest_24h, df_stable_kline_100h = fetch_live_features()
        stablecoin_results = run_ensemble_risk_analysis(df_stable_latest_24h)
        
        # 確保 K 線依時間由舊到新排序，使最新時間點靠右
        if 'timestamp' in df_stable_kline_100h.columns:
            df_stable_kline_100h = df_stable_kline_100h.sort_values('timestamp')

        usdc_kline_list = []
        tusd_kline_list = []
        for _, row in df_stable_kline_100h.iterrows():
            # 幣安 API 回傳為 UTC 時間，轉換為台灣時間 (UTC+8)
            t = pd.to_datetime(row['timestamp'])
            t_local = t + pd.Timedelta(hours=8) if t.tzinfo is None else t.tz_convert('Asia/Taipei')
            formatted_time = t_local.strftime("%Y-%m-%d %H:%M:%S")

            usdc_kline_list.append({
                "time": formatted_time,
                "open": float(row['USDCUSDT_Open']),
                "high": float(row['USDCUSDT_High']),
                "low": float(row['USDCUSDT_Low']),
                "close": float(row['USDCUSDT_Price']),
                "vol": float(row['USDCUSDT_Volume'])
            })
            tusd_kline_list.append({
                "time": formatted_time,
                "open": float(row['TUSDUSDT_Open']),
                "high": float(row['TUSDUSDT_High']),
                "low": float(row['TUSDUSDT_Low']),
                "close": float(row['TUSDUSDT_Price']),
                "vol": float(row['TUSDUSDT_Volume'])
            })
        
        sys_max_prob = max(
            stablecoin_results.get('usdc_ensemble_depeg_probability', 0), 
            stablecoin_results.get('tusd_ensemble_depeg_probability', 0)
        )

        stablecoin_data = {
            "system_max_risk_probability": sys_max_prob,
            **stablecoin_results,
            "usdc_kline_data": usdc_kline_list,
            "tusd_kline_data": tusd_kline_list,
            "model_info": get_stablecoin_model_info(),
        }

        # ==================== B. 四幣種 4 小時方向分數 ====================
        crypto_data = []
        for symbol, bundle in request.app.state.crypto_models.items():
            feature_rows, kline_rows = fetch_crypto_features(symbol)
            result = predict_latest_score(bundle, feature_rows)
            raw_up_score = float(result["up_score"])
            up_score = round(raw_up_score * 100, 2)
            display_threshold = round(bundle.display_threshold * 100, 2)
            threshold_gap = round(
                (raw_up_score - bundle.display_threshold) * 100,
                2,
            )
            model_data_until = pd.to_datetime(
                feature_rows.index[-1], utc=True
            ).tz_convert("Asia/Taipei")

            kline_rows = kline_rows.sort_values("open_time")
            kline_data = []
            for _, row in kline_rows.iterrows():
                local_time = pd.to_datetime(row["open_time"], utc=True).tz_convert("Asia/Taipei")
                kline_data.append({
                    "time": local_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "vol": float(row["volume"]),
                })

            crypto_data.append({
                "coin": symbol,
                "current_price": kline_data[-1]["close"],
                "prediction_horizon_hours": 4,
                "up_score": up_score,
                "trend_label": result["trend_label"],
                "display_threshold": display_threshold,
                "threshold_gap": threshold_gap,
                "model_data_until": model_data_until.strftime("%Y-%m-%d %H:%M:%S"),
                "kline_interval": "1h",
                "kline_data": kline_data,
                "model_info": {
                    "model_name": "Transformer",
                    "prediction_target": "4 小時後收盤價是否高於目前收盤價",
                    "target_definition": bundle.target_definition,
                    "prediction_horizon_hours": 4,
                    "input_kline_count": bundle.sequence_length,
                    "feature_count": len(bundle.feature_columns),
                    "display_threshold": display_threshold,
                    "validation_roc_auc": bundle.validation_auc,
                    "training_coverage": bundle.training_coverage,
                    "source_checkpoint": bundle.source_checkpoint,
                },
            })

        # ==================== C. 統整回傳單一 JSON ====================
        return {
            "success": True,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": {
                "stablecoins": stablecoin_data,
                "cryptos": crypto_data
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error_message": str(e)
        }
