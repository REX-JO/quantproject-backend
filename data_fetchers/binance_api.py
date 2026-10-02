import requests
import pandas as pd
import numpy as np
import os
import joblib

# 引入我們寫好的 BTC 特徵工程函數
from models.cryptos.btc_inference_features import build_inference_features

def get_binance_klines(symbol, interval="1h", limit=100):
    """向幣安 REST API 請求 K 線資料"""
    url = "https://data-api.binance.vision/api/v3/klines"
    response = requests.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError(
            f"Binance 未回傳 {symbol} K 線："
            f"status={response.status_code}、response={response.text[:300]}"
        )
    
    # 解析幣安回傳的格式
    df = pd.DataFrame(data, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 轉換型態為浮點數
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    return df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

def fetch_live_features():
    """抓取 USDC 和 TUSD 資料，並即時計算出模型需要的特徵"""
    print("📡 正在從幣安獲取最新行情資料...")
    
    # 1. 獲取原始 K 線
    df_usdc = get_binance_klines("USDCUSDT", interval="1h", limit=100)
    df_tusd = get_binance_klines("TUSDUSDT", interval="1h", limit=100)

    # 重新命名以符合模型特徵名稱
    df_usdc = df_usdc.rename(columns={'Open': 'USDCUSDT_Open', 'High': 'USDCUSDT_High', 'Low': 'USDCUSDT_Low', 'Close': 'USDCUSDT_Price', 'Volume': 'USDCUSDT_Volume'})
    df_tusd = df_tusd.rename(columns={'Open': 'TUSDUSDT_Open', 'High': 'TUSDUSDT_High', 'Low': 'TUSDUSDT_Low', 'Close': 'TUSDUSDT_Price', 'Volume': 'TUSDUSDT_Volume'})

    # 根據時間合併兩張表
    df = pd.merge(df_usdc, df_tusd, on='timestamp', how='inner')

    # ========================================================
    # 2. 特徵工程 (Feature Engineering)
    # ========================================================
    df['Spread'] = df['USDCUSDT_Price'] - df['TUSDUSDT_Price']
    df['USDC_Deviation'] = df['USDCUSDT_Price'] - 1.0
    df['TUSD_Deviation'] = df['TUSDUSDT_Price'] - 1.0
    
    df['USDC_Candle_Body'] = df['USDCUSDT_Price'] - df['USDCUSDT_Open']
    df['TUSD_Candle_Body'] = df['TUSDUSDT_Price'] - df['TUSDUSDT_Open']
    df['USDC_Amplitude'] = df['USDCUSDT_High'] - df['USDCUSDT_Low']
    df['TUSD_Amplitude'] = df['TUSDUSDT_High'] - df['TUSDUSDT_Low']

    window = 24
    df['USDC_Volatility'] = df['USDCUSDT_Price'].rolling(window=window).std().fillna(0)
    df['TUSD_Volatility'] = df['TUSDUSDT_Price'].rolling(window=window).std().fillna(0)
    df['USDC_Rolling_Min'] = df['USDCUSDT_Low'].rolling(window=window).min().fillna(1.0)
    df['USDC_Rolling_Max'] = df['USDCUSDT_High'].rolling(window=window).max().fillna(1.0)
    df['TUSD_Rolling_Min'] = df['TUSDUSDT_Low'].rolling(window=window).min().fillna(1.0)
    df['TUSD_Rolling_Max'] = df['TUSDUSDT_High'].rolling(window=window).max().fillna(1.0)
    
    df['USDC_Dist_to_Min'] = df['USDCUSDT_Price'] - df['USDC_Rolling_Min']
    df['TUSD_Dist_to_Min'] = df['TUSDUSDT_Price'] - df['TUSD_Rolling_Min']
    df['USDC_Price_Position'] = (df['USDCUSDT_Price'] - df['USDC_Rolling_Min']) / (df['USDC_Rolling_Max'] - df['USDC_Rolling_Min'] + 1e-8)
    df['TUSD_Price_Position'] = (df['TUSDUSDT_Price'] - df['TUSD_Rolling_Min']) / (df['TUSD_Rolling_Max'] - df['TUSD_Rolling_Min'] + 1e-8)

    df['USDC_Volume_Change'] = df['USDCUSDT_Volume'].pct_change().fillna(0)
    df['TUSD_Volume_Change'] = df['TUSDUSDT_Volume'].pct_change().fillna(0)
    df['USDC_Panic_Drop'] = df['USDCUSDT_Price'].diff().apply(lambda x: min(x, 0)).fillna(0)
    df['TUSD_Panic_Drop'] = df['TUSDUSDT_Price'].diff().apply(lambda x: min(x, 0)).fillna(0)

    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        iso_path = os.path.join(base_dir, 'models', 'stablecoins', 'weights', 'stablecoin_iso_forest_package.pkl')
        
        iso_package = joblib.load(iso_path)
        
        if isinstance(iso_package, dict):
            iso_model = iso_package.get('model', iso_package.get('iso_forest'))
            iso_features = iso_package.get('features', iso_package.get('feature_cols')) 
            iso_scaler = iso_package.get('scaler', None)
        else:
            iso_model = iso_package
            iso_features = getattr(iso_model, 'feature_names_in_', None)
            iso_scaler = None
            
        if iso_features is None:
            iso_features = ['Spread', 'USDC_Deviation', 'USDC_Volatility', 'TUSD_Volatility'] 
            
        iso_data = df[iso_features].fillna(0)
        
        if iso_scaler is not None:
            iso_data = iso_scaler.transform(iso_data)
            
        df['anomaly_score'] = -iso_model.decision_function(iso_data)
        print("✅ 成功應用 IsolationForest 計算即時異常分數！")
        
    except Exception as e:
        print(f"⚠️ 載入 IsolationForest 模型失敗，異常分數暫時以 0.0 代替。錯誤: {e}")
        df['anomaly_score'] = 0.0

    df = df.replace([np.inf, -np.inf], 0)
    df_without_missing = df.dropna()
    if len(df_without_missing) < 24:
        missing_columns = {
            column: int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        }
        raise ValueError(
            "穩定幣資料清理後不足24小時："
            f"USDC={len(df_usdc)}、TUSD={len(df_tusd)}、"
            f"合併後={len(df)}、完整列={len(df_without_missing)}、"
            f"缺值欄位={missing_columns}"
        )
    df_clean = df_without_missing.tail(24)
    
    return df_clean, df

def fetch_btc_features():
    """從幣安抓取 BTC 歷史 K 線，並套用即時推論專用的特徵工程。"""
    print("📡 正在從幣安獲取 BTC 最新行情與深度特徵...")
    url = "https://data-api.binance.vision/api/v3/klines"
    
    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": 150
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'close_time', 'quote_asset_volume', 'number_of_trades', 
                   'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(data, columns=columns)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                        'quote_asset_volume', 'number_of_trades', 
                        'taker_buy_base', 'taker_buy_quote']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df_features = build_inference_features(df)
        df_kline_100h = df.tail(100).copy()
        
        return df_features, df_kline_100h
        
    except Exception as e:
        raise RuntimeError(f"獲取或處理 BTC 資料失敗: {e}")
