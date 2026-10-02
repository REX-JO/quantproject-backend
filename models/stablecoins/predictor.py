# ============================================================
# prediction_service.py
# Stablecoin Depeg Risk Prediction Backend
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Backend Device:", DEVICE)


# ============================================================
# USDC Transformer Classification (USDC 0.99 / 0.995)
# ============================================================
class StablecoinTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x).squeeze(-1)


# ============================================================
# TUSD Transformer Classification (TUSD 0.99 / 0.995)
# ============================================================
class TUSDTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.classifier(x).squeeze(-1)


# ============================================================
# Transformer Regression (USDC / TUSD 共用架構)
# ============================================================
class TransformerRegressorV2(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=1024, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.regressor(x[:, -1, :])


# ============================================================
# Global Cache & Model Loader
# ============================================================
_models_loaded = False
models_cache = {}


def get_stablecoin_model_info():
    """Return display metadata that can be verified from the deployed code and weights."""
    metric_note = "目前權重檔未記錄驗證 ROC-AUC"

    def coin_info(transformer_classifier_features):
        return {
            "prediction_horizon_hours": 6,
            "prediction_target": "未來 6 小時最低價與脫鉤風險",
            "kline_interval": "1h",
            "input_kline_count": 24,
            "validation_roc_auc": None,
            "validation_metric_note": metric_note,
            "models": [
                {
                    "model_key": "ensemble",
                    "display_name": "XGBoost+Transformer",
                    "components": ["XGBoost", "Transformer 0.995"],
                    "feature_counts": {
                        "xgboost": 32,
                        "transformer_classifier": transformer_classifier_features,
                        "transformer_regressor": 8,
                    },
                },
                {
                    "model_key": "transformer_0995",
                    "display_name": "Transformer0.995",
                    "components": ["Transformer 0.995"],
                    "feature_counts": {
                        "classifier": transformer_classifier_features,
                        "regressor": 8,
                    },
                },
                {
                    "model_key": "transformer_099",
                    "display_name": "Transformer0.99",
                    "components": ["Transformer 0.99"],
                    "feature_counts": {
                        "classifier": transformer_classifier_features,
                        "regressor": 8,
                    },
                },
                {
                    "model_key": "xgboost",
                    "display_name": "XGBoost",
                    "components": ["XGBoost"],
                    "feature_counts": {
                        "classifier": 32,
                        "regressor": 32,
                    },
                },
            ],
        }

    return {
        "USDC": coin_info(transformer_classifier_features=9),
        "TUSD": coin_info(transformer_classifier_features=11),
    }

def load_transformer(model_class, path, input_dim):
    model = model_class(input_dim=input_dim)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def load_all_models():
    global _models_loaded, models_cache
    if _models_loaded:
        return

    print("\n========== Loading Models ==========")
    base_path = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_path, "weights")

    try:
        # --- Scalers (修正為實際檔名 scaler_reg.pkl) ---
        models_cache["usdc_scaler_cls_99"] = joblib.load(os.path.join(weights_dir, "scaler_cls_0.99.pkl"))
        models_cache["usdc_scaler_cls_995"] = joblib.load(os.path.join(weights_dir, "scaler_cls_0.995.pkl"))
        models_cache["usdc_scaler_reg"] = joblib.load(os.path.join(weights_dir, "scaler_reg.pkl"))
        
        models_cache["tusd_scaler_cls_99"] = joblib.load(os.path.join(weights_dir, "tusd_scaler_cls_0.99.pkl"))
        models_cache["tusd_scaler_cls_995"] = joblib.load(os.path.join(weights_dir, "tusd_scaler_cls_0.995.pkl"))
        models_cache["tusd_scaler_reg"] = joblib.load(os.path.join(weights_dir, "tusd_scaler_reg.pkl"))

        # --- XGBoost ---
        models_cache["usdc_xgb_cls"] = joblib.load(os.path.join(weights_dir, "xgboost_depeg_classifier.pkl"))
        models_cache["usdc_xgb_reg"] = joblib.load(os.path.join(weights_dir, "xgboost_price_regressor.pkl"))
        models_cache["tusd_xgb_cls"] = joblib.load(os.path.join(weights_dir, "xgboost_tusd_classifier.pkl"))
        models_cache["tusd_xgb_reg"] = joblib.load(os.path.join(weights_dir, "xgboost_tusd_regressor.pkl"))

        # --- Transformers ---
        models_cache["usdc_trans_cls_99"] = load_transformer(StablecoinTransformer, os.path.join(weights_dir, "transformer_classifier_0.99.pth"), input_dim=9)
        models_cache["usdc_trans_cls_995"] = load_transformer(StablecoinTransformer, os.path.join(weights_dir, "transformer_classifier_0.995.pth"), input_dim=9)
        models_cache["usdc_trans_reg"] = load_transformer(TransformerRegressorV2, os.path.join(weights_dir, "transformer_regressor.pth"), input_dim=8)

        models_cache["tusd_trans_cls_99"] = load_transformer(TUSDTransformer, os.path.join(weights_dir, "tusd_transformer_classifier_0.99.pth"), input_dim=11)
        models_cache["tusd_trans_cls_995"] = load_transformer(TUSDTransformer, os.path.join(weights_dir, "tusd_transformer_classifier_0.995.pth"), input_dim=11)
        models_cache["tusd_trans_reg"] = load_transformer(TransformerRegressorV2, os.path.join(weights_dir, "tusd_transformer_regressor.pth"), input_dim=8)

        _models_loaded = True
        print("========== All Models Loaded ==========\n")

    except Exception as e:
        print(f"\n❌ Model Loading Error: {e}")
        raise e


# ============================================================
# Main Prediction Logic
# ============================================================
def run_ensemble_risk_analysis(df_24h: pd.DataFrame):
    load_all_models()

    if len(df_24h) < 24:
        raise ValueError("資料不足，需要至少24小時資料")

    df_window = df_24h.sort_values("timestamp").tail(24).copy()

    # =========================
    # 特徵定義區塊 (分流)
    # =========================
    features_usdc_cls = [
        'USDCUSDT_Price', 'TUSDUSDT_Price', 'Spread', 'USDC_Deviation', 'USDC_Volatility', 
        'USDC_Amplitude', 'USDC_Panic_Drop', 'USDC_Candle_Body', 'anomaly_score'
    ]

    features_tusd_cls = [
        'USDCUSDT_Price', 'TUSDUSDT_Price', 'Spread', 'TUSD_Deviation', 'TUSD_Volatility', 
        'TUSD_Amplitude', 'TUSD_Panic_Drop', 'TUSD_Candle_Body', 'TUSD_Volume_Change', 
        'TUSD_Price_Position', 'anomaly_score'
    ]

    features_8_reg = [
        'USDCUSDT_Price', 'TUSDUSDT_Price', 'Spread', 'USDC_Deviation', 'USDC_Volatility', 
        'USDC_Amplitude', 'USDC_Panic_Drop', 'anomaly_score'
    ]

    features_32_xgb = [
        'USDCUSDT_Open', 'USDCUSDT_High', 'USDCUSDT_Low', 'USDCUSDT_Price', 'USDCUSDT_Volume',
        'TUSDUSDT_Open', 'TUSDUSDT_High', 'TUSDUSDT_Low', 'TUSDUSDT_Price', 'TUSDUSDT_Volume',
        'Spread', 'USDC_Deviation', 'USDC_Volatility', 'USDC_Amplitude', 'USDC_Panic_Drop', 
        'USDC_Candle_Body', 'USDC_Volume_Change', 'USDC_Rolling_Min', 'USDC_Rolling_Max', 
        'USDC_Dist_to_Min', 'USDC_Price_Position', 'TUSD_Deviation', 'TUSD_Volatility', 
        'TUSD_Amplitude', 'TUSD_Panic_Drop', 'TUSD_Candle_Body', 'TUSD_Volume_Change',
        'TUSD_Rolling_Min', 'TUSD_Rolling_Max', 'TUSD_Dist_to_Min', 'TUSD_Price_Position', 'anomaly_score'
    ]

    # =========================
    # 資料準備與縮放 (萬能 Scaler 探針防呆)
    # =========================
    raw_usdc_cls = df_window[features_usdc_cls].values
    raw_tusd_cls = df_window[features_tusd_cls].values
    raw_reg = df_window[features_8_reg].values
    xgb_input = df_window[features_32_xgb].iloc[-1:].values

    def get_scaler_obj(scaler_item):
        if hasattr(scaler_item, "transform"):
            return scaler_item
        if isinstance(scaler_item, dict):
            for k, v in scaler_item.items():
                if hasattr(v, "transform"):
                    return v
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if hasattr(sub_v, "transform"):
                            return sub_v
        return scaler_item

    s_usdc_99 = get_scaler_obj(models_cache["usdc_scaler_cls_99"])
    s_usdc_995 = get_scaler_obj(models_cache["usdc_scaler_cls_995"])
    s_usdc_reg = get_scaler_obj(models_cache["usdc_scaler_reg"])

    s_tusd_99 = get_scaler_obj(models_cache["tusd_scaler_cls_99"])
    s_tusd_995 = get_scaler_obj(models_cache["tusd_scaler_cls_995"])
    s_tusd_reg = get_scaler_obj(models_cache["tusd_scaler_reg"])

    usdc_cls_99_input = torch.tensor(s_usdc_99.transform(raw_usdc_cls), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    usdc_cls_995_input = torch.tensor(s_usdc_995.transform(raw_usdc_cls), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    usdc_reg_scaler_x = get_scaler_obj(models_cache["usdc_scaler_reg"])
    usdc_reg_input = torch.tensor(usdc_reg_scaler_x.transform(raw_reg), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    tusd_cls_99_input = torch.tensor(s_tusd_99.transform(raw_tusd_cls), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    tusd_cls_995_input = torch.tensor(s_tusd_995.transform(raw_tusd_cls), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    tusd_reg_input = torch.tensor(s_tusd_reg.transform(raw_reg), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    current_usdc = float(df_window["USDCUSDT_Price"].iloc[-1])
    current_tusd = float(df_window["TUSDUSDT_Price"].iloc[-1])

    # =========================
    # 執行推論
    # =========================
    result = {}

    def risk_level(prob):
        if prob >= 0.7: return "🔴 嚴重脫鉤"
        elif prob >= 0.5: return "🟡 風險預警"
        else: return "🟢 狀態安全"

    def output_block(name, current, future, prob):
        return {
            f"{name}_current_price": round(current, 6),
            f"{name}_future_6h_low": round(future, 6),
            f"{name}_price_diff": round(future - current, 6),
            f"{name}_depeg_probability": round(prob * 100, 2),
            f"{name}_risk_level": risk_level(prob)
        }

    with torch.no_grad():
        # USDC 預測
        usdc_prob_99 = torch.sigmoid(models_cache["usdc_trans_cls_99"](usdc_cls_99_input)).item()
        usdc_prob_995 = torch.sigmoid(models_cache["usdc_trans_cls_995"](usdc_cls_995_input)).item()
        
        # 【學術標準反正規化 + 轉原生 float】
        usdc_reg_raw_pred = models_cache["usdc_trans_reg"](usdc_reg_input).cpu().numpy()
        usdc_scaler_obj = models_cache["usdc_scaler_reg"]
        if isinstance(usdc_scaler_obj, dict) and "scaler_y" in usdc_scaler_obj:
            usdc_true_diff = float(usdc_scaler_obj["scaler_y"].inverse_transform(usdc_reg_raw_pred.reshape(-1, 1))[0][0])
        else:
            usdc_true_diff = float(usdc_reg_raw_pred.flatten()[0])
        usdc_trans_future = current_usdc + usdc_true_diff

        # TUSD 預測
        tusd_prob_99 = torch.sigmoid(models_cache["tusd_trans_cls_99"](tusd_cls_99_input)).item()
        tusd_prob_995 = torch.sigmoid(models_cache["tusd_trans_cls_995"](tusd_cls_995_input)).item()
        
        # 【修正這裡】：補上與 USDC 相同的反正規化邏輯
        tusd_reg_raw_pred = models_cache["tusd_trans_reg"](tusd_reg_input).cpu().numpy()
        tusd_scaler_obj = models_cache["tusd_scaler_reg"]
        if isinstance(tusd_scaler_obj, dict) and "scaler_y" in tusd_scaler_obj:
            tusd_true_diff = float(tusd_scaler_obj["scaler_y"].inverse_transform(tusd_reg_raw_pred.reshape(-1, 1))[0][0])
        else:
            tusd_true_diff = float(tusd_reg_raw_pred.flatten()[0])
        tusd_trans_future = current_tusd + tusd_true_diff

    # XGBoost 預測
    usdc_xgb_prob = float(models_cache["usdc_xgb_cls"].predict_proba(xgb_input)[0][1])
    usdc_xgb_future = float(models_cache["usdc_xgb_reg"].predict(xgb_input)[0])
    tusd_xgb_prob = float(models_cache["tusd_xgb_cls"].predict_proba(xgb_input)[0][1])
    tusd_xgb_future = float(models_cache["tusd_xgb_reg"].predict(xgb_input)[0])

    # 封裝結果
    usdc_ens_prob = (usdc_prob_995 + usdc_xgb_prob) / 2
    usdc_ens_price = (usdc_trans_future + usdc_xgb_future) / 2
    result.update(output_block("usdc_ensemble", current_usdc, usdc_ens_price, usdc_ens_prob))
    result.update(output_block("usdc_transformer_0995", current_usdc, usdc_trans_future, usdc_prob_995))
    result.update(output_block("usdc_transformer_099", current_usdc, usdc_trans_future, usdc_prob_99))
    result.update(output_block("usdc_xgboost", current_usdc, usdc_xgb_future, usdc_xgb_prob))

    tusd_ens_prob = (tusd_prob_995 + tusd_xgb_prob) / 2
    tusd_ens_price = (tusd_trans_future + tusd_xgb_future) / 2
    result.update(output_block("tusd_ensemble", current_tusd, tusd_ens_price, tusd_ens_prob))
    result.update(output_block("tusd_transformer_0995", current_tusd, tusd_trans_future, tusd_prob_995))
    result.update(output_block("tusd_transformer_099", current_tusd, tusd_trans_future, tusd_prob_99))
    result.update(output_block("tusd_xgboost", current_tusd, tusd_xgb_future, tusd_xgb_prob))



    return result
