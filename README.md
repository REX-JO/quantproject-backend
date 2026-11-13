# 加密貨幣走勢預測與穩定幣脫鉤風險預警系統：後端

銘傳大學人工智慧應用學系專題「基於機器學習與深度學習之加密貨幣走勢預測與穩定幣脫鉤風險預警系統」的後端，以 FastAPI 撰寫，負責特徵計算與模型推論，部署在 Render。

- 公開網站：https://quantproject-frontend.onrender.com
- 前端程式：https://github.com/REX-JO/quantproject-frontend

## API

| 方法 | 路徑 | 說明 |
| --- | --- | --- |
| POST | `/api/v1/dashboard/overview/from-klines` | 接收前端傳來的六個交易對 K 線，回傳穩定幣與主流幣的分析結果。網站使用這一支。 |
| GET | `/api/v1/dashboard/overview` | 由後端自行向 Binance 取得 K 線。部署在 Render 時會被 Binance 拒絕，保留供本機測試。 |
| GET、HEAD | `/health` | 健康檢查，供 Render 確認服務正常。 |

成功時回傳 `success`、`timestamp` 與 `data`，其中 `data.stablecoins` 是穩定幣結果，`data.cryptos` 是四個主流幣的結果。發生錯誤時回傳 `success: false` 與 `error_message`。

## 模型

### 穩定幣（`models/stablecoins/`）

USDC 與 TUSD 各有一組模型：

| 用途 | 模型 | 特徵數 |
| --- | --- | --- |
| 分類 | Transformer，門檻 0.995 與 0.99 各一個 | USDC 9 項、TUSD 11 項 |
| 分類 | XGBoost | 32 項 |
| 迴歸：估計未來最低價 | Transformer | 8 項 |
| 迴歸：估計未來最低價 | XGBoost | 32 項 |

Transformer 以最近 24 根一小時 K 線為輸入，XGBoost 使用最新一小時的特徵。另以孤立森林計算每小時的異常分數，作為分類模型的輸入特徵。各模型的門檻與預測範圍不同，完整定義列在網站的模型說明。

### 主流幣（`models/cryptos/`）

BTC、ETH、SOL、XRP 各有一個 Transformer，以最近 24 根一小時 K 線的 53 項特徵，判斷四小時後的收盤價是否高於目前收盤價。這四個模型由組員訓練，後端負責在即時資料上重新計算相同的 53 項特徵，並只使用已收盤的 K 線。

### 分數的定位

模型輸出的分數未經機率校準，網站上標示為風險分數或方向分數，不作為投資建議。專題驗證發現，穩定幣分類模型沒有穩定勝過簡單的基準規則，例如延續上一小時狀態的規則，以及價格離門檻越近越先警報的規則。因此網站上的分數定位為風險監測的參考。

## 專案結構

| 路徑 | 內容 |
| --- | --- |
| `main.py` | API 路由與回傳格式 |
| `data_fetchers/binance_api.py` | 穩定幣 K 線解析與特徵計算 |
| `data_fetchers/crypto_api.py` | 主流幣 K 線補齊缺漏小時與特徵計算 |
| `models/stablecoins/` | 穩定幣模型的推論程式與權重 |
| `models/cryptos/` | 主流幣模型的推論程式與權重 |
| `render.yaml` | Render 部署設定 |

## 本機執行

需要 Python 3.14（見 `.python-version`）。

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

啟動後開啟 http://127.0.0.1:8000/docs ，可以直接測試 API。

## 分工

本專題由五人團隊完成。穩定幣模型、後端與部署由專題組長負責；主流幣方向模型由組員訓練。
