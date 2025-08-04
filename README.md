# crypto_backtesting

## 專案簡介
本專案是一套針對加密貨幣的回測與資料分析平台，支援多種回測模式、參數優化、走勢分析與視覺化，適合用於量化策略開發、驗證與研究。

---

## 專案結構

```
crypto_backtesting/
│
├── main.py                      # 主程式入口
├── requirements.txt             # 依賴套件
├── config.py                    # 參數設定
├── analysis_tools/              # 資料分析、回測、視覺化等核心模組
│   ├── data_analysis.py
│   ├── generate_image.py
│   ├── crypto_data_service.py
│   ├── backtest_engine.py
│   ├── crypto_exchange_data_service.py
│   ├── utilities.py
│   └── glassnode_data_service.py
├── overfitting_prevention/      # 防止過度擬合相關模組
│   ├── cross_validator.py
│   ├── parameter_plateau.py
│   └── walk_forward_analysis.py
├── backtest_results/            # 回測結果輸出
├── data/                        # 原始資料
└── ...
```

---

## 主要功能

- 加密貨幣歷史資料抓取與分析
- 策略回測（支援多種交易所與資料來源）
- 參數優化（Parameter Plateau）
- Walk Forward Analysis
- 交叉驗證（Cross Validation）
- 資產淨值曲線與熱力圖繪製
- Dashboard 視覺化（可選）

---

## 安裝方式

1. 複製本專案到本地
2. 安裝依賴套件：
   ```bash
   pip install -r requirements.txt
   ```

---

## 使用說明

執行主程式：
```bash
python main.py
```
可於 `main.py` 中調整各項參數（如資產、資料來源、回測模式等），以符合你的需求。

---

## 主要參數說明

- `asset_currency`：標的資產（如 BTC）
- `data_source`：資料來源（如 exchange、glassnode）
- `backtest_mode`：是否啟用回測
- `cross_validate`：是否啟用交叉驗證
- `parameter_plateau`：是否啟用參數優化
- `walk_forward`：是否啟用 Walk Forward Analysis
- 其他參數請參考 `main.py` 內註解

---

## 依賴套件

請參考 `requirements.txt`，主要依賴如 pandas、numpy、matplotlib、ccxt、scikit-learn、plotly 等。

---

## 資料來源

- 支援多家交易所（如 Binance）
- 支援 Glassnode 等鏈上數據

---

## 輸出結果

- 回測結果（可於 `backtest_results/` 目錄下查看）
- 圖表與熱力圖
- Dashboard（如啟用）

---