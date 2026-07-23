# Theoretical-Stock-Selection

台股量化選股與紙交易專案。結合 **道氏理論（趨勢過濾）**、**橫截面動能（Cross-sectional Momentum）** 與 **系統化風控（固定停損 + 移動停損）**，以 0050 為核心配置，搭配 0050 成分股策略池進行回測與實盤前驗證。

資料來源：[FinMind](https://finmindtrade.com/) · 目標市場：台股（0050 成分股）

---

## 策略概覽

| 層級 | 內容 |
|------|------|
| **核心配置** | 定期定額買入 0050（預設 50%） |
| **策略池** | 0050 成分股中，MA50 > MA120 且 20 日動能 > 0 的標的 |
| **選股排序** | 20 / 60 / 120 日動能加權排名（0.2 / 0.3 / 0.5） |
| **持倉限制** | 最多 5 檔、等權配置（每檔約 20% 策略淨值） |
| **風控** | 固定停損 -8%、自最高價回撤 -10% 移動停損 |
| **成本模型** | 手續費 0.1425%（最低 20 元）、賣出證交稅 0.3% |

回測邏輯見 `scripts/backtest_portfolio.py`；紙交易參數對齊 `paper_trading/settings.csv`。

---

## 目錄結構

```
pythonFinmind/
├── notebooks/                  # Jupyter 分析與回測
│   ├── fetch_tw50_stock_data.ipynb   # 抓取 0050 成分股日線
│   └── portfolio_backtest.ipynb      # 動能選股 + 組合回測
├── scripts/                    # 可重用模組與 CLI
│   ├── project_paths.py        # 專案路徑常數
│   ├── notebook_setup.py       # Notebook 路徑 / .env 初始化
│   ├── backtest_portfolio.py   # 回測引擎（DCA、XIRR、停損）
│   ├── paper_trading_summary.py    # 紙交易績效摘要
│   ├── paper_trading_allocation.py # 策略池水位與下一筆買進額度
│   ├── TaiwanStockPrice.py         # FinMind 股價範例
│   └── TaiwanStockInstitutionalInvestorsBuySell.py
├── data/                       # 股價 CSV（git 忽略，本機產生）
├── paper_trading/              # 紙交易紀錄
│   ├── settings.csv            # 投入方式、停損、持倉上限等
│   ├── transactions.csv        # 每筆交易
│   └── open_positions.csv      # 目前持倉
└── .env                        # FINMIND_TOKEN（勿 commit）
```

> **規劃中**：`api/` 目錄將以 FastAPI 包裝 `scripts/` 邏輯，對外提供 REST 介面（見下方 Roadmap）。

---

## 快速開始

### 1. 環境準備

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install FinMind pandas numpy python-dotenv jupyter requests
```

### 2. FinMind Token

在專案根目錄建立 `.env`：

```env
FINMIND_TOKEN=your_token_here
```

Token 可至 [FinMind 會員頁](https://finmindtrade.com/) 取得。

### 3. 抓取資料

在 Jupyter 中執行 `notebooks/fetch_tw50_stock_data.ipynb`，會：

- 從元大 ETF 網站取得最新 0050 成分股（失敗時使用內建備援清單）
- 透過 FinMind 下載各股日線，寫入 `data/`

### 4. 回測與紙交易

```bash
# 回測（在 notebook 中呼叫 backtest_portfolio 模組）
jupyter notebook notebooks/portfolio_backtest.ipynb

# 紙交易摘要
python scripts/paper_trading_summary.py

# 指定日期
python scripts/paper_trading_summary.py --as-of 2026-06-27
```

策略池買進額度（可在 Notebook 中 import 使用）：

```python
from paper_trading_allocation import show_buy_allocation

show_buy_allocation()
# show_buy_allocation(hypothetical_sells=["2330"])  # 模擬賣出後水位
```

---

## 紙交易檔案說明

| 檔案 | 用途 |
|------|------|
| `settings.csv` | 每期金額、0050/策略比例、最大持倉、停損參數 |
| `transactions.csv` | 買賣、入金紀錄（標的類型：`0050` / `策略`） |
| `open_positions.csv` | 目前策略池持倉（含持倉最高價，供移動停損計算） |

更新交易後執行 `paper_trading_summary.py` 即可檢視混合配置淨值與各持倉停損價。

---

## 常用指令

```bash
# 紙交易摘要
python scripts/paper_trading_summary.py

# Jupyter（建議在專案根目錄啟動）
jupyter notebook notebooks/

# CLI 需在 scripts/ 目錄下執行，或從根目錄指定路徑
cd scripts && python paper_trading_summary.py
```

---

## Roadmap：FastAPI 服務化

目前核心邏輯集中在 `scripts/`，後續計畫以 **FastAPI** 對外暴露，方便前端、排程或第三方整合，而不必每次手動跑 CLI / Notebook。

### 預期架構

```
api/
├── main.py              # FastAPI app 入口
├── routers/
│   ├── portfolio.py     # 淨值、持倉、報酬摘要
│   ├── allocation.py    # 策略池水位、下一筆買進額度
│   ├── backtest.py      # 回測結果、XIRR、權益曲線
│   └── market.py        # 0050 成分股、即時/歷史報價（FinMind）
├── schemas/             # Pydantic request/response models
└── deps.py              # 共用依賴（路徑、settings 載入）
```

`scripts/` 維持為 **domain / service 層**；API 層只做路由、驗證與序列化，避免重複實作。

### 規劃中的 API 端點

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/health` | 服務健康檢查 |
| `GET` | `/portfolio/summary` | 對應 `paper_trading_summary.summarize` |
| `GET` | `/portfolio/positions` | 目前持倉與停損價 |
| `GET` | `/allocation/next-buy` | 對應 `paper_trading_allocation.next_buy_allocation` |
| `POST` | `/backtest/run` | 觸發回測（參數：起訖日、DCA 設定） |
| `GET` | `/market/0050/universe` | 0050 成分股清單 |

### 本地開發（規劃）

```bash
pip install fastapi uvicorn

uvicorn api.main:app --reload --port 8000
# 文件：http://127.0.0.1:8000/docs
```

實作時會補上 `requirements.txt` 與 `.env.example`，並在 README 更新實際啟動方式。

---

## 注意事項

- `data/`、`paper_trading/*.csv` 預設不納入版控；clone 後需自行抓取資料或還原本機紀錄。
- 0050 於 2025-06-18 採 1 拆 4，回測模組已處理 FinMind 未還原價之調整。
- 本專案僅供研究與紙交易驗證，不構成投資建議。

---

## License

Private / personal use. 若需開源請自行補充授權條款。
