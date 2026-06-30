#!/usr/bin/env python3
"""紙交易摘要：讀取 paper_trading/ 下的 CSV，計算混合配置淨值（預設 70/30）。

用法：
  python scripts/paper_trading_summary.py
  python scripts/paper_trading_summary.py --as-of 2026-06-27
"""

from __future__ import annotations

import argparse

import pandas as pd

from project_paths import DATA_DIR, PAPER_DIR, ROOT

STOCK_CSV = DATA_DIR / "TSMC_stock_data.csv"
MARKET_ID = "0050"


def load_settings() -> dict:
    df = pd.read_csv(PAPER_DIR / "settings.csv")
    return dict(zip(df["項目"], df["數值"]))


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(PAPER_DIR / "transactions.csv")
    df["交易日期"] = pd.to_datetime(df["交易日期"])
    if "信號日期" in df.columns:
        df["信號日期"] = pd.to_datetime(df["信號日期"], errors="coerce")
    for col in ("成交價", "股數", "成交金額", "手續費"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.sort_values("交易日期")


def load_open_positions() -> pd.DataFrame:
    path = PAPER_DIR / "open_positions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ("買入價", "股數", "成本含手續費", "持倉最高價"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _norm_code(sid) -> str:
    s = str(sid).strip()
    return s.zfill(4) if s.isdigit() else s


def fetch_0050_close(as_of: pd.Timestamp) -> float | None:
    try:
        from dotenv import load_dotenv
        from FinMind.data import DataLoader
        import os

        load_dotenv(ROOT / ".env")
        token = os.environ.get("FINMIND_TOKEN")
        if not token:
            return None
        api = DataLoader()
        api.login_by_token(api_token=token)
        px = api.taiwan_stock_daily(stock_id=MARKET_ID, start_date="2016-01-01")
        px["date"] = pd.to_datetime(px["date"])
        px = px[px["date"] <= as_of].sort_values("date")
        if px.empty:
            return None
        return float(px.iloc[-1]["close"])
    except Exception:
        return None


def latest_closes(as_of: pd.Timestamp) -> dict[str, float]:
    closes: dict[str, float] = {}
    if STOCK_CSV.exists():
        px = pd.read_csv(STOCK_CSV, usecols=["stock_id", "date", "close"])
        px["date"] = pd.to_datetime(px["date"])
        px = px[px["date"] <= as_of]
        for sid, g in px.groupby("stock_id"):
            code = _norm_code(sid)
            closes[code] = float(g.sort_values("date").iloc[-1]["close"])
    if MARKET_ID not in closes:
        etf = fetch_0050_close(as_of)
        if etf is not None:
            closes[MARKET_ID] = etf
    return closes


def fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x:.2%}"


def _is_dca(settings: dict) -> bool:
    mode = str(settings.get("投入方式", "")).strip()
    return "定期" in mode


def summarize(as_of: str | None = None) -> None:
    settings = load_settings()
    tx = load_transactions()
    positions = load_open_positions()
    dca = _is_dca(settings)

    total_capital = float(settings.get("總資金", 0) or 0)
    weight_0050 = float(settings["0050比例"])
    weight_strategy = float(settings["策略比例"])
    target_0050 = float(settings.get("0050目標金額", 0) or 0)
    target_strategy = float(settings.get("策略目標金額", 0) or 0)
    dca_per_period = float(settings.get("每期金額", 0) or 0)
    max_positions = int(float(settings["最大持倉檔數"]))
    blend_label = f"{weight_0050:.0%} 0050 + {weight_strategy:.0%} 策略"

    as_of_ts = pd.Timestamp(as_of) if as_of else tx["交易日期"].max()
    if pd.isna(as_of_ts):
        as_of_ts = pd.Timestamp.today().normalize()

    tx = tx[tx["交易日期"] <= as_of_ts]
    closes = latest_closes(as_of_ts)

    # 0050 桶
    etf = tx[tx["標的類型"] == "0050"]
    etf_shares = 0.0
    etf_cost = 0.0
    for _, row in etf.iterrows():
        sign = 1 if row["動作"] == "買" else -1
        etf_shares += sign * row["股數"]
        etf_cost += sign * (row["成交金額"] + row["手續費"])

    etf_price = closes.get(MARKET_ID, float("nan"))
    etf_value = etf_shares * etf_price if pd.notna(etf_price) else float("nan")
    if dca:
        etf_return = (etf_value / etf_cost - 1) if etf_cost and pd.notna(etf_value) else float("nan")
    else:
        etf_return = (etf_value / target_0050 - 1) if target_0050 and pd.notna(etf_value) else float("nan")

    # 策略桶：現金 + 持倉
    strategy_cash = 0.0 if dca else target_strategy
    strategy_deposits = 0.0
    for _, row in tx[tx["標的類型"] == "策略"].iterrows():
        action = str(row["動作"]).strip()
        if action == "入金":
            strategy_cash += row["成交金額"]
            strategy_deposits += row["成交金額"]
        elif action == "買":
            strategy_cash -= row["成交金額"] + row["手續費"]
        elif action == "賣":
            strategy_cash += row["成交金額"] - row["手續費"]

    pos_rows = []
    pos_value = 0.0
    if not positions.empty:
        for _, p in positions.iterrows():
            code = str(p["股票代碼"]).zfill(4) if str(p["股票代碼"]).isdigit() else str(p["股票代碼"])
            price = closes.get(code, float("nan"))
            shares = float(p["股數"])
            mv = shares * price if pd.notna(price) else float("nan")
            cost = float(p.get("成本含手續費", shares * float(p["買入價"])))
            high = float(p.get("持倉最高價", p["買入價"]))
            if pd.notna(price):
                high = max(high, price)
            pnl = (mv / cost - 1) if cost and pd.notna(mv) else float("nan")
            stop_price = float(p["買入價"]) * (1 + float(settings["固定停損"]))
            trail_price = high * (1 + float(settings["移動停損回撤"]))
            pos_rows.append(
                {
                    "代碼": code,
                    "名稱": p.get("股票名稱", ""),
                    "股數": int(shares),
                    "成本": cost,
                    "現價": price,
                    "市值": mv,
                    "報酬率": pnl,
                    "最高價": high,
                    "固定停損價": stop_price,
                    "移動停損價": trail_price,
                }
            )
            if pd.notna(mv):
                pos_value += mv

    strategy_equity = strategy_cash + pos_value
    if dca:
        strategy_return = (
            (strategy_equity / strategy_deposits - 1) if strategy_deposits else float("nan")
        )
        cumulative_invested = etf_cost + strategy_deposits
    else:
        strategy_return = (strategy_equity / target_strategy - 1) if target_strategy else float("nan")
        cumulative_invested = total_capital

    total_equity = (etf_value if pd.notna(etf_value) else 0) + strategy_equity
    invested_base = cumulative_invested if dca else total_capital
    blend_return = (total_equity / invested_base - 1) if invested_base else float("nan")

    print("=" * 60)
    print(f"紙交易摘要（截至 {as_of_ts.date()}）")
    print("=" * 60)
    if dca:
        n_periods = len(tx[(tx["動作"] == "入金") & (tx["標的類型"] == "策略")])
        print(
            f"投入方式：定期定額  |  每期 {fmt_money(dca_per_period)}  |  "
            f"配置：{blend_label}  |  已投入 {n_periods} 期"
        )
        print(f"累計投入：{fmt_money(cumulative_invested)}")
    else:
        print(f"總資金：{fmt_money(total_capital)}  |  配置：{blend_label}")
    print()

    print("【0050 桶】")
    if dca:
        print(f"  持股：{etf_shares:,.0f} 股  |  累計成本：{fmt_money(etf_cost)}")
    else:
        print(f"  持股：{etf_shares:,.0f} 股  |  成本：{fmt_money(etf_cost)}  |  目標：{fmt_money(target_0050)}")
    if pd.notna(etf_price):
        print(f"  現價：{etf_price:.2f}  |  市值：{fmt_money(etf_value)}  |  報酬：{fmt_pct(etf_return)}")
    else:
        print("  ⚠ 找不到 0050 收盤價，請確認 .env 的 FINMIND_TOKEN 或網路連線")
    print()

    print("【策略桶】")
    print(f"  現金：{fmt_money(strategy_cash)}  |  持倉市值：{fmt_money(pos_value)}  |  淨值：{fmt_money(strategy_equity)}")
    print(f"  報酬：{fmt_pct(strategy_return)}  |  持倉：{len(pos_rows)}/{max_positions} 檔")
    if dca:
        per_slot = strategy_equity / max_positions if strategy_equity else 0
        print(f"  累計入金策略池：{fmt_money(strategy_deposits)}  |  單檔上限（參考）：{fmt_money(per_slot)}")
    else:
        per_slot = target_strategy / max_positions
        print(f"  單檔上限（參考）：{fmt_money(per_slot)}")
    print()

    if pos_rows:
        print("【目前持倉】")
        pos_df = pd.DataFrame(pos_rows)
        display = pos_df.copy()
        for c in ("成本", "現價", "市值", "最高價", "固定停損價", "移動停損價"):
            display[c] = display[c].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
        display["報酬率"] = pos_df["報酬率"].map(lambda x: fmt_pct(x) if pd.notna(x) else "-")
        print(display.to_string(index=False))
        print()

    print(f"【{blend_label} 混合】")
    print(f"  總淨值：{fmt_money(total_equity)}  |  累計報酬：{fmt_pct(blend_return)}")
    print()
    print("提示：每筆交易寫入 transactions.csv；持倉維護在 open_positions.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="紙交易混合配置摘要")
    parser.add_argument("--as-of", help="截止日期 YYYY-MM-DD")
    args = parser.parse_args()
    summarize(args.as_of)


if __name__ == "__main__":
    main()
