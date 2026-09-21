#!/usr/bin/env python3
"""Scan trailing-stop activation thresholds vs legacy (activate=0%)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_portfolio import (  # noqa: E402
    DCA_AMOUNT,
    DCA_0050_WEIGHT,
    DCA_START,
    DCA_STRATEGY_WEIGHT,
    MAX_POSITIONS,
    STRATEGY_CASH_CAP_MONTHS,
    evaluate_exit_signals,
    market_df_from_stock_table,
    net_return_from_prices,
    simulate_portfolio_dca,
)

TOP_K = 5
VOLUME_SURGE_MULT = 1.2
REFERENCE_TRADE_CASH = DCA_AMOUNT * DCA_STRATEGY_WEIGHT

THRESHOLDS = [0.0, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]


def build_signals(data: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    sig = data.copy().sort_values(["stock_id", "date"]).reset_index(drop=True)
    sig["prev_close"] = sig.groupby("stock_id")["close"].shift(1)
    sig["prev_MA_50"] = sig.groupby("stock_id")["MA_50"].shift(1)
    sig["volume_surge"] = sig["Trading_Volume"] >= sig["volume_ma_20"] * VOLUME_SURGE_MULT
    sig = sig.merge(scores, on=["date", "stock_id"], how="left")
    sig["score_prev"] = sig.groupby("stock_id")["score"].shift(1)
    sig["score_rank_prev"] = sig.groupby("date")["score_prev"].rank(ascending=False, method="first")
    sig["in_top_k"] = sig["score_rank_prev"] <= TOP_K
    sig["buy_signal"] = (
        sig["market_bull"]
        & (sig["MA_50"] > sig["MA_120"])
        & (sig["close"] > sig["MA_50"])
        & (sig["momentum_20"] > 0)
        & sig["in_top_k"]
    )
    sig["sell_signal"] = (
        (sig["MA_50"] < sig["MA_120"])
        & ((sig["close"] < sig["MA_50"]) & (sig["prev_close"] >= sig["prev_MA_50"]))
        & sig["volume_surge"]
    )
    return sig


def compute_momentum_score(data: pd.DataFrame) -> pd.DataFrame:
    pool = data[
        (data["MA_50"] > data["MA_120"]) & (data["momentum_20"] > 0)
    ].dropna(subset=["momentum_20", "momentum_60", "momentum_120"]).copy()
    pool["score"] = (
        pool.groupby("date")["momentum_20"].rank() * 0.2
        + pool.groupby("date")["momentum_60"].rank() * 0.3
        + pool.groupby("date")["momentum_120"].rank() * 0.5
    )
    return pool[["date", "stock_id", "score"]]


def collect_trades(sig: pd.DataFrame, trailing_activate_pct: float) -> pd.DataFrame:
    trade_list = []
    for stock in sig["stock_id"].unique():
        df_s = sig[sig["stock_id"] == stock]
        in_position = False
        entry_price = entry_date = entry_score = highest_price = 0

        for i in df_s.index:
            row = df_s.loc[i]
            if (not in_position) and row["buy_signal"]:
                in_position = True
                entry_price = row["close"]
                entry_date = row["date"]
                entry_score = row["score"]
                highest_price = row["close"]
            elif in_position:
                highest_price = max(highest_price, row["close"])
                should_exit, _, _, _, exit_reason = evaluate_exit_signals(
                    entry_price,
                    row["close"],
                    highest_price,
                    row["sell_signal"],
                    trailing_activate_pct=trailing_activate_pct,
                )
                if should_exit:
                    exit_price = row["close"]
                    trade_list.append(
                        {
                            "stock_id": stock,
                            "entry_date": entry_date,
                            "exit_date": row["date"],
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "score": entry_score,
                            "return": net_return_from_prices(
                                REFERENCE_TRADE_CASH, entry_price, exit_price
                            ),
                            "exit_reason": exit_reason,
                        }
                    )
                    in_position = False
    return pd.DataFrame(trade_list)


def run_portfolio(trades_df: pd.DataFrame, sig: pd.DataFrame, df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"xirr": 0.0, "total_return": 0.0, "max_dd": 0.0, "n_trades": 0, "n_trades_executed": 0}
    tdf = trades_df.copy()
    tdf["entry_date"] = pd.to_datetime(tdf["entry_date"])
    tdf["exit_date"] = pd.to_datetime(tdf["exit_date"])
    market_df = market_df_from_stock_table(df)
    _, metrics = simulate_portfolio_dca(
        tdf,
        sig,
        market_df,
        dca_amount=DCA_AMOUNT,
        w_0050=DCA_0050_WEIGHT,
        w_strategy=DCA_STRATEGY_WEIGHT,
        dca_start=DCA_START,
        max_positions=MAX_POSITIONS,
        strategy_cash_cap_months=STRATEGY_CASH_CAP_MONTHS,
    )
    return metrics


def main() -> None:
    data_path = ROOT / "data" / "all_stock_with_MA50_MA120_return20.csv"
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    sig = build_signals(df, compute_momentum_score(df))

    rows = []
    for th in THRESHOLDS:
        trades = collect_trades(sig, trailing_activate_pct=th)
        metrics = run_portfolio(trades, sig, df)
        win = (trades["return"] > 0).mean() if not trades.empty else float("nan")
        trail_pct = (
            (trades["exit_reason"] == "trailing_stop").mean() if not trades.empty else float("nan")
        )
        rows.append(
            {
                "啟動門檻": th,
                "XIRR": metrics["xirr"],
                "總報酬率": metrics["total_return"],
                "最大回撤": metrics["max_dd"],
                "信號筆數": len(trades),
                "實際成交": metrics.get("n_trades_executed", 0),
                "勝率": win,
                "移停占比": trail_pct,
            }
        )

    result = pd.DataFrame(rows)
    legacy = result.iloc[0]
    result["ΔXIRR"] = result["XIRR"] - legacy["XIRR"]
    result["Δ總報酬"] = result["總報酬率"] - legacy["總報酬率"]

    print("回測區間:", df["date"].min().date(), "~", df["date"].max().date())
    print("舊版 = 啟動門檻 0%（進場後即啟用移動停損 -10%）\n")

    display = result.copy()
    for col in ("啟動門檻", "XIRR", "總報酬率", "最大回撤", "勝率", "移停占比", "ΔXIRR", "Δ總報酬"):
        if col == "啟動門檻":
            display[col] = display[col].map(lambda x: f"{x:.0%}")
        elif col in ("信號筆數", "實際成交"):
            continue
        else:
            display[col] = display[col].map(lambda x: f"{x:.2%}")

    print(
        display[
            ["啟動門檻", "XIRR", "總報酬率", "最大回撤", "勝率", "移停占比", "ΔXIRR", "Δ總報酬"]
        ].to_string(index=False)
    )

    # closest to legacy by XIRR
    ranked = result.iloc[1:].copy()
    ranked["xirr_gap"] = (ranked["XIRR"] - legacy["XIRR"]).abs()
    ranked["ret_gap"] = (ranked["總報酬率"] - legacy["總報酬率"]).abs()
    best_xirr = ranked.loc[ranked["xirr_gap"].idxmin()]
    best_ret = ranked.loc[ranked["ret_gap"].idxmin()]

    print("\n" + "=" * 60)
    print("最接近舊版（不含 0% 本身）")
    print("=" * 60)
    print(
        f"依 XIRR：門檻 {best_xirr['啟動門檻']:.0%}  "
        f"XIRR {best_xirr['XIRR']:.2%}（Δ {best_xirr['ΔXIRR']:+.2%}）  "
        f"總報酬 {best_xirr['總報酬率']:.2%}"
    )
    print(
        f"依總報酬：門檻 {best_ret['啟動門檻']:.0%}  "
        f"總報酬 {best_ret['總報酬率']:.2%}（Δ {best_ret['Δ總報酬']:+.2%}）  "
        f"XIRR {best_ret['XIRR']:.2%}"
    )

    within_xirr = ranked[ranked["xirr_gap"] <= 0.001]
    within_ret = ranked[ranked["ret_gap"] <= 0.05]
    print("\nΔXIRR 在 ±0.10% 以內：", "無" if within_xirr.empty else ", ".join(f"{x:.0%}" for x in within_xirr["啟動門檻"]))
    print("Δ總報酬 在 ±5% 以內：", "無" if within_ret.empty else ", ".join(f"{x:.0%}" for x in within_ret["啟動門檻"]))


if __name__ == "__main__":
    main()
