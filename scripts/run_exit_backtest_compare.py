#!/usr/bin/env python3
"""Compare legacy vs plan-A (profit-activated trailing stop) exit rules."""

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
    TRAILING_ACTIVATE_PCT,
    evaluate_exit_signals,
    market_df_from_stock_table,
    net_return_from_prices,
    simulate_matched_dca_baseline,
    simulate_portfolio_dca,
    simulate_0050_dca,
)

TOP_K = 5
VOLUME_SURGE_MULT = 1.2
REFERENCE_TRADE_CASH = DCA_AMOUNT * DCA_STRATEGY_WEIGHT


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


def collect_trades(sig: pd.DataFrame, *, plan_a: bool) -> pd.DataFrame:
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
                if plan_a:
                    should_exit, _, _, _, exit_reason = evaluate_exit_signals(
                        entry_price, row["close"], highest_price, row["sell_signal"]
                    )
                else:
                    current_return = row["close"] / entry_price - 1
                    stop_loss_hit = current_return <= -0.08
                    trailing_stop_hit = row["close"] < highest_price * 0.9
                    trend_exit = row["sell_signal"]
                    should_exit = stop_loss_hit or trailing_stop_hit or trend_exit
                    if stop_loss_hit:
                        exit_reason = "stop_loss"
                    elif trailing_stop_hit:
                        exit_reason = "trailing_stop"
                    else:
                        exit_reason = "trend_exit"

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
                            "gross_return": exit_price / entry_price - 1,
                            "return": net_return_from_prices(
                                REFERENCE_TRADE_CASH, entry_price, exit_price
                            ),
                            "exit_reason": exit_reason,
                        }
                    )
                    in_position = False
    return pd.DataFrame(trade_list)


def run_portfolio(trades_df: pd.DataFrame, sig: pd.DataFrame, df: pd.DataFrame) -> dict:
    trades_df = trades_df.copy()
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    market_df = market_df_from_stock_table(df)
    _, metrics = simulate_portfolio_dca(
        trades_df,
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


def print_block(label: str, trades_df: pd.DataFrame, metrics: dict) -> None:
    print(f"\n{'=' * 54}")
    print(label)
    print("=" * 54)
    print(f"XIRR:           {metrics['xirr']:.2%}")
    print(f"總報酬率:       {metrics['total_return']:.2%}")
    print(f"最大回撤:       {metrics['max_dd']:.2%}")
    print(f"策略信號筆數:   {metrics['n_trades']}")
    print(f"實際成交筆數:   {metrics.get('n_trades_executed', 0)}")
    exec_df = trades_df  # signal-level stats below
    if not exec_df.empty:
        win = (exec_df["return"] > 0).mean()
        print(f"訊號層級勝率:   {win:.1%}")
        print(f"平均持有天數:   {(pd.to_datetime(exec_df['exit_date']) - pd.to_datetime(exec_df['entry_date'])).dt.days.mean():.0f}")
        print("出場原因:")
        for reason, pct in exec_df["exit_reason"].value_counts(normalize=True).items():
            avg = exec_df.loc[exec_df["exit_reason"] == reason, "return"].mean()
            print(f"  {reason:14s} {pct:6.1%}  平均報酬 {avg:+.2%}")


def main() -> None:
    data_path = ROOT / "data" / "all_stock_with_MA50_MA120_return20.csv"
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    scores = compute_momentum_score(df)
    sig = build_signals(df, scores)

    trades_legacy = collect_trades(sig, plan_a=False)
    trades_plan_a = collect_trades(sig, plan_a=True)

    metrics_legacy = run_portfolio(trades_legacy, sig, df)
    metrics_plan_a = run_portfolio(trades_plan_a, sig, df)

    print("回測區間:", df["date"].min().date(), "~", df["date"].max().date())
    print(f"方案 A：持倉最高價浮盈 ≥ {TRAILING_ACTIVATE_PCT:.0%} 才啟用移動停損 -10%")
    print_block("【舊版】進場起即啟用移動停損", trades_legacy, metrics_legacy)
    print_block(f"【方案 A】浮盈 {TRAILING_ACTIVATE_PCT:.0%} 才啟用移動停損", trades_plan_a, metrics_plan_a)

    print(f"\n{'=' * 54}")
    print("方案 A vs 舊版 差異")
    print("=" * 54)
    print(f"XIRR       {metrics_plan_a['xirr'] - metrics_legacy['xirr']:+.2%}")
    print(f"總報酬率   {metrics_plan_a['total_return'] - metrics_legacy['total_return']:+.2%}")
    print(f"最大回撤   {metrics_plan_a['max_dd'] - metrics_legacy['max_dd']:+.2%}")
    print(f"信號筆數   {len(trades_plan_a) - len(trades_legacy):+d}")


if __name__ == "__main__":
    main()
