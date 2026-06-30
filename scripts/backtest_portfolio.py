"""Portfolio backtest with monthly DCA and XIRR (aligned with paper_trading/settings.csv)."""

from __future__ import annotations

import numpy as np
import pandas as pd

COMMISSION_RATE = 0.001425
TAX_RATE = 0.003
MIN_COMMISSION = 20.0
MAX_POSITIONS = 5

DCA_AMOUNT = 5_500
DCA_0050_WEIGHT = 0.5
DCA_STRATEGY_WEIGHT = 0.5
DCA_START = pd.Timestamp("2010-01-11")
DCA_DAY = 11

# 方案 B：策略池現金超過 N 期 30% 投入量 → 溢出買 0050（0 = 關閉，資金留在策略池）
STRATEGY_CASH_CAP_MONTHS = 0

# 策略池現金占比超過此值時，依剩餘空 slot 加大單筆進場（提高資金使用率）
STRATEGY_DEPLOY_CASH_PCT = 0.40

# 0050 於 2025-06-18 恢復交易，採 1 拆 4（FinMind 為未還原收盤價）
SPLIT_0050_EFFECTIVE = pd.Timestamp("2025-06-18")
SPLIT_0050_RATIO = 4.0


def calc_commission(amount: float) -> float:
    """單邊手續費（對齊 settings.csv：rate 與最低 20 元取大）。"""
    if amount <= 0:
        return 0.0
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def buy_shares_after_commission(cash_out: float, price: float) -> tuple[float, float, float]:
    """Returns (shares, buy_commission, net_invest)."""
    if cash_out <= 0 or price <= 0:
        return 0.0, 0.0, 0.0
    comm = calc_commission(cash_out)
    net = cash_out - comm
    if net <= 0:
        return 0.0, comm, 0.0
    return net / price, comm, net


def sell_proceeds_after_costs(shares: float, price: float) -> tuple[float, float, float, float]:
    """Returns (net_proceeds, gross, sell_commission, tax)."""
    if shares <= 0 or price <= 0:
        return 0.0, 0.0, 0.0, 0.0
    gross = shares * price
    sell_comm = calc_commission(gross)
    tax = gross * TAX_RATE
    return gross - sell_comm - tax, gross, sell_comm, tax


def net_return_from_prices(cash_out: float, entry_price: float, exit_price: float) -> float:
    """含買賣手續費與證交稅的 round-trip 報酬率（cash_out 為自策略池扣出的總現金）。"""
    shares, _, _ = buy_shares_after_commission(cash_out, entry_price)
    if shares <= 0:
        return np.nan
    net_proceeds, _, _, _ = sell_proceeds_after_costs(shares, exit_price)
    return net_proceeds / cash_out - 1


def adjust_0050_price(
    price: float,
    date: pd.Timestamp,
    effective: pd.Timestamp = SPLIT_0050_EFFECTIVE,
    ratio: float = SPLIT_0050_RATIO,
) -> float:
    """還原價 → 分割後基準：分割生效日前之收盤價除以 ratio。"""
    if price is None or (isinstance(price, float) and np.isnan(price)):
        return np.nan
    if pd.Timestamp(date) < effective:
        return float(price) / ratio
    return float(price)


def prepare_market_df(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    輸入 FinMind 原始 0050 收盤價，產出：
    - market_close_raw：原始報價（供顯示）
    - market_close：分割還原後連續價格（供 DCA / MA / 回測）
    """
    mkt = market_df.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])

    if "market_close_raw" in mkt.columns:
        raw = mkt["market_close_raw"]
    elif "close" in mkt.columns:
        raw = mkt["close"]
    elif "market_close" in mkt.columns:
        raw = mkt["market_close"]
    else:
        raise ValueError("market_df 需含 close 或 market_close 欄位")

    mkt["market_close_raw"] = raw
    mkt = mkt.sort_values("date")
    # 分割停牌期（2025/6/11～17）FinMind 可能無收盤價，向前填補以利 MA 連續
    mkt["market_close_raw"] = mkt["market_close_raw"].ffill()
    mkt["market_close"] = [
        adjust_0050_price(p, d) for p, d in zip(mkt["market_close_raw"], mkt["date"])
    ]
    return mkt


def market_df_from_stock_table(df: pd.DataFrame) -> pd.DataFrame:
    """從 df 取出 0050 原始收盤價（優先 market_close_raw）供 prepare_market_df 使用。"""
    if "market_close_raw" in df.columns:
        out = df[["date", "market_close_raw"]].drop_duplicates().rename(
            columns={"market_close_raw": "close"}
        )
    else:
        out = df[["date", "market_close"]].drop_duplicates().rename(
            columns={"market_close": "close"}
        )
    return out


def xirr(cashflows: list[float], dates: list[pd.Timestamp], day_count: float = 365.25) -> float:
    """IRR for irregular dated cash flows (investor outflows negative)."""
    if not cashflows or len(cashflows) < 2:
        return np.nan
    d0 = pd.Timestamp(dates[0])
    days = np.array([(pd.Timestamp(d) - d0).days for d in dates], dtype=float)

    def npv(rate: float) -> float:
        return sum(cf / (1 + rate) ** (day / day_count) for cf, day in zip(cashflows, days))

    lo, hi = -0.999, 10.0
    if npv(lo) * npv(hi) > 0:
        return np.nan
    for _ in range(128):
        mid = (lo + hi) / 2
        if abs(npv(mid)) < 1e-9:
            return float(mid)
        if npv(mid) * npv(lo) <= 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)


def make_dca_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    day: int = DCA_DAY,
    trading_dates: pd.DatetimeIndex | None = None,
) -> pd.DatetimeIndex:
    """Monthly contribution dates on `day`, snapped to next available trading day."""
    cur = pd.Timestamp(start.year, start.month, min(day, 28))
    if cur < start:
        cur += pd.DateOffset(months=1)
        cur = pd.Timestamp(cur.year, cur.month, min(day, 28))

    raw: list[pd.Timestamp] = []
    while cur <= end:
        raw.append(cur)
        cur += pd.DateOffset(months=1)
        cur = pd.Timestamp(cur.year, cur.month, min(day, 28))

    if not raw:
        return pd.DatetimeIndex([])

    if trading_dates is None:
        return pd.DatetimeIndex(raw)

    td = pd.DatetimeIndex(trading_dates).sort_values()
    snapped = []
    for d in raw:
        idx = td.searchsorted(d, side="left")
        if idx < len(td):
            snapped.append(td[idx])
    return pd.DatetimeIndex(snapped).unique()


def _position_value(open_positions: list[dict], date: pd.Timestamp, price_lookup) -> float:
    total = 0.0
    for pos in open_positions:
        try:
            price = price_lookup.loc[(date, pos["stock_id"])]
            if pos.get("shares"):
                total += pos["shares"] * price
            else:
                entry_p = price_lookup.loc[(pos["entry_date"], pos["stock_id"])]
                total += pos["allocated"] * (price / entry_p)
        except KeyError:
            total += pos["allocated"]
    return total


def _strategy_entry_allocate(
    cash_strategy: float,
    strat_equity: float,
    open_count: int,
    max_positions: int,
    deploy_cash_pct: float = STRATEGY_DEPLOY_CASH_PCT,
) -> float:
    """依策略池閒置程度決定單筆進場金額。"""
    slots_available = max_positions - open_count
    if slots_available <= 0 or cash_strategy <= 0 or strat_equity <= 0:
        return 0.0
    target_per_slot = strat_equity / max_positions
    if cash_strategy / strat_equity > deploy_cash_pct:
        return min(cash_strategy, cash_strategy / slots_available)
    return min(target_per_slot, cash_strategy)


def _sweep_strategy_overflow_to_0050(
    cash_strategy: float,
    cash_cap: float,
    date: pd.Timestamp,
    price_0050: pd.Series,
    shares_0050: float,
) -> tuple[float, float, float]:
    """策略池現金 > 上限時，溢出部分改買 0050。Returns (cash_strategy, shares_0050, swept)."""
    if cash_cap <= 0 or cash_strategy <= cash_cap:
        return cash_strategy, shares_0050, 0.0
    if date not in price_0050.index:
        return cash_strategy, shares_0050, 0.0
    swept = cash_strategy - cash_cap
    shares_bought, _, _ = buy_shares_after_commission(swept, price_0050.loc[date])
    shares_0050 += shares_bought
    return cash_cap, shares_0050, swept


def simulate_0050_dca(
    market_df: pd.DataFrame,
    dca_amount: float = DCA_AMOUNT,
    w_0050: float = 1.0,
    dca_start: pd.Timestamp = DCA_START,
    dca_day: int = DCA_DAY,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """DCA into 0050 only; returns (equity_df, metrics)."""
    mkt = prepare_market_df(market_df)
    mkt = mkt.dropna(subset=["market_close"]).sort_values("date")
    if mkt.empty:
        empty = {"xirr": np.nan, "total_invested": 0.0, "total_return": 0.0, "max_dd": 0.0, "years": 0.0}
        return pd.DataFrame(), empty

    sim_start = period_start if period_start is not None else dca_start
    sim_end = period_end if period_end is not None else mkt["date"].max()
    mkt = mkt[(mkt["date"] >= sim_start) & (mkt["date"] <= sim_end)]
    if mkt.empty:
        empty = {"xirr": np.nan, "total_invested": 0.0, "total_return": 0.0, "max_dd": 0.0, "years": 0.0}
        return pd.DataFrame(), empty

    trading_dates = pd.DatetimeIndex(mkt["date"])
    dca_dates = make_dca_dates(max(dca_start, sim_start), sim_end, dca_day, trading_dates)
    dca_set = set(dca_dates)
    price_by_date = mkt.set_index("date")["market_close"]

    shares = 0.0
    cash_0050 = 0.0
    cumulative_invested = 0.0
    records = []
    dca_flows: list[tuple[pd.Timestamp, float]] = []

    contrib = dca_amount * w_0050

    for date in trading_dates:
        if date in dca_set and contrib > 0:
            cumulative_invested += contrib
            dca_flows.append((date, contrib))
            buy_comm = calc_commission(contrib)
            buy_cash = contrib - buy_comm
            px = price_by_date.loc[date]
            shares += buy_cash / px

        equity = shares * price_by_date.loc[date] + cash_0050
        records.append(
            {
                "date": date,
                "equity": equity,
                "equity_0050": equity,
                "equity_strategy": 0.0,
                "cumulative_invested": cumulative_invested,
                "shares_0050": shares,
            }
        )

    equity_df = pd.DataFrame(records)
    metrics = _finalize_metrics(equity_df, dca_flows, cumulative_invested)
    return equity_df, metrics


def simulate_matched_dca_baseline(
    market_df: pd.DataFrame,
    w_0050: float = DCA_0050_WEIGHT,
    w_strategy: float = DCA_STRATEGY_WEIGHT,
    dca_amount: float = DCA_AMOUNT,
    dca_start: pd.Timestamp = DCA_START,
    dca_day: int = DCA_DAY,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> dict:
    """
    同配置基準：w_0050 定額買 0050，w_strategy 全部留現金（不選股）。
    用於評估「30% 策略池」是否贏過單純囤現金。
    """
    if w_0050 + w_strategy <= 0:
        return {"xirr": np.nan, "final_equity": 0.0, "total_invested": 0.0}

    mkt = prepare_market_df(market_df)
    mkt = mkt.dropna(subset=["market_close"]).sort_values("date")
    sim_start = period_start if period_start is not None else dca_start
    sim_end = period_end if period_end is not None else mkt["date"].max()
    mkt = mkt[(mkt["date"] >= sim_start) & (mkt["date"] <= sim_end)]

    dca_dates = make_dca_dates(max(dca_start, sim_start), sim_end, dca_day, mkt["date"])
    if len(dca_dates) == 0:
        return {"xirr": np.nan, "final_equity": 0.0, "total_invested": 0.0}

    _, m0050 = simulate_0050_dca(
        market_df,
        dca_amount=dca_amount * w_0050,
        w_0050=1.0,
        dca_start=dca_start,
        dca_day=dca_day,
        period_start=period_start,
        period_end=period_end,
    )
    strat_cash = len(dca_dates) * dca_amount * w_strategy
    total_invested = len(dca_dates) * dca_amount
    final_equity = m0050["final_equity"] + strat_cash

    dca_flows = [(d, dca_amount) for d in dca_dates]
    irr = xirr(
        [-amt for _, amt in dca_flows] + [final_equity],
        [d for d, _ in dca_flows] + [mkt["date"].iloc[-1]],
    )
    return {
        "xirr": irr,
        "final_equity": final_equity,
        "total_invested": total_invested,
        "final_equity_0050": m0050["final_equity"],
        "final_strategy_cash": strat_cash,
    }


def simulate_portfolio_dca(
    trade_list: list[dict] | pd.DataFrame,
    sig: pd.DataFrame,
    market_df: pd.DataFrame,
    dca_amount: float = DCA_AMOUNT,
    w_0050: float = DCA_0050_WEIGHT,
    w_strategy: float = DCA_STRATEGY_WEIGHT,
    dca_start: pd.Timestamp = DCA_START,
    dca_day: int = DCA_DAY,
    max_positions: int = MAX_POSITIONS,
    strategy_cash_cap_months: float | None = STRATEGY_CASH_CAP_MONTHS,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Monthly DCA: w_0050 → 0050 bucket, w_strategy → strategy pool (stock picks).
    方案 B：策略池現金 > cap（每期策略投入 × strategy_cash_cap_months）時，溢出買 0050。
    strategy_cash_cap_months=None 或 0 則關閉溢出機制。
    Returns (equity_df, metrics dict including xirr).
    """
    tdf = pd.DataFrame(trade_list) if not isinstance(trade_list, pd.DataFrame) else trade_list.copy()
    empty_metrics = {
        "n_trades": 0,
        "win_rate": np.nan,
        "total_invested": 0.0,
        "total_return": 0.0,
        "xirr": np.nan,
        "max_dd": 0.0,
        "years": 0.0,
        "avg_cash_pct": 100.0,
        "final_equity": 0.0,
    }
    if tdf.empty:
        return pd.DataFrame(), empty_metrics

    tdf["entry_date"] = pd.to_datetime(tdf["entry_date"])
    tdf["exit_date"] = pd.to_datetime(tdf["exit_date"])

    mkt = prepare_market_df(market_df)
    mkt = mkt.dropna(subset=["market_close"]).sort_values("date")
    price_0050 = mkt.set_index("date")["market_close"]

    price_df = sig[["date", "stock_id", "close"]].copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_lookup = price_df.set_index(["date", "stock_id"])["close"]

    sim_start = period_start if period_start is not None else min(dca_start, tdf["entry_date"].min())
    sim_end = period_end if period_end is not None else max(tdf["exit_date"].max(), mkt["date"].max())

    mkt_in_range = mkt[(mkt["date"] >= sim_start) & (mkt["date"] <= sim_end)]
    trading_dates = pd.DatetimeIndex(mkt_in_range["date"]).sort_values()
    trade_dates = pd.DatetimeIndex(
        pd.unique(pd.concat([tdf["entry_date"], tdf["exit_date"]], ignore_index=True))
    )
    trade_dates = trade_dates[(trade_dates >= sim_start) & (trade_dates <= sim_end)]
    trading_dates = trading_dates.union(trade_dates).sort_values()
    if len(trading_dates) == 0:
        trading_dates = pd.date_range(sim_start, sim_end, freq="B")

    dca_dates = make_dca_dates(max(dca_start, sim_start), sim_end, dca_day, mkt["date"])
    dca_set = set(dca_dates)

    shares_0050 = 0.0
    cash_0050 = 0.0
    cash_strategy = 0.0
    open_positions: list[dict] = []
    cumulative_invested = 0.0
    cumulative_0050_deposits = 0.0
    cumulative_strategy_deposits = 0.0
    strategy_realized_profit = 0.0  # 平倉獲利再投入策略池（複利來源）
    equity_records = []
    dca_flows: list[tuple[pd.Timestamp, float]] = []

    contrib_0050 = dca_amount * w_0050
    contrib_strategy = dca_amount * w_strategy
    cash_cap = (
        contrib_strategy * strategy_cash_cap_months
        if strategy_cash_cap_months and strategy_cash_cap_months > 0
        else None
    )
    total_swept_to_0050 = 0.0
    closed_trade_returns: list[float] = []

    for date in trading_dates:
        # DCA injection
        if date in dca_set:
            cumulative_invested += dca_amount
            dca_flows.append((date, dca_amount))
            if contrib_0050 > 0:
                cumulative_0050_deposits += contrib_0050
            if contrib_strategy > 0:
                cumulative_strategy_deposits += contrib_strategy
            if contrib_0050 > 0 and date in price_0050.index:
                shares_bought, _, _ = buy_shares_after_commission(
                    contrib_0050, price_0050.loc[date]
                )
                shares_0050 += shares_bought
            elif contrib_0050 > 0:
                cash_0050 += contrib_0050
            cash_strategy += contrib_strategy

        # Strategy exits（含賣出手續費 + 證交稅）
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= date:
                exit_price = pos["exit_price"]
                if exit_price is None or (isinstance(exit_price, float) and np.isnan(exit_price)):
                    try:
                        exit_price = float(price_lookup.loc[(pos["exit_date"], pos["stock_id"])])
                    except KeyError:
                        exit_price = float(price_lookup.loc[(date, pos["stock_id"])])
                net_proceeds, _, _, _ = sell_proceeds_after_costs(pos["shares"], exit_price)
                strategy_realized_profit += net_proceeds - pos["allocated"]
                closed_trade_returns.append(net_proceeds / pos["allocated"] - 1)
                cash_strategy += net_proceeds
            else:
                still_open.append(pos)
        open_positions = still_open

        # Strategy entries (score priority)
        new_trades = (
            tdf[tdf["entry_date"] == date]
            .sort_values("score", ascending=False, na_position="last")
        )
        for _, t in new_trades.iterrows():
            if len(open_positions) >= max_positions:
                break
            strat_positions = _position_value(open_positions, date, price_lookup)
            strat_equity = cash_strategy + strat_positions
            allocate = _strategy_entry_allocate(
                cash_strategy, strat_equity, len(open_positions), max_positions
            )
            if allocate <= 0:
                continue
            try:
                entry_price = float(t["entry_price"])
            except (KeyError, TypeError, ValueError):
                entry_price = float(price_lookup.loc[(date, t["stock_id"])])
            if entry_price <= 0:
                continue
            shares, buy_comm, _ = buy_shares_after_commission(allocate, entry_price)
            if shares <= 0:
                continue
            cash_strategy -= allocate
            exit_price = t["exit_price"] if "exit_price" in t.index else np.nan
            open_positions.append(
                {
                    "stock_id": t["stock_id"],
                    "entry_date": t["entry_date"],
                    "exit_date": t["exit_date"],
                    "exit_price": exit_price,
                    "allocated": allocate,
                    "buy_commission": buy_comm,
                    "shares": shares,
                    "entry_price": entry_price,
                }
            )

        # 方案 B：策略池閒置現金溢出 → 0050（在進場之後，保留 cap 供後續訊號）
        if cash_cap is not None:
            cash_strategy, shares_0050, swept = _sweep_strategy_overflow_to_0050(
                cash_strategy, cash_cap, date, price_0050, shares_0050
            )
            total_swept_to_0050 += swept

        # Mark to market
        equity_0050 = cash_0050
        if date in price_0050.index:
            equity_0050 += shares_0050 * price_0050.loc[date]
        strat_positions = _position_value(open_positions, date, price_lookup)
        equity_strategy = cash_strategy + strat_positions
        total_equity = equity_0050 + equity_strategy

        equity_records.append(
            {
                "date": date,
                "equity": total_equity,
                "equity_0050": equity_0050,
                "equity_strategy": equity_strategy,
                "positions_value": strat_positions,
                "cumulative_invested": cumulative_invested,
                "cumulative_0050_deposits": cumulative_0050_deposits,
                "cumulative_strategy_deposits": cumulative_strategy_deposits,
                "strategy_realized_profit": strategy_realized_profit,
                "cash_strategy": cash_strategy,
                "n_positions": len(open_positions),
                "swept_to_0050_cumulative": total_swept_to_0050,
            }
        )

    equity_df = pd.DataFrame(equity_records)
    n_trades = len(tdf)
    n_trades_executed = len(closed_trade_returns)
    win_rate = (
        float(np.mean(np.array(closed_trade_returns) > 0))
        if closed_trade_returns
        else ((tdf["return"] > 0).mean() if "return" in tdf.columns and n_trades else np.nan)
    )
    metrics = _finalize_metrics(equity_df, dca_flows, cumulative_invested)
    metrics["n_trades"] = n_trades
    metrics["n_trades_executed"] = n_trades_executed
    metrics["win_rate"] = win_rate
    if not equity_df.empty:
        strat_eq = equity_df["equity_strategy"].replace(0, np.nan)
        pos_val = equity_df["positions_value"]
        metrics["avg_cash_pct"] = (equity_df["cash_strategy"] / strat_eq).mean(skipna=True) * 100
        metrics["avg_invested_pct"] = (pos_val / strat_eq).mean(skipna=True) * 100
        metrics["avg_n_positions"] = equity_df["n_positions"].mean()
        last = equity_df.iloc[-1]
        last_date = last["date"]
        final_0050 = float(last["equity_0050"])
        final_strategy = float(last["equity_strategy"])
        dep_0050 = float(last["cumulative_0050_deposits"])
        dep_strat = float(last["cumulative_strategy_deposits"])
        metrics["final_equity_0050"] = final_0050
        metrics["final_equity_strategy"] = final_strategy
        metrics["cumulative_0050_deposits"] = dep_0050
        metrics["cumulative_strategy_deposits"] = dep_strat
        metrics["compound_profit_0050"] = final_0050 - dep_0050
        metrics["compound_profit_strategy"] = final_strategy - dep_strat
        metrics["compound_profit_total"] = metrics["final_equity"] - cumulative_invested
        metrics["strategy_realized_profit"] = float(last["strategy_realized_profit"])
        unrealized = 0.0
        for pos in open_positions:
            try:
                price = price_lookup.loc[(last_date, pos["stock_id"])]
                if pos.get("shares"):
                    net_if_sold, _, _, _ = sell_proceeds_after_costs(pos["shares"], float(price))
                    unrealized += net_if_sold - pos["allocated"]
                else:
                    entry_p = price_lookup.loc[(pos["entry_date"], pos["stock_id"])]
                    unrealized += pos["allocated"] * (price / entry_p - 1)
            except KeyError:
                pass
        metrics["strategy_unrealized_pnl"] = unrealized
        metrics["strategy_cash_cap"] = cash_cap if cash_cap is not None else 0.0
        metrics["total_swept_to_0050"] = total_swept_to_0050
    return equity_df, metrics


def _finalize_metrics(
    equity_df: pd.DataFrame,
    dca_flows: list[tuple[pd.Timestamp, float]],
    cumulative_invested: float,
) -> dict:
    if equity_df.empty or cumulative_invested <= 0:
        return {
            "total_invested": cumulative_invested,
            "total_return": 0.0,
            "xirr": np.nan,
            "max_dd": 0.0,
            "years": 0.0,
            "final_equity": 0.0,
        }

    final_equity = float(equity_df["equity"].iloc[-1])
    years = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days / 365.25
    total_return = final_equity / cumulative_invested - 1

    flows = [-amt for _, amt in dca_flows]
    dates = [d for d, _ in dca_flows]
    flows.append(final_equity)
    dates.append(equity_df["date"].iloc[-1])
    irr = xirr(flows, dates)

    eq = equity_df["equity"]
    max_dd = (eq / eq.cummax() - 1).min()
    compound_profit = final_equity - cumulative_invested
    fv_at_xirr = compound_fv_from_xirr(dca_flows, equity_df["date"].iloc[-1], irr)

    return {
        "total_invested": cumulative_invested,
        "total_return": total_return,
        "xirr": irr,
        "max_dd": float(max_dd),
        "years": years,
        "final_equity": final_equity,
        "compound_profit_total": compound_profit,
        "compound_fv_at_xirr": fv_at_xirr,
    }


def compound_fv_from_xirr(
    dca_flows: list[tuple[pd.Timestamp, float]],
    final_date: pd.Timestamp,
    rate: float,
    day_count: float = 365.25,
) -> float:
    """將每期投入依 XIRR 複利累積至期末的理論總值（應≈ final_equity）。"""
    if not dca_flows or np.isnan(rate):
        return np.nan
    end = pd.Timestamp(final_date)
    fv = 0.0
    for dt, amt in dca_flows:
        years_fwd = (end - pd.Timestamp(dt)).days / day_count
        fv += amt * (1 + rate) ** years_fwd
    return fv
