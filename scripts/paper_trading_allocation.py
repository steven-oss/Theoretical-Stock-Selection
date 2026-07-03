"""策略池水位與下一筆買進額度（對齊 backtest_portfolio._strategy_entry_allocate）。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest_portfolio import (
    STRATEGY_DEPLOY_CASH_PCT,
    _strategy_entry_allocate,
    calc_commission,
    sell_proceeds_after_costs,
)
from paper_trading_summary import (
    latest_closes,
    load_open_positions,
    load_settings,
    load_transactions,
)


@dataclass
class StrategySnapshot:
    as_of: pd.Timestamp
    cash: float
    position_value: float
    equity: float
    cash_pct: float
    open_count: int
    max_positions: int
    slots_available: int
    slot_weight: float


@dataclass
class BuyAllocation:
    budget: float
    rule: str
    target_per_slot: float
    slot_cap: float
    slots_available: int


def _norm_code(sid) -> str:
    s = str(sid).strip()
    return s.zfill(4) if s.isdigit() else s


def compute_strategy_snapshot(
    as_of: pd.Timestamp | None = None,
    hypothetical_sells: list[str] | None = None,
) -> tuple[StrategySnapshot, pd.DataFrame, dict[str, float]]:
    """依 transactions + open_positions 計算策略池現金、淨值、水位。"""
    settings = load_settings()
    tx = load_transactions()
    positions = load_open_positions()
    max_positions = int(float(settings["最大持倉檔數"]))
    slot_weight = float(settings.get("單檔上限比例", 0.2) or 0.2)

    as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()

    tx = tx[tx["交易日期"] <= as_of_ts]
    closes = latest_closes(as_of_ts)

    strategy_cash = 0.0
    for _, row in tx[tx["標的類型"] == "策略"].iterrows():
        action = str(row["動作"]).strip()
        if action == "入金":
            strategy_cash += row["成交金額"]
        elif action == "賣":
            strategy_cash += row["成交金額"] - row["手續費"]
        elif action == "買":
            strategy_cash -= row["成交金額"] + row["手續費"]

    sell_codes = {_norm_code(c) for c in (hypothetical_sells or [])}
    pos_rows = []
    position_value = 0.0
    open_count = 0

    if not positions.empty:
        for _, p in positions.iterrows():
            code = _norm_code(p["股票代碼"])
            shares = float(p["股數"])
            price = closes.get(code, float("nan"))
            if code in sell_codes and pd.notna(price):
                net, gross, comm, tax = sell_proceeds_after_costs(shares, price)
                strategy_cash += net
                pos_rows.append(
                    {
                        "股票代碼": code,
                        "股票名稱": p.get("股票名稱", ""),
                        "股數": int(shares),
                        "現價": price,
                        "市值": shares * price,
                        "模擬賣出淨額": net,
                        "手續費": comm,
                        "證交稅": tax,
                        "狀態": "模擬賣出",
                    }
                )
                continue

            open_count += 1
            mv = shares * price if pd.notna(price) else float("nan")
            if pd.notna(mv):
                position_value += mv
            pos_rows.append(
                {
                    "股票代碼": code,
                    "股票名稱": p.get("股票名稱", ""),
                    "股數": int(shares),
                    "現價": price,
                    "市值": mv,
                    "狀態": "持有中",
                }
            )

    equity = strategy_cash + position_value
    cash_pct = strategy_cash / equity if equity > 0 else 0.0
    slots_available = max(max_positions - open_count, 0)

    snapshot = StrategySnapshot(
        as_of=as_of_ts,
        cash=strategy_cash,
        position_value=position_value,
        equity=equity,
        cash_pct=cash_pct,
        open_count=open_count,
        max_positions=max_positions,
        slots_available=slots_available,
        slot_weight=slot_weight,
    )
    return snapshot, pd.DataFrame(pos_rows), closes


def next_buy_allocation(snapshot: StrategySnapshot) -> BuyAllocation:
    """計算下一筆買進可用額度（含單檔上限比例）。"""
    if snapshot.slots_available <= 0 or snapshot.cash <= 0 or snapshot.equity <= 0:
        return BuyAllocation(0.0, "無空位或無現金", 0.0, 0.0, snapshot.slots_available)

    target_per_slot = snapshot.equity / snapshot.max_positions
    raw = _strategy_entry_allocate(
        snapshot.cash,
        snapshot.equity,
        snapshot.open_count,
        snapshot.max_positions,
        deploy_cash_pct=STRATEGY_DEPLOY_CASH_PCT,
    )
    slot_cap = snapshot.equity * snapshot.slot_weight
    budget = min(raw, slot_cap, snapshot.cash)

    if snapshot.cash_pct > STRATEGY_DEPLOY_CASH_PCT:
        rule = f"水位>{STRATEGY_DEPLOY_CASH_PCT:.0%}：現金÷剩餘空位"
    else:
        rule = f"水位≤{STRATEGY_DEPLOY_CASH_PCT:.0%}：淨值÷{snapshot.max_positions}"

    return BuyAllocation(
        budget=budget,
        rule=rule,
        target_per_slot=target_per_slot,
        slot_cap=slot_cap,
        slots_available=snapshot.slots_available,
    )


def integer_buy_from_budget(budget: float, price: float) -> dict[str, float | int]:
    """零股：在預算內往下取整股數（含手續費）。"""
    if budget <= 0 or price <= 0:
        return {"shares": 0, "gross": 0.0, "commission": 0.0, "total": 0.0}

    shares = int((budget - calc_commission(budget)) // price)
    while shares > 0:
        gross = shares * price
        comm = calc_commission(gross)
        total = gross + comm
        if total <= budget:
            return {"shares": shares, "gross": gross, "commission": comm, "total": total}
        shares -= 1
    return {"shares": 0, "gross": 0.0, "commission": 0.0, "total": 0.0}


def simulate_sequential_buys(
    snapshot: StrategySnapshot,
    prices: dict[str, float],
) -> pd.DataFrame:
    """同一日多檔訊號：每買一檔後重新計算下一檔額度。"""
    rows = []
    cash = snapshot.cash
    open_count = snapshot.open_count
    pos_value = snapshot.position_value

    for code, price in prices.items():
        equity = cash + pos_value
        if open_count >= snapshot.max_positions or cash <= 0:
            break

        snap = StrategySnapshot(
            as_of=snapshot.as_of,
            cash=cash,
            position_value=pos_value,
            equity=equity,
            cash_pct=cash / equity if equity > 0 else 0.0,
            open_count=open_count,
            max_positions=snapshot.max_positions,
            slots_available=snapshot.max_positions - open_count,
            slot_weight=snapshot.slot_weight,
        )
        alloc = next_buy_allocation(snap)
        order = integer_buy_from_budget(alloc.budget, price)
        rows.append(
            {
                "股票代碼": _norm_code(code),
                "現價": price,
                "建議額度": alloc.budget,
                "建議股數": order["shares"],
                "成交金額": order["gross"],
                "手續費": order["commission"],
                "總花費": order["total"],
                "分配規則": alloc.rule,
            }
        )
        if order["shares"] <= 0:
            continue

        cash -= order["total"]
        pos_value += order["gross"]
        open_count += 1

    return pd.DataFrame(rows)


def show_buy_allocation(
    as_of: str | None = None,
    hypothetical_sells: list[str] | None = None,
    candidate_prices: dict[str, float] | None = None,
) -> BuyAllocation:
    """印出策略池水位與下一筆買進建議；可在 Jupyter 直接呼叫。"""
    as_of_ts = pd.Timestamp(as_of) if as_of else None
    snapshot, positions_df, closes = compute_strategy_snapshot(
        as_of_ts, hypothetical_sells=hypothetical_sells
    )
    alloc = next_buy_allocation(snapshot)

    print("=" * 60)
    print(f"【策略池水位】截至 {snapshot.as_of.date()}")
    print("=" * 60)
    print(f"  現金：{snapshot.cash:,.1f}")
    print(f"  持倉市值：{snapshot.position_value:,.1f}")
    print(f"  淨值：{snapshot.equity:,.1f}")
    print(f"  現金水位：{snapshot.cash_pct:.1%}  （門檻 {STRATEGY_DEPLOY_CASH_PCT:.0%}）")
    print(f"  持倉：{snapshot.open_count}/{snapshot.max_positions}  |  剩餘空位：{snapshot.slots_available}")
    print()
    print("【下一筆買進額度】")
    print(f"  規則：{alloc.rule}")
    print(f"  等權目標（淨值÷{snapshot.max_positions}）：{alloc.target_per_slot:,.1f}")
    print(f"  單檔上限（淨值×{snapshot.slot_weight:.0%}）：{alloc.slot_cap:,.1f}")
    print(f"  → 建議單筆預算：{alloc.budget:,.1f} 元（含手續費）")

    if hypothetical_sells:
        print(f"\n  ※ 已模擬賣出：{', '.join(hypothetical_sells)}")

    if not positions_df.empty:
        print("\n【持倉明細】")
        print(positions_df.to_string(index=False))

    if candidate_prices:
        print("\n【若明日有買進訊號 — 依序分配】")
        seq = simulate_sequential_buys(snapshot, candidate_prices)
        if seq.empty:
            print("  無法分配（無空位或無現金）")
        else:
            print(seq.to_string(index=False))

    return alloc
