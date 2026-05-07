# ============================================================
#  QQQ Trading Strategy — Standalone Backtest
#  Strategy : Adaptive Momentum + Mean-Reversion Hybrid
#  Universe  : QQQ US Equity
#  Data      : Yahoo Finance via yfinance (free, no Bloomberg needed)
# ============================================================
#
#  HOW TO RUN
#  ----------
#  1. Install dependencies:  pip install yfinance pandas numpy matplotlib
#  2. Run:  python3 qqq_strategy_bqnt.py
#  3. Adjust CONFIG section to taste
#
#  To run inside Bloomberg BQNT instead, replace the yfinance
#  data section with bql calls and add:  import bql, import bqviz
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ── 1. CONFIG ───────────────────────────────────────────────
TICKER          = "QQQ"
START_DATE      = "2010-01-01"
END_DATE        = datetime.today().strftime("%Y-%m-%d")
INITIAL_CASH    = 1_000_000   # $1 M notional

# Moving-average periods
EMA_FAST        = 20
EMA_SLOW        = 50
TREND_SMA       = 200         # regime filter — only long above this

# RSI
RSI_PERIOD      = 14
RSI_OVERSOLD    = 35
RSI_OVERBOUGHT  = 70

# Bollinger Bands
BB_PERIOD       = 20
BB_STD          = 2.0

# Risk
ATR_PERIOD      = 14
STOP_ATR_MULT   = 2.5         # stop = entry_price − 2.5 × ATR
VOL_TARGET      = 0.15        # 15 % annualised volatility target for sizing
TRADING_DAYS    = 252

# ── 2. DATA FETCH (yfinance) ────────────────────────────────
raw = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

prices = raw[["Close", "High", "Low", "Volume"]].copy()
prices.columns = ["close", "high", "low", "volume"]
prices.index = pd.to_datetime(prices.index)
prices.sort_index(inplace=True)
prices.dropna(inplace=True)

print(f"Loaded {len(prices):,} trading days  "
      f"|  {prices.index[0].date()} → {prices.index[-1].date()}")

# ── 3. INDICATORS ───────────────────────────────────────────

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta    = series.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def bollinger(series, period=20, num_std=2.0):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std(ddof=0)
    return mid, mid + num_std * sigma, mid - num_std * sigma

ind = pd.DataFrame(index=prices.index)
ind["ema_fast"]  = ema(prices["close"], EMA_FAST)
ind["ema_slow"]  = ema(prices["close"], EMA_SLOW)
ind["sma_trend"] = prices["close"].rolling(TREND_SMA).mean()
ind["rsi"]       = rsi(prices["close"], RSI_PERIOD)
ind["atr"]       = atr(prices["high"], prices["low"], prices["close"], ATR_PERIOD)
ind["bb_mid"], ind["bb_upper"], ind["bb_lower"] = bollinger(
    prices["close"], BB_PERIOD, BB_STD
)
ind["ann_vol"] = (
    prices["close"].pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS)
)

# ── 4. SIGNAL GENERATION ────────────────────────────────────
#
#  Two entry archetypes — both gated by indicators; positions are
#  closed by an exit signal OR a trailing ATR stop.
#
#  MOMENTUM entry  : EMA-fast crosses above EMA-slow
#                    AND price > 200-day SMA (uptrend regime)
#                    AND RSI not overbought
#
#  MEAN-REVERSION  : Price ≤ lower Bollinger Band
#                    AND RSI ≤ oversold threshold
#                    (no trend gate — captures dip-buying opportunities)
#
#  EXIT            : EMA-fast crosses below EMA-slow
#                    OR RSI overbought
#                    OR price ≥ upper Bollinger Band

sig = pd.DataFrame(index=prices.index)
sig["in_uptrend"]   = prices["close"] > ind["sma_trend"]
sig["ema_cross_up"] = (
    (ind["ema_fast"] > ind["ema_slow"]) &
    (ind["ema_fast"].shift(1) <= ind["ema_slow"].shift(1))
)
sig["ema_cross_dn"] = (
    (ind["ema_fast"] < ind["ema_slow"]) &
    (ind["ema_fast"].shift(1) >= ind["ema_slow"].shift(1))
)
sig["entry_momentum"] = (
    sig["ema_cross_up"] & sig["in_uptrend"] & (ind["rsi"] < RSI_OVERBOUGHT)
)
sig["entry_meanrev"]  = (
    (prices["close"] <= ind["bb_lower"]) & (ind["rsi"] <= RSI_OVERSOLD)
)
sig["entry"] = sig["entry_momentum"] | sig["entry_meanrev"]
sig["exit"]  = (
    sig["ema_cross_dn"] |
    (ind["rsi"] >= RSI_OVERBOUGHT) |
    (prices["close"] >= ind["bb_upper"])
)

# ── 5. POSITION CONSTRUCTION ────────────────────────────────
#  Volatility-targeting: size = VOL_TARGET / realised_vol, capped at 1.0

def vol_weight(ann_vol):
    return np.minimum(VOL_TARGET / ann_vol.replace(0, np.nan), 1.0).fillna(0)

bt = pd.DataFrame(index=prices.index)
bt["close"]    = prices["close"]
bt["signal"]   = 0.0   # binary: 1 = long, 0 = flat
bt["position"] = 0.0   # volatility-scaled weight ∈ [0, 1]
bt["stop"]     = np.nan

in_pos     = False
stop_price = np.nan

for i in range(TREND_SMA, len(bt)):
    idx = bt.index[i]
    px  = prices["close"].iloc[i]

    # Stop-loss check (evaluated before exit/entry signals)
    if in_pos and not np.isnan(stop_price) and px < stop_price:
        in_pos     = False
        stop_price = np.nan

    # Exit signal check
    if in_pos and sig["exit"].iloc[i]:
        in_pos     = False
        stop_price = np.nan

    # Entry signal check
    if not in_pos and sig["entry"].iloc[i]:
        in_pos     = True
        stop_price = px - STOP_ATR_MULT * ind["atr"].iloc[i]

    bt.at[idx, "signal"]   = float(in_pos)
    bt.at[idx, "position"] = float(in_pos) * vol_weight(ind["ann_vol"]).iloc[i]
    bt.at[idx, "stop"]     = stop_price

# ── 6. P&L ──────────────────────────────────────────────────
bt["returns"]       = prices["close"].pct_change()
bt["strat_returns"] = bt["position"].shift(1) * bt["returns"]
bt["bh_returns"]    = bt["returns"]

bt["equity_strat"]  = INITIAL_CASH * (1 + bt["strat_returns"]).cumprod()
bt["equity_bh"]     = INITIAL_CASH * (1 + bt["bh_returns"]).cumprod()
bt.dropna(inplace=True)

# ── 7. PERFORMANCE METRICS ──────────────────────────────────

def annual_return(r):
    return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1

def max_drawdown(equity):
    return ((equity - equity.cummax()) / equity.cummax()).min()

def sharpe(r, rf=0.04):
    excess = r - rf / TRADING_DAYS
    return np.sqrt(TRADING_DAYS) * excess.mean() / excess.std()

def sortino(r, rf=0.04):
    excess   = r - rf / TRADING_DAYS
    downside = excess[excess < 0].std()
    return np.sqrt(TRADING_DAYS) * excess.mean() / downside

def calmar(r, equity):
    mdd = abs(max_drawdown(equity))
    return annual_return(r) / mdd if mdd else np.nan

def win_rate(r):
    return (r > 0).mean()

sr, bhr = bt["strat_returns"], bt["bh_returns"]

metrics = pd.DataFrame(
    {
        "Strategy": [
            f"{annual_return(sr):.2%}",
            f"{sr.std() * np.sqrt(TRADING_DAYS):.2%}",
            f"{sharpe(sr):.2f}",
            f"{sortino(sr):.2f}",
            f"{max_drawdown(bt['equity_strat']):.2%}",
            f"{calmar(sr, bt['equity_strat']):.2f}",
            f"{win_rate(sr):.2%}",
        ],
        "Buy & Hold": [
            f"{annual_return(bhr):.2%}",
            f"{bhr.std() * np.sqrt(TRADING_DAYS):.2%}",
            f"{sharpe(bhr):.2f}",
            f"{sortino(bhr):.2f}",
            f"{max_drawdown(bt['equity_bh']):.2%}",
            f"{calmar(bhr, bt['equity_bh']):.2f}",
            f"{win_rate(bhr):.2%}",
        ],
    },
    index=[
        "Annual Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Calmar Ratio",
        "Win Rate (daily)",
    ],
)

divider = "=" * 54
print(f"\n{divider}")
print("  QQQ ADAPTIVE MOMENTUM STRATEGY  —  BACKTEST RESULTS")
print(divider)
print(metrics.to_string())
print(f"{divider}\n")

# ── 8. VISUALISATION ────────────────────────────────────────
fig, axes = plt.subplots(
    4, 1, figsize=(14, 18), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.5, 1, 1]}
)
fig.suptitle("QQQ — Adaptive Momentum + Mean-Reversion Backtest",
             fontsize=13, fontweight="bold")

# ─ Panel 1: Price + indicators ─
ax1 = axes[0]
ax1.plot(prices.index, prices["close"],  label="QQQ Close",         color="#1f77b4", lw=1.2)
ax1.plot(ind.index,    ind["ema_fast"],  label=f"EMA {EMA_FAST}",   color="#ff7f0e", lw=0.9, ls="--")
ax1.plot(ind.index,    ind["ema_slow"],  label=f"EMA {EMA_SLOW}",   color="#2ca02c", lw=0.9, ls="--")
ax1.plot(ind.index,    ind["sma_trend"], label=f"SMA {TREND_SMA}",  color="#d62728", lw=0.9, ls=":")
ax1.fill_between(ind.index, ind["bb_lower"], ind["bb_upper"],
                 alpha=0.07, color="grey", label="Bollinger Bands")
ax1.fill_between(bt.index,
                 prices["close"].min(), prices["close"].max(),
                 where=(bt["signal"] > 0),
                 alpha=0.10, color="green", label="Long exposure")
ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left", fontsize=8, ncol=3)
ax1.grid(True, alpha=0.3)

# ─ Panel 2: Equity curves ─
ax2 = axes[1]
ax2.plot(bt.index, bt["equity_strat"] / INITIAL_CASH, label="Strategy",   color="#2ca02c", lw=1.5)
ax2.plot(bt.index, bt["equity_bh"]    / INITIAL_CASH, label="Buy & Hold", color="#1f77b4", lw=1.5, ls="--")
ax2.set_ylabel("Normalised Equity")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.3)

# ─ Panel 3: Drawdown ─
ax3 = axes[2]
dd_strat = (bt["equity_strat"] - bt["equity_strat"].cummax()) / bt["equity_strat"].cummax()
dd_bh    = (bt["equity_bh"]    - bt["equity_bh"].cummax())    / bt["equity_bh"].cummax()
ax3.fill_between(bt.index, dd_strat, 0, alpha=0.65, color="#2ca02c", label="Strategy DD")
ax3.fill_between(bt.index, dd_bh,    0, alpha=0.30, color="#1f77b4", label="Buy & Hold DD")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax3.set_ylabel("Drawdown")
ax3.legend(loc="lower left", fontsize=8)
ax3.grid(True, alpha=0.3)

# ─ Panel 4: RSI ─
ax4 = axes[3]
ax4.plot(ind.index, ind["rsi"], color="#9467bd", lw=0.9, label="RSI 14")
ax4.axhline(RSI_OVERSOLD,   color="green", lw=0.8, ls="--")
ax4.axhline(RSI_OVERBOUGHT, color="red",   lw=0.8, ls="--")
ax4.axhline(50,             color="grey",  lw=0.5)
ax4.fill_between(ind.index, RSI_OVERSOLD,   ind["rsi"],
                 where=(ind["rsi"] < RSI_OVERSOLD),  alpha=0.25, color="green")
ax4.fill_between(ind.index, RSI_OVERBOUGHT, ind["rsi"],
                 where=(ind["rsi"] > RSI_OVERBOUGHT), alpha=0.25, color="red")
ax4.set_ylim(0, 100)
ax4.set_ylabel("RSI")
ax4.legend(loc="upper left", fontsize=8)
ax4.grid(True, alpha=0.3)

ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax4.xaxis.set_major_locator(mdates.YearLocator(2))
plt.xticks(rotation=30)
plt.tight_layout()

output_file = "qqq_strategy_backtest.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
plt.show()
print(f"Chart saved → {output_file}")

# ── 9. TRADE LOG (optional) ─────────────────────────────────
entries = bt.index[bt["signal"].diff() == 1]
exits   = bt.index[bt["signal"].diff() == -1]

n_trades = min(len(entries), len(exits))
trades = pd.DataFrame({
    "Entry Date": entries[:n_trades],
    "Exit Date":  exits[:n_trades],
    "Entry Price": [bt["close"].loc[d] for d in entries[:n_trades]],
    "Exit Price":  [bt["close"].loc[d] for d in exits[:n_trades]],
})
trades["Return"] = (trades["Exit Price"] / trades["Entry Price"] - 1).map("{:.2%}".format)
trades["Days"]   = (pd.to_datetime(trades["Exit Date"]) -
                    pd.to_datetime(trades["Entry Date"])).dt.days

print(f"\nTotal trades executed : {len(trades)}")
print(trades.to_string(index=False))
