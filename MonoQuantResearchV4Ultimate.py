# pragma pylint: disable=missing-module-docstring, invalid-name, pointless-string-statement
# flake8: noqa
"""
MonoQuantResearchV4Ultimate - Aggressive Test Variant (Upbit/KRW focus)

This is a consolidated single-file Freqtrade strategy implementing the agreed modifications:
- Max concurrent trades: 3 (internal guard + config should also set max_open_trades=3)
- Whitelist assumed 5 pairs (config-side)
- BTC regime: risk_on + neutral gate (neutral allows reduced stake; volatility spike still blocks)
- Faster first entry: relaxed gates, shorter windows
- Risk containment: ATR%-based dynamic stake + multi-stage stoploss (BE + trailing)

Notes:
- Spot long-only.
- Designed to run on 5m with 1h informative timeframe.
- No external network calls beyond Freqtrade's data provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
import talib.abstract as ta


def _ema_slope(series: Series, length: int = 5) -> Series:
    """
    Approximate EMA slope as (ema - ema.shift(n)) / ema.shift(n).
    Uses a short lookback for responsiveness.
    """
    ema = ta.EMA(series, timeperiod=length)
    prev = ema.shift(length)
    return (ema - prev) / prev.replace(0, np.nan)


def _safe_zscore(series: Series, window: int) -> Series:
    mean = series.rolling(window=window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window=window, min_periods=max(5, window // 4)).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def _atrp(df: DataFrame, length: int = 14) -> Series:
    atr = ta.ATR(df, timeperiod=length)
    return atr / df["close"].replace(0, np.nan)


def _donchian(df: DataFrame, length: int = 20) -> DataFrame:
    out = df.copy()
    out["donch_high"] = out["high"].rolling(length, min_periods=length).max()
    out["donch_low"] = out["low"].rolling(length, min_periods=length).min()
    out["donch_high_prev"] = out["donch_high"].shift(1)
    out["donch_low_prev"] = out["donch_low"].shift(1)
    return out


@dataclass
class _OrderIssueState:
    window_start: Optional[datetime] = None
    count: int = 0


class MonoQuantResearchV4Ultimate(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    # ---- Timeframes
    timeframe = "5m"
    informative_timeframe = "1h"

    process_only_new_candles = True

    # ---- Startup candles (reduced vs original 2500 for live/dr) 
    startup_candle_count = 800

    # ---- Multi-position
    max_open_trades = 3
    max_concurrent_trades_internal = 3  # internal guard

    # ===== BTC market regime params (relaxed) =====
    btc_ema_fast = 50
    btc_ema_slow = 200
    btc_atr_len = 14
    btc_rsi_len = 14

    btc_vol_spike_mult = 1.8
    btc_atrp_sma_win = 48
    btc_atrp_abs_max = 0.06

    btc_rsi_min = 42.0
    btc_slope_min = -0.0005

    # ===== Asset regime =====
    adx_len = 14
    adx_trend_min = 19.0
    adx_range_max = 21.0
    slope_atr_min = 0.05

    # ===== Micro / quality filters (relaxed) =====
    atr_len = 14
    atrp_min = 0.0010
    atrp_max = 0.0350
    atrp_sma_win = 72
    atrp_spike_mult = 2.2

    gap_pct_max = 0.025
    range_pct_max = 0.050

    vol_z_win = 24
    vol_z_min = 0.20
    vol_z_spike_max = 5.5

    min_qvol_1h_krw = 8_000_000.0

    rsi_len = 14
    rsi_overheat = 82.0

    # ===== Structures =====
    donch_len = 18
    bb_len = 20
    bb_dev = 2.0

    # ===== Score thresholds (relaxed) =====
    score_breakout_th = 0.64
    score_pullback_th = 0.61
    score_meanrev_th = 0.58

    # ===== Stake sizing =====
    base_stake_krw = 50_000.0
    min_stake_safe_krw = 10_200.0
    atrp_target = 0.0080

    stake_clip_min = 0.25
    stake_clip_max = 2.50

    stake_mult_breakout = 1.05
    stake_mult_pullback = 0.95
    stake_mult_meanrev = 0.80

    stake_when_riskoff = 0.35  # when btc_risk_on is False but btc_neutral True

    # ===== Order / safety (relaxed for test) =====
    entry_timeout_min = 8.0
    exit_timeout_min = 8.0
    lock_after_order_issue_min = 15.0

    order_issue_window_min = 12.0
    order_issue_max = 10
    global_pause_min = 25.0

    post_exit_lock_min = 10.0

    # ===== Stoploss tuning =====
    sl_atrp_mult_breakout = 5.0
    sl_atrp_mult_pullback = 4.7
    sl_atrp_mult_meanrev = 4.0

    sl_be_profit = 0.010
    sl_be_open_rel = -0.005

    sl_trail_start = 0.025
    sl_trail_atrp_mult = 3.1

    # ---- Defaults / order types
    minimal_roi: Dict[str, float] = {"0": 10.0}
    stoploss = -0.99  # handled by custom_stoploss

    use_custom_stoploss = True
    use_custom_exit = True

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Internal state (per-run)
    _global_pause_until: Optional[datetime] = None
    _order_issues: _OrderIssueState = _OrderIssueState()

    # ---- Informative pairs
    def informative_pairs(self):
        # Need BTC/KRW (or BTC/USDT etc). For Upbit KRW market, BTC/KRW is typical.
        # Also include each traded pair on 1h for regime filter.
        pairs = self.dp.current_whitelist()
        inf = [(p, self.informative_timeframe) for p in pairs]
        # Add BTC/KRW informative
        if "BTC/KRW" not in pairs:
            inf.append(("BTC/KRW", self.informative_timeframe))
        return inf

    # ---- Utility: time helpers
    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def _global_pause_active(self, now: datetime) -> bool:
        return self._global_pause_until is not None and now < self._global_pause_until

    # Pair cooldown via built-in lock_pair if available at runtime.
    def _lock_pair_minutes(self, pair: str, minutes: float, now: datetime) -> None:
        try:
            self.lock_pair(pair, now + timedelta(minutes=float(minutes)))
        except Exception:
            # In backtesting manual locks may be ignored; that's fine.
            return

    def _is_pair_locked(self, pair: str) -> bool:
        try:
            return self.is_pair_locked(pair)
        except Exception:
            return False

    def _register_order_issue(self, now: datetime) -> None:
        st = self._order_issues
        window = timedelta(minutes=float(self.order_issue_window_min))
        if st.window_start is None or (now - st.window_start) > window:
            st.window_start = now
            st.count = 1
        else:
            st.count += 1

        if st.count >= int(self.order_issue_max):
            # global pause
            self._global_pause_until = now + timedelta(minutes=float(self.global_pause_min))

    # ---- Indicators
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        if df is None or df.empty:
            return df

        # Core indicators (5m)
        df["rsi"] = ta.RSI(df, timeperiod=self.rsi_len)
        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema200"] = ta.EMA(df, timeperiod=200)

        df["adx"] = ta.ADX(df, timeperiod=self.adx_len)
        df["atrp"] = _atrp(df, length=self.atr_len)
        df["atrp_sma"] = df["atrp"].rolling(self.atrp_sma_win, min_periods=max(10, self.atrp_sma_win // 4)).mean()

        # Candle geometry filters
        df["gap_pct"] = (df["open"] - df["close"].shift(1)).abs() / df["close"].shift(1).replace(0, np.nan)
        df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

        # Volume
        # For Upbit, volume is coin units, not KRW. We'll compute qvol as close*volume.
        df["qvol"] = df["close"] * df["volume"]
        # informative 1h volume zscore will be created on merged 1h.
        df["qvol_z"] = _safe_zscore(df["qvol"], window=self.vol_z_win)

        # Donchian
        df = _donchian(df, length=self.donch_len)

        # Bollinger Bands
        bb = ta.BBANDS(df, timeperiod=self.bb_len, nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0)
        df["bb_upper"] = bb["upperband"]
        df["bb_middle"] = bb["middleband"]
        df["bb_lower"] = bb["lowerband"]

        # Heat filter
        df["heat_ok"] = df["rsi"] < float(self.rsi_overheat)

        # Regime (5m): trend vs range
        # Use EMA200 slope proxy via ATRP and adx
        df["ema200_slope"] = _ema_slope(df["close"], length=5)
        df["regime_trend"] = (df["adx"] >= float(self.adx_trend_min))
        df["regime_range"] = (df["adx"] <= float(self.adx_range_max))

        # Micro quality
        atrp_ok = (df["atrp"] >= float(self.atrp_min)) & (df["atrp"] <= float(self.atrp_max))
        atrp_spike = df["atrp"] > (df["atrp_sma"] * float(self.atrp_spike_mult))
        gap_ok = df["gap_pct"].fillna(0) <= float(self.gap_pct_max)
        range_ok = df["range_pct"].fillna(0) <= float(self.range_pct_max)
        vol_ok = (df["qvol_z"].fillna(-999) >= float(self.vol_z_min)) & (df["qvol_z"].fillna(0) <= float(self.vol_z_spike_max))

        # ---- Informative merge (1h): BTC + pair 1h qvol
        # Pair 1h
        inf = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        if inf is not None and not inf.empty:
            inf = inf.copy()
            inf["qvol_1h"] = inf["close"] * inf["volume"]
            inf["qvol_1h_krw"] = inf["qvol_1h"]
            # 1h qvol min threshold (approx KRW proxy)
            inf["qvol_1h_ok"] = inf["qvol_1h_krw"] >= float(self.min_qvol_1h_krw)
            df = merge_informative_pair(df, inf, self.timeframe, self.informative_timeframe, ffill=True)

        # BTC 1h
        btc = self.dp.get_pair_dataframe(pair="BTC/KRW", timeframe=self.informative_timeframe)
        if btc is not None and not btc.empty:
            btc = btc.copy()
            btc["btc_close_1h"] = btc["close"]
            btc["btc_ema50_1h"] = ta.EMA(btc, timeperiod=int(self.btc_ema_fast))
            btc["btc_ema200_1h"] = ta.EMA(btc, timeperiod=int(self.btc_ema_slow))
            btc["btc_rsi_1h"] = ta.RSI(btc, timeperiod=int(self.btc_rsi_len))
            btc["btc_atr_1h"] = ta.ATR(btc, timeperiod=int(self.btc_atr_len))
            btc["btc_atrp_1h"] = btc["btc_atr_1h"] / btc["btc_close_1h"].replace(0, np.nan)
            btc["btc_atrp_sma_1h"] = btc["btc_atrp_1h"].rolling(int(self.btc_atrp_sma_win), min_periods=max(10, int(self.btc_atrp_sma_win)//4)).mean()
            btc["btc_ema50_slope_1h"] = (btc["btc_ema50_1h"] - btc["btc_ema50_1h"].shift(5)) / btc["btc_ema50_1h"].shift(5).replace(0, np.nan)

            df = merge_informative_pair(df, btc[
                [
                    "btc_close_1h",
                    "btc_ema50_1h",
                    "btc_ema200_1h",
                    "btc_rsi_1h",
                    "btc_atrp_1h",
                    "btc_atrp_sma_1h",
                    "btc_ema50_slope_1h",
                ]
            ], self.timeframe, self.informative_timeframe, ffill=True)

        # ---- BTC regime columns
        btc_valid = (
            df.get("btc_close_1h", pd.Series(index=df.index, dtype=float)).notna()
            & df.get("btc_ema200_1h", pd.Series(index=df.index, dtype=float)).notna()
            & df.get("btc_ema50_1h", pd.Series(index=df.index, dtype=float)).notna()
        )

        df["btc_vol_spike"] = btc_valid & (
            (df["btc_atrp_1h"] > (df["btc_atrp_sma_1h"] * float(self.btc_vol_spike_mult)))
            | (df["btc_atrp_1h"] > float(self.btc_atrp_abs_max))
        )

        df["btc_risk_on"] = btc_valid & (
            (df["btc_close_1h"] > df["btc_ema200_1h"])
            & (df["btc_ema50_slope_1h"] > float(self.btc_slope_min))
            & (df["btc_rsi_1h"] > float(self.btc_rsi_min))
            & (~df["btc_vol_spike"])
        )

        df["btc_neutral"] = btc_valid & (
            (df["btc_close_1h"] > df["btc_ema200_1h"]) & (~df["btc_vol_spike"])
        )

        # ---- Final micro_ok: include 1h qvol if present
        qvol_1h_ok = df.get("qvol_1h_ok_1h", pd.Series(True, index=df.index))
        df["micro_ok"] = atrp_ok & (~atrp_spike) & gap_ok & range_ok & vol_ok & qvol_1h_ok.fillna(True)

        # ---- Scores
        # Breakout score: distance above donch high prev + volume z + trend
        # Pullback score: oversold depth + reclaim strength + trend
        # Meanrev score: bb reclaim + rsi reversal + range-ness
        dist_break = (df["close"] / df["donch_high_prev"].replace(0, np.nan)) - 1.0
        dist_break = dist_break.clip(lower=0.0, upper=0.2)
        df["score_breakout"] = (
            0.45 * dist_break.fillna(0.0) / 0.2
            + 0.35 * (df["qvol_z"].clip(-2, 6).fillna(0.0) / 6.0)
            + 0.20 * (df["adx"].clip(0, 50).fillna(0.0) / 50.0)
        ).clip(0, 1)

        oversold = ((df["close"] < df["bb_lower"]).astype(float) + (df["rsi"] < 42.0).astype(float)).clip(0, 1)
        reclaim = ((df["close"] > df["ema20"]) & (df["close"].shift(1) <= df["ema20"].shift(1))).astype(float)
        df["score_pullback"] = (
            0.45 * oversold.fillna(0.0)
            + 0.35 * reclaim.fillna(0.0)
            + 0.20 * (df["adx"].clip(0, 50).fillna(0.0) / 50.0)
        ).clip(0, 1)

        bb_reclaim = ((df["close"] > df["bb_lower"]) & (df["close"].shift(1) < df["bb_lower"].shift(1))).astype(float)
        rsi_confirm = ((df["rsi"] < 48.0) & (df["rsi"] > df["rsi"].shift(1))).astype(float)
        range_bias = (1.0 - (df["adx"].clip(10, 35) - 10.0) / 25.0).clip(0, 1)  # low adx => higher score
        df["score_meanrev"] = (0.45 * bb_reclaim + 0.35 * rsi_confirm + 0.20 * range_bias).clip(0, 1)

        return df

    # ---- Entries
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        if df is None or df.empty:
            return df

        df["enter_long"] = 0
        df["enter_tag"] = ""

        market_ok = (df["btc_risk_on"].fillna(False) | df["btc_neutral"].fillna(False))
        quality_ok = df["micro_ok"].fillna(False)
        heat_ok = df["heat_ok"].fillna(False)

        # Breakout
        donch_break = df["close"] > df["donch_high_prev"]
        vol_ok = df["qvol_z"].fillna(-999) >= float(self.vol_z_min)

        cond_breakout = (
            market_ok
            & df["regime_trend"].fillna(False)
            & quality_ok
            & ((heat_ok) | (df["rsi"] < 86.0))
            & donch_break
            & vol_ok
            & (df["atrp"] >= max(float(self.atrp_min), 0.0015))
            & (df["score_breakout"] >= float(self.score_breakout_th))
        )
        df.loc[cond_breakout, ["enter_long", "enter_tag"]] = (1, "TB_BREAKOUT")

        # Pullback
        oversold = (df["close"] < df["bb_lower"]) | (df["rsi"] < 42.0)
        reclaim = (df["close"] > df["ema20"]) & (df["close"].shift(1) <= df["ema20"].shift(1))

        cond_pullback = (
            (df["enter_long"] == 0)
            & market_ok
            & (df["regime_trend"].fillna(False))
            & quality_ok
            & oversold
            & reclaim
            & (df["score_pullback"] >= float(self.score_pullback_th))
        )
        df.loc[cond_pullback, ["enter_long", "enter_tag"]] = (1, "TP_PULLBACK")

        # Mean reversion
        bb_reclaim = (df["close"] > df["bb_lower"]) & (df["close"].shift(1) < df["bb_lower"].shift(1))
        rsi_confirm = (df["rsi"] < 48.0) & (df["rsi"] > df["rsi"].shift(1))

        cond_meanrev = (
            (df["enter_long"] == 0)
            & (df["btc_vol_spike"].fillna(False) == False)
            & (df["regime_range"].fillna(False) | (df["adx"] < 24.0))
            & quality_ok
            & bb_reclaim
            & rsi_confirm
            & (df["score_meanrev"] >= float(self.score_meanrev_th))
        )
        df.loc[cond_meanrev, ["enter_long", "enter_tag"]] = (1, "MR_RANGE")

        return df

    # ---- Exits (use custom_exit; keep populate_exit_trend minimal)
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        if df is None or df.empty:
            return df
        df["exit_long"] = 0
        df["exit_tag"] = ""
        return df

    # ---- Stake sizing
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        now = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        if self._global_pause_active(now):
            return 0.0
        if self._is_pair_locked(pair):
            return 0.0

        # internal concurrent guard
        try:
            if Trade.get_open_trade_count() >= int(self.max_concurrent_trades_internal):
                return 0.0
        except Exception:
            pass

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return float(self.base_stake_krw)

        last = df.iloc[-1].to_dict()

        # BTC spike blocks completely
        if bool(last.get("btc_vol_spike", False)):
            return 0.0

        btc_risk_on = bool(last.get("btc_risk_on", False))
        risk_mult = 1.0 if btc_risk_on else float(self.stake_when_riskoff)

        atrp = float(last.get("atrp", np.nan))
        if not np.isfinite(atrp) or atrp <= 0:
            vol_mult = 1.0
        else:
            vol_mult = float(self.atrp_target) / atrp
            vol_mult = float(np.clip(vol_mult, self.stake_clip_min, self.stake_clip_max))

        tag = (entry_tag or "").upper()
        if "TB_BREAKOUT" in tag:
            tag_mult = float(self.stake_mult_breakout)
        elif "TP_PULLBACK" in tag:
            tag_mult = float(self.stake_mult_pullback)
        elif "MR_RANGE" in tag:
            tag_mult = float(self.stake_mult_meanrev)
        else:
            tag_mult = 1.0

        stake = float(self.base_stake_krw) * vol_mult * risk_mult * tag_mult

        # enforce min/max
        hard_min = max(float(self.min_stake_safe_krw), float(min_stake))
        stake = float(np.clip(stake, hard_min, float(max_stake)))

        return stake

    # ---- Confirm trade entry (guard)
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        now = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)

        if self._global_pause_active(now):
            return False
        if self._is_pair_locked(pair):
            return False
        try:
            if Trade.get_open_trade_count() >= int(self.max_concurrent_trades_internal):
                return False
        except Exception:
            pass

        # Block if BTC spike
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is not None and not df.empty:
            last = df.iloc[-1]
            if bool(last.get("btc_vol_spike", False)):
                return False

        return True

    # ---- Custom stoploss
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Return stoploss as relative distance from current_rate (negative value).
        Multi-stage:
        - Initial: based on entry_tag with ATRP multiplier (relative to open)
        - BE: once profit >= sl_be_profit, move stop to near open (open-relative)
        - Trail: once profit >= sl_trail_start, trail by ATRP*mult relative to open (converted to current-rate relative)
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return -0.25

        last = df.iloc[-1]
        atrp = float(last.get("atrp", np.nan))
        if not np.isfinite(atrp) or atrp <= 0:
            atrp = float(self.atrp_target)

        tag = (trade.enter_tag or "").upper()
        if "TB_BREAKOUT" in tag:
            mult = float(self.sl_atrp_mult_breakout)
        elif "TP_PULLBACK" in tag:
            mult = float(self.sl_atrp_mult_pullback)
        elif "MR_RANGE" in tag:
            mult = float(self.sl_atrp_mult_meanrev)
        else:
            mult = float(self.sl_atrp_mult_pullback)

        # base stop relative to open
        stop_from_open = -mult * atrp  # negative (e.g., -0.04)
        # break-even adjustment
        if current_profit >= float(self.sl_be_profit):
            stop_from_open = max(stop_from_open, float(self.sl_be_open_rel))

        # trailing
        if current_profit >= float(self.sl_trail_start):
            # trail distance relative to open
            trail_from_open = -float(self.sl_trail_atrp_mult) * atrp
            stop_from_open = max(stop_from_open, trail_from_open)

        # Convert open-relative stop to current-rate relative.
        # If open price = trade.open_rate, stop price = open_rate * (1 + stop_from_open).
        # Relative to current_rate: (stop_price/current_rate) - 1.
        stop_price = float(trade.open_rate) * (1.0 + float(stop_from_open))
        rel = (stop_price / float(current_rate)) - 1.0
        # clamp: never tighter than -0.001 and never looser than -0.99
        rel = float(np.clip(rel, -0.99, -0.001))
        return rel

    # ---- Custom exit
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        Simple regime-based exit + overheat exit.
        Also applies post-exit lock to avoid immediate re-entry on the same pair.
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        last = df.iloc[-1]

        # Exit on BTC vol spike when in profit (avoid panic sells at a loss here)
        if bool(last.get("btc_vol_spike", False)) and current_profit > 0.002:
            self._lock_pair_minutes(pair, self.post_exit_lock_min, current_time)
            return "BTC_VOL_SPIKE_EXIT"

        # Exit if overheat + fading (RSI drop)
        rsi = float(last.get("rsi", np.nan))
        rsi_prev = float(df["rsi"].iloc[-2]) if len(df) > 2 and "rsi" in df.columns else rsi
        if np.isfinite(rsi) and rsi > 80.0 and rsi < rsi_prev and current_profit > 0.004:
            self._lock_pair_minutes(pair, self.post_exit_lock_min, current_time)
            return "OVERHEAT_FADE"

        # Trend breakdown: price falls below ema50 after being above
        ema50 = float(last.get("ema50", np.nan))
        if np.isfinite(ema50):
            was_above = bool((df["close"].shift(1).iloc[-1] > df["ema50"].shift(1).iloc[-1]) if len(df) > 2 else False)
            now_below = bool(last["close"] < ema50)
            if was_above and now_below and current_profit > 0.001:
                self._lock_pair_minutes(pair, self.post_exit_lock_min, current_time)
                return "EMA50_LOSS"

        return None

    # ---- Optional callbacks: order timeouts
    def check_entry_timeout(self, pair: str, trade: Trade, order: object, current_time: datetime, **kwargs) -> bool:
        """
        Return True to cancel entry order.
        For market orders this usually won't trigger, but kept for compatibility.
        """
        # keep a soft cancel if order hangs beyond entry_timeout_min
        try:
            if hasattr(order, "order_date") and order.order_date:
                dt = order.order_date
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if (current_time.replace(tzinfo=timezone.utc) - dt) > timedelta(minutes=float(self.entry_timeout_min)):
                    self._register_order_issue(self._utcnow())
                    self._lock_pair_minutes(pair, self.lock_after_order_issue_min, current_time)
                    return True
        except Exception:
            return False
        return False

    def check_exit_timeout(self, pair: str, trade: Trade, order: object, current_time: datetime, **kwargs) -> bool:
        """
        Return True to cancel exit order.
        """
        try:
            if hasattr(order, "order_date") and order.order_date:
                dt = order.order_date
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if (current_time.replace(tzinfo=timezone.utc) - dt) > timedelta(minutes=float(self.exit_timeout_min)):
                    self._register_order_issue(self._utcnow())
                    self._lock_pair_minutes(pair, self.lock_after_order_issue_min, current_time)
                    return True
        except Exception:
            return False
        return False
