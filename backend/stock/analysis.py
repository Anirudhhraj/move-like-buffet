"""
Pure computation functions for stock analysis.

Each function takes DataFrames/dicts as input and returns plain dicts.
No HTTP, no yfinance imports, no side effects.
Testable in isolation.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd


# ── Safe math helpers ────────────────────────────────────────────────────

def _safe(df: pd.DataFrame, row: str, col=None) -> Optional[float]:
    try:
        v = df.loc[row, col] if col is not None else df.loc[row].iloc[0]
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def _find_row(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.index:
            return c
    return None


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    r = num / den
    return None if not math.isfinite(r) else round(r, 4)


def _pct(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    return None if not math.isfinite(val) else round(val * 100, 2)


def _safe_round(val: Optional[float], decimals: int = 2) -> Optional[float]:
    if val is None or not math.isfinite(val):
        return None
    return round(val, decimals)


# ── Buffett Ratios ───────────────────────────────────────────────────────

def compute_buffett_ratios(
    fin: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame, info: dict,
) -> dict:
    """Compute Buffett-style financial ratios across all available periods."""
    eps_vals = []
    for col in fin.columns:
        eps_vals.append(_safe(fin, "Basic EPS", col))

    results = []
    for i, col in enumerate(fin.columns):
        gp   = _safe(fin, _find_row(fin, ["Gross Profit"]) or "Gross Profit", col)
        rev  = _safe(fin, _find_row(fin, ["Total Revenue", "Revenue"]) or "Total Revenue", col)
        sga  = _safe(fin, _find_row(fin, ["Selling General And Administration", "Selling General Administrative"]) or "SGA", col)
        rnd  = _safe(fin, _find_row(fin, ["Research And Development", "Research And Development Expenses"]) or "R&D", col)
        dep  = _safe(fin, _find_row(fin, ["Reconciled Depreciation", "Depreciation And Amortization"]) or "Depreciation", col)
        iexp = _safe(fin, _find_row(fin, ["Interest Expense", "Interest Expense Non Operating"]) or "Interest Expense", col)
        oinc = _safe(fin, _find_row(fin, ["Operating Income"]) or "Operating Income", col)
        tax  = _safe(fin, _find_row(fin, ["Tax Provision"]) or "Tax Provision", col)
        pre  = _safe(fin, _find_row(fin, ["Pretax Income"]) or "Pretax Income", col)
        ni   = _safe(fin, _find_row(fin, ["Net Income"]) or "Net Income", col)

        bs_col = _col_at(bs, i)
        cash       = _safe(bs, "Cash And Cash Equivalents", bs_col) if bs_col else None
        curr_debt  = _safe(bs, "Current Debt", bs_col)                if bs_col else None
        total_debt = _safe(bs, _find_row(bs, ["Total Debt"]) or "Total Debt", bs_col) if bs_col else None
        total_assets = _safe(bs, "Total Assets", bs_col)              if bs_col else None
        pref_stock   = _safe(bs, _find_row(bs, ["Preferred Stock"]) or "Preferred Stock", bs_col) if bs_col else None
        treasury     = _safe(bs, _find_row(bs, ["Treasury Stock"]) or "Treasury Stock", bs_col)   if bs_col else None

        cf_col = _col_at(cf, i)
        capex  = _safe(cf, _find_row(cf, ["Capital Expenditure"]) or "Capital Expenditure", cf_col) if cf_col else None
        ni_cf  = _safe(cf, _find_row(cf, ["Net Income From Continuing Operations", "Net Income"]) or "NI", cf_col) if cf_col else None

        eps_growth = None
        if i + 1 < len(eps_vals) and eps_vals[i] and eps_vals[i + 1]:
            eg = eps_vals[i] / eps_vals[i + 1]
            eps_growth = _safe_round(eg, 4) if math.isfinite(eg) else None

        cash_ratio = None
        if cash and curr_debt and curr_debt != 0:
            cr = cash / curr_debt
            cash_ratio = _safe_round(cr) if math.isfinite(cr) else None

        equity = (total_assets - total_debt) if total_assets and total_debt else None

        results.append({
            "period":                  str(col)[:10],
            "gross_margin":            _pct(_ratio(gp, rev)),
            "sga_margin":              _pct(_ratio(sga, gp)),
            "rnd_margin":              _pct(_ratio(rnd, gp)),
            "depreciation_margin":     _pct(_ratio(dep, gp)),
            "interest_expense_margin": _pct(_ratio(iexp, oinc)),
            "income_tax_rate":         _pct(_ratio(tax, pre)),
            "net_margin":              _pct(_ratio(ni, rev)),
            "eps_growth":              eps_growth,
            "cash_gt_debt":            cash_ratio,
            "adj_debt_to_equity":      _ratio(total_debt, equity),
            "has_preferred_stock":     pref_stock is not None and pref_stock != 0,
            "has_treasury_stock":      treasury is not None and treasury != 0,
            "capex_margin":            _pct(_ratio(-capex if capex else None, ni_cf)),
        })

    return {
        "symbol": info.get("symbol", ""),
        "name": info.get("longName", ""),
        "ratios": results,
    }


# ── DCF Valuation ────────────────────────────────────────────────────────

def compute_dcf(
    current_price: float,
    shares: float,
    market_cap: float,
    pe: float,
    beta: float,
    cagr: float,
) -> dict:
    """Ten-year DCF with terminal value."""
    growth_rate = max(min(cagr, 0.30), 0.01)
    if pe <= 0 or pe > 200:
        pe = 20.0
    fcf = (market_cap / pe) * 0.75
    if fcf <= 0:
        raise ValueError("Could not estimate a positive FCF")

    beta = max(min(beta, 3.0), 0.3)
    wacc = 0.045 + beta * 0.055
    terminal_g = 0.025

    pv_fcfs = []
    for yr in range(1, 11):
        pv = fcf * ((1 + growth_rate) ** yr) / ((1 + wacc) ** yr)
        pv_fcfs.append(round(pv, 0))

    fcf_yr10 = fcf * ((1 + growth_rate) ** 10)
    terminal_val = fcf_yr10 * (1 + terminal_g) / (wacc - terminal_g)
    pv_terminal = terminal_val / ((1 + wacc) ** 10)

    intrinsic_total = sum(pv_fcfs) + pv_terminal
    intrinsic_per_share = intrinsic_total / shares

    mos = None
    if intrinsic_per_share > 0 and current_price > 0:
        m = (intrinsic_per_share - current_price) / intrinsic_per_share * 100
        mos = _safe_round(m, 1)

    verdict = (
        "Undervalued"   if mos and mos > 15  else
        "Fairly Valued" if mos and mos > -15 else
        "Overvalued"
    )

    return {
        "fcf": round(fcf, 0),
        "growth_rate_used": round(growth_rate * 100, 2),
        "wacc": round(wacc * 100, 2),
        "terminal_growth_rate": terminal_g * 100,
        "pv_fcf_by_year": pv_fcfs,
        "pv_terminal_value": round(pv_terminal, 0),
        "intrinsic_value_total": round(intrinsic_total, 0),
        "intrinsic_per_share": round(intrinsic_per_share, 2),
        "current_price": round(current_price, 2),
        "margin_of_safety_pct": mos,
        "verdict": verdict,
    }


# ── Altman Z-Score ───────────────────────────────────────────────────────

def compute_altman_z(fin: pd.DataFrame, bs: pd.DataFrame, market_cap: float) -> dict:
    total_assets   = _safe(bs, _find_row(bs, ["Total Assets"]) or "Total Assets")
    total_liab     = _safe(bs, _find_row(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"]) or "Total Liabilities")
    current_assets = _safe(bs, _find_row(bs, ["Current Assets"]) or "Current Assets")
    current_liab   = _safe(bs, _find_row(bs, ["Current Liabilities"]) or "Current Liabilities")
    retained       = _safe(bs, _find_row(bs, ["Retained Earnings", "Retained Earnings Accumulated Deficit"]) or "Retained Earnings")
    ebit           = _safe(fin, _find_row(fin, ["EBIT", "Operating Income"]) or "EBIT")
    revenue        = _safe(fin, _find_row(fin, ["Total Revenue", "Revenue"]) or "Total Revenue")

    missing = [n for n, v in [
        ("total_assets", total_assets), ("total_liab", total_liab),
        ("current_assets", current_assets), ("current_liab", current_liab),
        ("retained_earnings", retained), ("ebit", ebit), ("revenue", revenue),
    ] if v is None]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    wc = current_assets - current_liab
    X1 = wc / total_assets
    X2 = retained / total_assets
    X3 = ebit / total_assets
    X4 = market_cap / total_liab if total_liab else 0
    X5 = revenue / total_assets
    Z  = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    zone, desc = (
        ("Safe Zone",     "Low bankruptcy risk")             if Z > 2.99 else
        ("Grey Zone",     "Moderate risk — monitor closely") if Z > 1.81 else
        ("Distress Zone", "High bankruptcy risk")
    )
    return {
        "z_score": round(Z, 3), "zone": zone, "description": desc,
        "components": {
            "X1_working_capital_ratio":     round(X1, 4),
            "X2_retained_earnings_ratio":   round(X2, 4),
            "X3_ebit_ratio":                round(X3, 4),
            "X4_market_cap_to_liabilities": round(X4, 4),
            "X5_asset_turnover":            round(X5, 4),
        },
        "thresholds": {"safe": 2.99, "grey_lower": 1.81},
    }


# ── Piotroski F-Score ────────────────────────────────────────────────────

def compute_piotroski(fin: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame) -> dict:
    def _get(df, row, i):
        c = _col_at(df, i)
        return _safe(df, row, c) if c is not None else None

    def _get_multi(df, candidates, i):
        for row in candidates:
            v = _get(df, row, i)
            if v is not None:
                return v
        return None

    ta0 = _get(bs, "Total Assets", 0) or 1
    ta1 = _get(bs, "Total Assets", 1) or 1
    ni0 = _get(fin, "Net Income", 0)  or 0
    ni1 = _get(fin, "Net Income", 1)  or 0
    cfo = _get_multi(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0) or 0

    roa_curr  = ni0 / ta0
    roa_prior = ni1 / ta1
    accruals  = roa_curr - cfo / ta0

    ltd_curr  = _get(bs, "Long Term Debt", 0) or 0
    ltd_prior = _get(bs, "Long Term Debt", 1) or 0
    lev_curr  = ltd_curr / ta0
    lev_prior = ltd_prior / ta1

    ca0 = _get(bs, "Current Assets", 0) or 0
    cl0 = _get(bs, "Current Liabilities", 0) or 1
    ca1 = _get(bs, "Current Assets", 1) or 0
    cl1 = _get(bs, "Current Liabilities", 1) or 1

    sh_c = _get(bs, "Ordinary Shares Number", 0) or 0
    sh_p = _get(bs, "Ordinary Shares Number", 1) or 0

    rev0 = _get_multi(fin, ["Total Revenue", "Revenue"], 0) or 1
    rev1 = _get_multi(fin, ["Total Revenue", "Revenue"], 1) or 1
    gp0  = _get(fin, "Gross Profit", 0) or 0
    gp1  = _get(fin, "Gross Profit", 1) or 0
    gm_c, gm_p = gp0 / rev0, gp1 / rev1
    at_c, at_p = rev0 / ta0, rev1 / ta1

    signals = {
        "F1_positive_roa":           (1 if roa_curr > 0         else 0, "ROA > 0",                 "Profitability"),
        "F2_positive_cfo":           (1 if cfo > 0              else 0, "CFO > 0",                 "Profitability"),
        "F3_roa_improvement":        (1 if roa_curr > roa_prior else 0, "ROA increased YoY",       "Profitability"),
        "F4_accruals":               (1 if accruals < 0         else 0, "CFO/Assets > ROA",        "Profitability"),
        "F5_leverage_decrease":      (1 if lev_curr < lev_prior else 0, "Debt ratio decreased",    "Leverage"),
        "F6_liquidity_increase":     (1 if ca0/cl0 > ca1/cl1    else 0, "Current ratio improved",  "Leverage"),
        "F7_no_dilution":            (1 if sh_c <= sh_p         else 0, "No new shares issued",    "Leverage"),
        "F8_gross_margin_improve":   (1 if gm_c > gm_p         else 0, "Gross margin improved",   "Efficiency"),
        "F9_asset_turnover_improve": (1 if at_c > at_p         else 0, "Asset turnover improved",  "Efficiency"),
    }
    total = sum(v[0] for v in signals.values())
    strength = "Strong" if total >= 8 else "Moderate" if total >= 5 else "Weak"

    return {
        "f_score": total,
        "strength": strength,
        "signals": {k: {"score": v[0], "label": v[1], "category": v[2]} for k, v in signals.items()},
    }


# ── Technical Analysis ───────────────────────────────────────────────────

def compute_technical(hist: pd.DataFrame) -> dict:
    close = hist["Close"].squeeze()
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi_s = 100 - 100 / (1 + rs)
    rsi   = float(rsi_s.iloc[-1])
    if not math.isfinite(rsi):
        rsi = 50.0

    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    price = float(close.iloc[-1])

    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_up = float((ma20 + 2 * std20).iloc[-1])
    bb_lo = float((ma20 - 2 * std20).iloc[-1])
    bb_rng = bb_up - bb_lo
    bb_pct = ((price - bb_lo) / bb_rng * 100) if bb_rng != 0 else 50.0
    if not math.isfinite(bb_pct):
        bb_pct = 50.0

    recent = close.tail(90).resample("W").last().dropna()
    price_series = [
        {"date": str(d.date()), "price": round(float(v), 2)}
        for d, v in recent.items()
    ]

    return {
        "price": round(price, 2),
        "rsi_14": round(rsi, 1),
        "rsi_signal": "Overbought (RSI > 70)" if rsi > 70 else "Oversold (RSI < 30)" if rsi < 30 else "Neutral",
        "ma_50": round(ma50, 2),
        "ma_200": round(ma200, 2),
        "ma_signal": "Golden Cross (Bullish)" if ma50 > ma200 else "Death Cross (Bearish)",
        "bb_upper": round(bb_up, 2),
        "bb_lower": round(bb_lo, 2),
        "bb_pct": round(bb_pct, 1),
        "bb_signal": "Near Upper Band (Overbought)" if bb_pct > 80 else "Near Lower Band (Oversold)" if bb_pct < 20 else "Within Bands (Neutral)",
        "high_52w": round(float(close.max()), 2),
        "low_52w": round(float(close.min()), 2),
        "price_series": price_series,
    }


# ── Monte Carlo ──────────────────────────────────────────────────────────

def compute_monte_carlo(hist: pd.DataFrame, simulations: int = 1000) -> dict:
    close   = hist["Close"].squeeze()
    returns = np.log(close / close.shift(1)).dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    mu    = float(returns.mean())
    sigma = float(returns.std())
    S0    = float(close.iloc[-1])
    T     = 252

    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma == 0:
        raise ValueError("Could not compute valid return statistics")

    rng   = np.random.default_rng(42)
    rand  = rng.standard_normal((simulations, T))
    paths = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) + sigma * rand, axis=1))
    paths = np.where(np.isfinite(paths), paths, S0)

    final   = paths[:, -1]
    pct_pos = float((final > S0).mean() * 100)
    pcts    = {p: np.percentile(paths, p, axis=0) for p in [10, 25, 50, 75, 90]}

    steps = list(range(0, T, 5)) + [T - 1]
    chart_data = []
    for d in steps:
        row = {"day": d}
        for p in [10, 25, 50, 75, 90]:
            v = float(pcts[p][d])
            row[f"p{p}"] = _safe_round(v) if math.isfinite(v) else None
        chart_data.append(row)

    return {
        "current_price": round(S0, 2),
        "simulations": simulations,
        "annual_drift_pct": round(mu * 252 * 100, 2),
        "annual_vol_pct": round(sigma * math.sqrt(252) * 100, 2),
        "pct_paths_positive": round(pct_pos, 1),
        "forecast": {
            f"p{p}": round(float(np.percentile(final, p)), 2)
            for p in [10, 25, 50, 75, 90]
        },
        "chart_data": chart_data,
    }


# ── Peer Clustering ─────────────────────────────────────────────────────

SECTOR_PEERS = {
    "Technology":         ["AAPL","MSFT","GOOGL","META","NVDA","INTC","AMD","CRM","ORCL","ADBE"],
    "Financial Services": ["JPM","BAC","WFC","GS","MS","C","BLK","AXP","USB","PNC"],
    "Healthcare":         ["JNJ","UNH","PFE","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN"],
    "Consumer Defensive": ["KO","PEP","PG","WMT","COST","MCD","PM","MO","CL","KHC"],
    "Consumer Cyclical":  ["AMZN","TSLA","HD","NKE","SBUX","TGT","LOW","GM","F","BKNG"],
    "Energy":             ["XOM","CVX","COP","SLB","EOG","PXD","MPC","VLO","PSX","OXY"],
    "Industrials":        ["BA","CAT","GE","HON","MMM","UPS","RTX","LMT","DE","EMR"],
    "Communication":      ["GOOGL","META","DIS","NFLX","CMCSA","T","VZ","ATVI","EA","TTWO"],
    "Real Estate":        ["AMT","PLD","CCI","EQIX","PSA","DLR","O","SPG","WELL","AVB"],
    "Utilities":          ["NEE","DUK","SO","D","EXC","AEP","XEL","ED","ES","PPL"],
}


# ── DataFrame serializer ────────────────────────────────────────────────

def serialize_statement(df: pd.DataFrame) -> dict:
    out = {}
    for col in df.columns:
        label = str(col)[:10]
        out[label] = {}
        for row in df.index:
            val = df.loc[row, col]
            if pd.isna(val):
                out[label][row] = None
            elif isinstance(val, (int, float, np.integer, np.floating)):
                v = float(val)
                out[label][row] = None if not math.isfinite(v) else round(v, 4)
            else:
                out[label][row] = str(val)
    return out


# ── Internal helpers ─────────────────────────────────────────────────────

def _col_at(df: pd.DataFrame, i: int):
    """Safely get column at index i."""
    if df is None or df.empty or i >= len(df.columns):
        return df.columns[0] if df is not None and not df.empty else None
    return df.columns[i]