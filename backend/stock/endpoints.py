"""
FastAPI endpoints for stock analysis.
"""

import logging
import math

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from .cache import (
    get_ticker_info, get_statements, get_fast_info,
    get_history, get_quotes, clear_cache, get_ban_status,
    get_ticker_info_no_td, get_statements_no_td, get_fast_info_no_td,
)
from .analysis import (
    compute_buffett_ratios, compute_dcf, compute_altman_z,
    compute_piotroski, compute_technical, compute_monte_carlo,
    serialize_statement, SECTOR_PEERS, _safe, _find_row, _ratio,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _clean(obj):
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj

def safe_json(data: dict) -> JSONResponse:
    return JSONResponse(content=_clean(data))

def _validate_symbol(symbol: str) -> str:
    s = symbol.strip().upper()[:10]
    if not s or not all(c.isalpha() or c in "-." for c in s):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    return s


@router.get("/quotes")
def api_quotes(symbols: str = Query(..., description="Comma-separated symbols")):
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:30]
    results = get_quotes(sym_list)
    return {"quoteResponse": {"result": results}}


@router.get("/info")
def api_info(symbol: str):
    sym = _validate_symbol(symbol)
    try:
        info = get_ticker_info(sym)
        keys = [
            "longName", "sector", "industry", "country",
            "marketCap", "currentPrice", "trailingPE",
            "dividendYield", "beta", "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow", "longBusinessSummary",
        ]
        result = {k: info.get(k) for k in keys}
        if info.get("_source"):
            result["_source"] = info["_source"]
        return safe_json(result)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data source error: {exc}")


@router.get("/financials")
def api_financials(symbol: str):
    sym = _validate_symbol(symbol)
    try:
        fin, bs, cf = get_statements(sym)
        return safe_json({
            "symbol": sym,
            "income_statement": serialize_statement(fin) if not fin.empty else {},
            "balance_sheet":    serialize_statement(bs)  if not bs.empty  else {},
            "cash_flow":        serialize_statement(cf)  if not cf.empty  else {},
        })
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data source error: {exc}")


@router.get("/buffett-ratios")
def api_buffett_ratios(symbol: str):
    sym = _validate_symbol(symbol)
    try:
        fin, bs, cf = get_statements(sym)
        if fin is None or fin.empty:
            raise HTTPException(status_code=503, detail="Financial data temporarily unavailable.")
        info = get_ticker_info(sym)
        info["symbol"] = sym
        result = compute_buffett_ratios(fin, bs, cf, info)
        return safe_json(result)
    except HTTPException: raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Computation error: {exc}")


@router.get("/dcf")
def api_dcf(symbol: str):
    sym = _validate_symbol(symbol)
    try:
        fi = get_fast_info(sym)
        info = get_ticker_info(sym)
        current_price = float(fi.get("last_price") or 0)
        shares = float(fi.get("shares") or 0)
        market_cap = float(fi.get("market_cap") or 0)
        if shares == 0 and market_cap and current_price:
            shares = market_cap / current_price
        if shares == 0:
            raise ValueError("Could not determine shares outstanding")
        hist = get_history(sym, period="5y", interval="1mo")
        if hist.empty:
            raise ValueError("No price history available")
        close = hist["Close"].squeeze().dropna()
        if len(close) >= 24:
            cagr = (float(close.iloc[-1]) / float(close.iloc[-24])) ** (1 / 2) - 1
        else:
            cagr = 0.08
        pe = float(info.get("trailingPE") or 0)
        beta = float(info.get("beta") or 1.0)
        result = compute_dcf(current_price, shares, market_cap, pe, beta, cagr)
        result["symbol"] = sym
        return safe_json(result)
    except HTTPException: raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DCF error: {exc}")


@router.get("/altman-z")
def api_altman_z(symbol: str):
    sym = _validate_symbol(symbol)
    try:
        fin, bs, _ = get_statements(sym)
        info = get_ticker_info(sym)
        if fin.empty or bs.empty:
            raise HTTPException(status_code=503, detail="Financial data temporarily unavailable.")
        market_cap = float(info.get("marketCap") or 0)
        result = compute_altman_z(fin, bs, market_cap)
        result["symbol"] = sym
        return safe_json(result)
    except HTTPException: raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Altman Z error: {exc}")


@router.get("/piotroski")
def api_piotroski(symbol: str):
    sym = _validate_symbol(symbol)
    try:
        fin, bs, cf = get_statements(sym)
        if fin.empty or bs.empty:
            raise HTTPException(status_code=503, detail="Financial data temporarily unavailable.")
        result = compute_piotroski(fin, bs, cf)
        result["symbol"] = sym
        return safe_json(result)
    except HTTPException: raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Piotroski error: {exc}")


@router.get("/cluster")
def api_cluster(symbol: str):
    sym = _validate_symbol(symbol)
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        import pandas as pd

        info = get_ticker_info(sym)
        sector = info.get("sector", "Technology")
        peers = SECTOR_PEERS.get(sector, SECTOR_PEERS["Technology"])
        if sym not in peers:
            peers = [sym] + peers[:9]

        rows = {}
        for peer in peers:
            try:
                p_fin, p_bs, p_cf = get_statements_no_td(peer)
                p_info = get_ticker_info_no_td(peer)
                if p_fin.empty or p_bs.empty:
                    continue
                gp  = _safe(p_fin, _find_row(p_fin, ["Gross Profit"]) or "GP")
                rev = _safe(p_fin, _find_row(p_fin, ["Total Revenue", "Revenue"]) or "Rev")
                ni  = _safe(p_fin, _find_row(p_fin, ["Net Income"]) or "NI")
                td  = _safe(p_bs, _find_row(p_bs, ["Total Debt"]) or "TD")
                ta  = _safe(p_bs, _find_row(p_bs, ["Total Assets"]) or "TA")
                ocf = _safe(p_cf, _find_row(p_cf, ["Operating Cash Flow"]) or "OCF") if not p_cf.empty else None
                cap = _safe(p_cf, _find_row(p_cf, ["Capital Expenditure"]) or "CapEx") if not p_cf.empty else 0
                if rev is None or ta is None:
                    continue
                fi = get_fast_info_no_td(peer)
                pe = float(p_info.get("forwardPE") or p_info.get("trailingPE") or 0)
                pb = float(p_info.get("priceToBook") or 0)
                equity = (ta - td) if ta and td else None
                rows[peer] = {
                    "gross_margin": round(gp / rev * 100, 2) if gp and rev else None,
                    "net_margin": round(ni / rev * 100, 2) if ni and rev else None,
                    "roe": round(ni / equity * 100, 2) if ni and equity and equity != 0 else None,
                    "de_ratio": round(td / equity, 2) if td and equity and equity != 0 else None,
                    "fcf_margin": round(((ocf or 0) + (cap or 0)) / rev * 100, 2) if rev else None,
                    "pe_ratio": pe if pe > 0 else None,
                    "pb_ratio": pb if pb > 0 else None,
                }
            except Exception:
                continue

        if len(rows) < 3:
            raise ValueError("Not enough peers with complete data for clustering")

        df = pd.DataFrame.from_dict(rows, orient="index")
        features = ["gross_margin", "net_margin", "roe", "de_ratio", "fcf_margin", "pe_ratio", "pb_ratio"]
        df_feat = df[features].dropna()
        if len(df_feat) < 3:
            raise ValueError("Not enough peers with complete data")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_feat)
        k = min(3, len(df_feat))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)

        df_feat = df_feat.copy()
        df_feat["cluster"] = labels
        target_cluster = int(df_feat.loc[sym, "cluster"]) if sym in df_feat.index else -1
        cluster_members = {int(c): df_feat[df_feat["cluster"] == c].index.tolist() for c in range(k)}
        cluster_means = {
            int(c): {f: round(float(df_feat[df_feat["cluster"] == c][f].mean()), 2) for f in features}
            for c in range(k)
        }
        return safe_json({
            "symbol": sym, "sector": sector, "target_cluster": target_cluster,
            "cluster_members": cluster_members, "cluster_means": cluster_means,
            "features_used": features,
        })
    except HTTPException: raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Clustering error: {exc}")


@router.get("/technical")
def api_technical(symbol: str):
    sym = _validate_symbol(symbol)
    try:
        hist = get_history(sym, period="1y", interval="1d")
        if hist.empty: raise ValueError("No price history found")
        result = compute_technical(hist)
        result["symbol"] = sym
        return safe_json(result)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Technical analysis error: {exc}")


@router.get("/monte-carlo")
def api_monte_carlo(symbol: str, simulations: int = Query(1000, ge=100, le=10000)):
    sym = _validate_symbol(symbol)
    try:
        hist = get_history(sym, period="2y", interval="1d")
        if hist.empty: raise ValueError("No price data")
        result = compute_monte_carlo(hist, simulations)
        result["symbol"] = sym
        return safe_json(result)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Monte Carlo error: {exc}")


@router.post("/clear-cache")
def api_clear_cache(symbol: str = None):
    clear_cache(symbol)
    return {"status": "ok", "cleared": symbol or "all"}


@router.get("/health")
def api_health():
    from config import TD_API_KEY, TD_DAILY_BUDGET, TD_RESERVE_BUDGET
    status = get_ban_status()
    return {
        "yahoo_ban": {
            "banned": status["banned"],
            "remaining_seconds": status["remaining_seconds"],
        },
        "twelvedata": {
            "configured": bool(TD_API_KEY),
            "budget_total": TD_DAILY_BUDGET,
            "budget_used": status["td_used"],
            "budget_remaining": status["td_remaining"],
            "reserve_for_quotes": TD_RESERVE_BUDGET,
            "available_for_analysis": max(0, status["td_remaining"] - TD_RESERVE_BUDGET),
        },
        "tip": (
            "Yahoo IP ban active. App falls through to Twelve Data. "
            "Fastest fix: restart your router for a new IP. Budget resets at midnight."
        ) if status["banned"] else "All clear.",
    }