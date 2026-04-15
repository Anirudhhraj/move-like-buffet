"""
Thread-safe TTL cache for Yahoo Finance data.

One yf.Ticker call per symbol per TTL window. All downstream
endpoints read from cache instead of hammering Yahoo's API.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from config import YF_CACHE_TTL_SECONDS, YF_MAX_RETRIES

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: int = YF_CACHE_TTL_SECONDS

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.timestamp) < self.ttl


class YFCache:
    """
    Caches three categories of data per symbol:
      - info:       ticker.info dict
      - statements: (financials, balance_sheet, cashflow) DataFrames
      - history:    price history DataFrame (keyed by period)

    Thread-safe via a simple lock. Entries expire after TTL seconds.
    """

    def __init__(self, ttl: int = YF_CACHE_TTL_SECONDS):
        self.ttl = ttl
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry and entry.is_fresh:
                return entry.data
            return None

    def put(self, key: str, data: Any) -> None:
        with self._lock:
            self._store[key] = CacheEntry(data=data, timestamp=time.time(), ttl=self.ttl)

    def clear(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol:
                prefix = f"{symbol.upper()}:"
                self._store = {
                    k: v for k, v in self._store.items()
                    if not k.startswith(prefix)
                }
            else:
                self._store.clear()


# ── Singleton cache instance ─────────────────────────────────────────────
_cache = YFCache()


def get_ticker_info(symbol: str) -> dict:
    """Fetch ticker.info with caching and retry."""
    key = f"{symbol.upper()}:info"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    data = _fetch_with_retry(lambda: yf.Ticker(symbol.upper()).info)
    _cache.put(key, data)
    return data


def get_statements(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch (financials, balance_sheet, cashflow) with caching and retry."""
    key = f"{symbol.upper()}:stmts"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    def _fetch():
        t = yf.Ticker(symbol.upper())
        fin = t.financials
        bs = t.balancesheet
        cf = t.cashflow
        if fin is None or fin.empty:
            raise ValueError(f"No financial data for {symbol}")
        return fin, bs, cf

    data = _fetch_with_retry(_fetch)
    _cache.put(key, data)
    return data


def get_fast_info(symbol: str) -> dict:
    """Fetch fast_info with caching."""
    key = f"{symbol.upper()}:fast"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    fi = yf.Ticker(symbol.upper()).fast_info
    data = {
        "last_price": getattr(fi, "last_price", None),
        "previous_close": getattr(fi, "previous_close", None),
        "shares": getattr(fi, "shares", None),
        "market_cap": getattr(fi, "market_cap", None),
    }
    _cache.put(key, data)
    return data


def get_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch price history with caching."""
    key = f"{symbol.upper()}:hist:{period}:{interval}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    data = _fetch_with_retry(
        lambda: yf.download(
            symbol.upper(), period=period, interval=interval,
            auto_adjust=True, progress=False,
        )
    )
    _cache.put(key, data)
    return data


def get_quotes(symbols: list[str]) -> list[dict]:
    """Bulk quote fetch for the ticker bar. Cached per-symbol."""
    results = []
    uncached = []

    for sym in symbols:
        key = f"{sym.upper()}:quote"
        cached = _cache.get(key)
        if cached is not None:
            results.append(cached)
        else:
            uncached.append(sym)

    if uncached:
        try:
            tickers = yf.Tickers(" ".join(uncached))
            for sym in uncached:
                try:
                    fi = tickers.tickers[sym].fast_info
                    price = round(float(fi.last_price), 2) if fi.last_price else None
                    prev = fi.previous_close
                    chg = round(((price - prev) / prev) * 100, 2) if price and prev else None
                    quote = {
                        "symbol": sym,
                        "regularMarketPrice": price,
                        "regularMarketChangePercent": chg,
                    }
                except Exception:
                    quote = {
                        "symbol": sym,
                        "regularMarketPrice": None,
                        "regularMarketChangePercent": None,
                    }
                _cache.put(f"{sym.upper()}:quote", quote)
                results.append(quote)
        except Exception as exc:
            logger.warning("Bulk quote fetch failed: %s", exc)
            for sym in uncached:
                results.append({
                    "symbol": sym,
                    "regularMarketPrice": None,
                    "regularMarketChangePercent": None,
                })

    return results


def clear_cache(symbol: Optional[str] = None):
    _cache.clear(symbol)


# ── Retry logic ──────────────────────────────────────────────────────────

def _fetch_with_retry(fn, max_retries: int = YF_MAX_RETRIES):
    """Exponential backoff for Yahoo Finance requests."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "YF fetch failed (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, max_retries, wait, exc,
            )
            time.sleep(wait)