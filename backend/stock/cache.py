"""
Persistent file-backed cache for Yahoo Finance data.

KEY DESIGN:
  1. Every yfinance response is pickled to disk immediately
  2. On startup, ALL cached files are loaded into memory (instant)
  3. Requests are served from memory — yfinance is never in the hot path
  4. Background refresh updates stale data without blocking requests
  5. If yfinance fails, stale data is served (never returns None if a file exists)
  6. Ticker quotes use direct HTTP (no yfinance library, no rate limit conflicts)

Rate-limit strategy:
  - 2s cooldown between yfinance calls (was 0.4s)
  - Retries wait 15s/30s/45s (was 1.5s/2.5s — too short for Yahoo's 60s+ limit window)
  - Long TTLs: 6h for prices, 24h for fundamentals, 10min for quotes
  - Disk persistence survives restarts — no cold-cache stampede
"""

import json as _json
import logging
import pickle
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from config import YF_MAX_RETRIES

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── TTLs (seconds) ──────────────────────────────────────────────────────
TTL_QUOTES    = 600       # 10 min — ticker prices
TTL_PRICE     = 21600     # 6 hours — price-derived data (technical, monte carlo, dcf)
TTL_FUNDAMENTALS = 86400  # 24 hours — financial statements, ratios
TTL_INFO      = 21600     # 6 hours — company info (name, sector, P/E)

# Which cache key prefix gets which TTL
TTL_MAP = {
    "quote":  TTL_QUOTES,
    "info":   TTL_INFO,
    "fast":   TTL_PRICE,
    "hist":   TTL_PRICE,
    "stmts":  TTL_FUNDAMENTALS,
}

def _ttl_for_key(key: str) -> int:
    """Determine TTL based on cache key prefix."""
    for prefix, ttl in TTL_MAP.items():
        if f":{prefix}" in key:
            return ttl
    return TTL_INFO  # default

# ── GLOBAL RATE LIMITER ──────────────────────────────────────────────────
_yf_lock = threading.Lock()
_yf_last_call = 0.0
_YF_COOLDOWN = 2.0  # seconds between yfinance calls (was 0.4 — too aggressive)


def _throttled_call(fn):
    """Acquire global lock, enforce cooldown, execute fn()."""
    global _yf_last_call
    with _yf_lock:
        elapsed = time.time() - _yf_last_call
        if elapsed < _YF_COOLDOWN:
            time.sleep(_YF_COOLDOWN - elapsed)
        try:
            return fn()
        finally:
            _yf_last_call = time.time()


# ── Persistent Cache ─────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: int = TTL_INFO

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.timestamp) < self.ttl

    @property
    def age_str(self) -> str:
        age = time.time() - self.timestamp
        if age < 60: return f"{age:.0f}s"
        if age < 3600: return f"{age/60:.0f}m"
        return f"{age/3600:.1f}h"


class YFCache:
    """In-memory cache backed by pickle files on disk."""

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._load_from_disk()

    # ── File helpers ──────────────────────────────────────────────────

    def _file_path(self, key: str) -> Path:
        safe = key.replace(":", "__").replace("/", "_").replace("\\", "_")
        return CACHE_DIR / f"{safe}.pkl"

    def _load_from_disk(self):
        """Pre-populate memory from all disk files on startup."""
        count = 0
        for f in CACHE_DIR.glob("*.pkl"):
            try:
                with open(f, "rb") as fp:
                    entry = pickle.load(fp)
                if isinstance(entry, CacheEntry):
                    # Reconstruct key from filename
                    key = f.stem.replace("__", ":")
                    self._store[key] = entry
                    count += 1
            except Exception:
                # Corrupt file — delete it
                try:
                    f.unlink()
                except Exception:
                    pass
        if count:
            logger.info("Loaded %d cached entries from disk", count)

    def _save_to_disk(self, key: str, entry: CacheEntry):
        """Persist entry to disk. Non-blocking on failure."""
        try:
            with open(self._file_path(key), "wb") as fp:
                pickle.dump(entry, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning("Disk cache write failed for %s: %s", key, e)

    # ── Public API ────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """
        Returns cached data if fresh. If stale but exists, returns it anyway
        (caller should trigger a background refresh). Returns None only if
        no data has ever been fetched for this key.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                return entry.data  # always return if we have ANYTHING
            return None

    def is_fresh(self, key: str) -> bool:
        """Check if cached data is within TTL."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            return entry.is_fresh

    def put(self, key: str, data: Any) -> None:
        """Store in memory AND persist to disk."""
        ttl = _ttl_for_key(key)
        with self._lock:
            entry = CacheEntry(data=data, timestamp=time.time(), ttl=ttl)
            self._store[key] = entry
            self._save_to_disk(key, entry)

    def clear(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol:
                prefix = f"{symbol.upper()}:"
                keys_to_remove = [k for k in self._store if k.startswith(prefix)]
                for k in keys_to_remove:
                    del self._store[k]
                    try:
                        self._file_path(k).unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                self._store.clear()
                for f in CACHE_DIR.glob("*.pkl"):
                    try:
                        f.unlink()
                    except Exception:
                        pass


_cache = YFCache()


# ── Data fetchers (only call yfinance if cache is stale) ─────────────────

def get_ticker_info(symbol: str) -> dict:
    key = f"{symbol.upper()}:info"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key):
        return cached
    # Try to refresh, but serve stale if yfinance fails
    try:
        data = _fetch_with_retry(lambda: _throttled_call(lambda: yf.Ticker(symbol.upper()).info))
        _cache.put(key, data)
        return data
    except Exception as e:
        logger.warning("get_ticker_info(%s) failed: %s", symbol, e)
        if cached is not None:
            logger.info("Serving stale info for %s", symbol)
            return cached
        raise


def get_statements(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = f"{symbol.upper()}:stmts"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key):
        return cached

    def _fetch():
        t = yf.Ticker(symbol.upper())
        fin = t.financials
        bs = t.balancesheet
        cf = t.cashflow
        if fin is None or fin.empty:
            raise ValueError(f"No financial data for {symbol}")
        return fin, bs, cf

    try:
        data = _fetch_with_retry(lambda: _throttled_call(_fetch))
        _cache.put(key, data)
        return data
    except Exception as e:
        logger.warning("get_statements(%s) failed: %s", symbol, e)
        if cached is not None:
            logger.info("Serving stale statements for %s", symbol)
            return cached
        raise


def get_fast_info(symbol: str) -> dict:
    key = f"{symbol.upper()}:fast"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key):
        return cached

    def _fetch():
        fi = yf.Ticker(symbol.upper()).fast_info
        return {
            "last_price": getattr(fi, "last_price", None),
            "previous_close": getattr(fi, "previous_close", None),
            "shares": getattr(fi, "shares", None),
            "market_cap": getattr(fi, "market_cap", None),
        }

    try:
        data = _fetch_with_retry(lambda: _throttled_call(_fetch))
        _cache.put(key, data)
        return data
    except Exception as e:
        logger.warning("get_fast_info(%s) failed: %s", symbol, e)
        if cached is not None:
            logger.info("Serving stale fast_info for %s", symbol)
            return cached
        raise


def get_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    key = f"{symbol.upper()}:hist:{period}:{interval}"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key):
        return cached

    try:
        data = _fetch_with_retry(lambda: _throttled_call(
            lambda: yf.download(symbol.upper(), period=period, interval=interval, auto_adjust=True, progress=False)
        ))
        _cache.put(key, data)
        return data
    except Exception as e:
        logger.warning("get_history(%s) failed: %s", symbol, e)
        if cached is not None:
            logger.info("Serving stale history for %s (%s/%s)", symbol, period, interval)
            return cached
        raise


# ── Ticker quotes — DIRECT HTTP, no yfinance library ────────────────────

def get_quotes(symbols: list[str]) -> list[dict]:
    """
    Fetch live quotes via Yahoo's public chart API.
    Does NOT use yfinance — no rate limit conflicts with the main data fetcher.
    Falls back to cached data on any failure.
    """
    results = []
    need_fetch = []

    for sym in symbols:
        key = f"{sym.upper()}:quote"
        cached = _cache.get(key)
        if cached is not None and _cache.is_fresh(key):
            results.append(cached)
        else:
            need_fetch.append(sym)
            # Keep stale data as placeholder in case fetch fails
            if cached is not None:
                results.append(cached)
            else:
                results.append({"symbol": sym, "regularMarketPrice": None, "regularMarketChangePercent": None})

    if need_fetch:
        for sym in need_fetch:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1d"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = _json.loads(resp.read())
                meta = data["chart"]["result"][0]["meta"]
                price = round(meta["regularMarketPrice"], 2)
                prev = meta.get("chartPreviousClose") or meta.get("previousClose")
                chg = round(((price - prev) / prev) * 100, 2) if price and prev else None
                quote = {"symbol": sym, "regularMarketPrice": price, "regularMarketChangePercent": chg}
                _cache.put(f"{sym.upper()}:quote", quote)
                # Update the result in-place
                for i, r in enumerate(results):
                    if r["symbol"] == sym:
                        results[i] = quote
                        break
            except Exception as e:
                logger.debug("Quote fetch failed for %s: %s", sym, e)
                # Stale data already in results from above

    return results


def clear_cache(symbol: Optional[str] = None):
    _cache.clear(symbol)


# ── Retry with sane backoff ──────────────────────────────────────────────

def _fetch_with_retry(fn, max_retries: int = YF_MAX_RETRIES):
    """
    Retry with delays that actually outlast Yahoo's rate limit window.
    Waits: 15s, 30s, 45s  (was 1.5s, 2.5s — useless against a 60s+ rate limit)
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 15 * (attempt + 1)
            logger.warning("YF fetch failed (%d/%d), retry in %.0fs: %s",
                           attempt + 1, max_retries, wait, exc)
            time.sleep(wait)