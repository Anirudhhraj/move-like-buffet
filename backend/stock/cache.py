"""
Persistent file-backed cache for Yahoo Finance data with Twelve Data fallback.

Fallback chain:
  1. yfinance library (primary, unlimited but rate-limited)
  2. Yahoo v8 chart API (prices only, shares IP ban with yfinance)
  3. Twelve Data API (800 req/day free tier — quotes, fundamentals, history)
  4. Stale cached data (never returns None if a file exists)

Yahoo v10 quoteSummary is DEAD (returns 401). FMP free tier is DEAD (all 403).
"""

import json as _json
import logging
import pickle
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from config import (
    YF_MAX_RETRIES, YF_COOLDOWN_SECONDS, YF_RETRY_BASE_SECONDS,
    YF_RATE_LIMIT_WAIT, YF_QUOTE_DELAY_MS,
    YF_TTL_QUOTES, YF_TTL_PRICE, YF_TTL_FUNDAMENTALS, YF_TTL_INFO,
    TD_API_KEY, TD_DAILY_BUDGET, TD_RESERVE_BUDGET,
    YF_USER_AGENTS,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TTL_MAP = {
    "quote": YF_TTL_QUOTES, "info": YF_TTL_INFO,
    "fast": YF_TTL_PRICE, "hist": YF_TTL_PRICE, "stmts": YF_TTL_FUNDAMENTALS,
}

def _ttl_for_key(key: str) -> int:
    for prefix, ttl in TTL_MAP.items():
        if f":{prefix}" in key:
            return ttl
    return YF_TTL_INFO

def _random_ua() -> str:
    return random.choice(YF_USER_AGENTS)


# ══════════════════════════════════════════════════════════════════════════
# YAHOO RATE LIMITER & BAN TRACKER
# ══════════════════════════════════════════════════════════════════════════
_yf_lock = threading.Lock()
_yf_last_call = 0.0
_ban_lock = threading.Lock()
_ban_until = 0.0

def _is_banned() -> bool:
    with _ban_lock:
        return time.time() < _ban_until

def _set_ban(duration: float):
    global _ban_until
    with _ban_lock:
        _ban_until = time.time() + duration
        logger.warning("Yahoo ban — blocking yfinance for %.0fs", duration)

def _clear_ban():
    global _ban_until
    with _ban_lock:
        _ban_until = 0.0

def _throttled_call(fn):
    if _is_banned():
        raise RateLimitError("Yahoo ban active")
    global _yf_last_call
    with _yf_lock:
        elapsed = time.time() - _yf_last_call
        if elapsed < YF_COOLDOWN_SECONDS:
            time.sleep(YF_COOLDOWN_SECONDS - elapsed)
        try:
            result = fn()
            _clear_ban()
            return result
        except Exception as exc:
            if _is_rate_limit_error(exc):
                _set_ban(YF_RATE_LIMIT_WAIT)
                raise RateLimitError(f"Yahoo 429: {exc}") from exc
            raise
        finally:
            _yf_last_call = time.time()


# ══════════════════════════════════════════════════════════════════════════
# ERROR CLASSES
# ══════════════════════════════════════════════════════════════════════════
class RateLimitError(Exception): pass
class EmptyResponseError(Exception): pass
class TdBudgetExhausted(Exception): pass

def _is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc).lower()
    if hasattr(exc, 'response') and hasattr(exc.response, 'status_code'):
        if exc.response.status_code in (429, 403): return True
    if any(m in s for m in ['429','too many requests','rate limit','403','forbidden','throttle']):
        return True
    if isinstance(exc, urllib.error.HTTPError) and exc.code in (429, 403):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# RESPONSE VALIDATION
# ══════════════════════════════════════════════════════════════════════════
def _validate_info(data: Any, symbol: str) -> dict:
    if not isinstance(data, dict):
        raise EmptyResponseError(f"info for {symbol} not a dict")
    if not any(data.get(k) for k in ["longName","shortName","symbol","currentPrice","regularMarketPrice"]):
        raise EmptyResponseError(f"info for {symbol} has no core fields — keys: {list(data.keys())[:10]}")
    return data

def _validate_statements(fin, bs, cf, symbol: str):
    if fin is None or (isinstance(fin, pd.DataFrame) and fin.empty):
        raise EmptyResponseError(f"Empty income statement for {symbol}")
    if bs is None or (isinstance(bs, pd.DataFrame) and bs.empty):
        raise EmptyResponseError(f"Empty balance sheet for {symbol}")
    return fin, bs, cf

def _validate_history(data: Any, symbol: str) -> pd.DataFrame:
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        raise EmptyResponseError(f"Empty price history for {symbol}")
    if len(data) < 5:
        raise EmptyResponseError(f"History for {symbol} only {len(data)} rows")
    return data

def _validate_fast_info(data: Any, symbol: str) -> dict:
    if not isinstance(data, dict):
        raise EmptyResponseError(f"fast_info for {symbol} not a dict")
    if data.get("last_price") is None and data.get("market_cap") is None:
        raise EmptyResponseError(f"fast_info for {symbol} has no price or market cap")
    return data


# ══════════════════════════════════════════════════════════════════════════
# PERSISTENT CACHE
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: int = YF_TTL_INFO

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.timestamp) < self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    @property
    def age_str(self) -> str:
        age = self.age_seconds
        if age < 60: return f"{age:.0f}s"
        if age < 3600: return f"{age/60:.0f}m"
        return f"{age/3600:.1f}h"


class YFCache:
    def __init__(self):
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._load_from_disk()

    def _file_path(self, key: str) -> Path:
        safe = key.replace(":", "__").replace("/", "_").replace("\\", "_")
        return CACHE_DIR / f"{safe}.pkl"

    def _load_from_disk(self):
        count = 0
        for f in CACHE_DIR.glob("*.pkl"):
            try:
                with open(f, "rb") as fp:
                    entry = pickle.load(fp)
                if isinstance(entry, CacheEntry):
                    self._store[f.stem.replace("__", ":")] = entry
                    count += 1
            except Exception:
                try: f.unlink()
                except: pass
        if count:
            logger.info("Loaded %d cached entries from disk", count)

    def _save_to_disk(self, key: str, entry: CacheEntry):
        try:
            with open(self._file_path(key), "wb") as fp:
                pickle.dump(entry, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning("Disk write failed for %s: %s", key, e)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            return entry.data if entry else None

    def is_fresh(self, key: str) -> bool:
        with self._lock:
            entry = self._store.get(key)
            return entry is not None and entry.is_fresh

    def get_age(self, key: str) -> Optional[float]:
        with self._lock:
            entry = self._store.get(key)
            return entry.age_seconds if entry else None

    def put(self, key: str, data: Any) -> None:
        ttl = _ttl_for_key(key)
        with self._lock:
            entry = CacheEntry(data=data, timestamp=time.time(), ttl=ttl)
            self._store[key] = entry
            self._save_to_disk(key, entry)

    def clear(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol:
                prefix = f"{symbol.upper()}:"
                for k in [k for k in self._store if k.startswith(prefix)]:
                    del self._store[k]
                    try: self._file_path(k).unlink(missing_ok=True)
                    except: pass
            else:
                self._store.clear()
                for f in CACHE_DIR.glob("*.pkl"):
                    try: f.unlink()
                    except: pass

_cache = YFCache()


# ══════════════════════════════════════════════════════════════════════════
# TWELVE DATA BUDGET TRACKER — persisted to disk, resets daily
# ══════════════════════════════════════════════════════════════════════════
class _TdBudget:
    _FILE = CACHE_DIR / "_td_budget.json"

    def __init__(self):
        self._lock = threading.Lock()
        self._date: str = ""
        self._used: int = 0
        self._load()

    def _load(self):
        try:
            with open(self._FILE) as f:
                data = _json.load(f)
            if data.get("date") == str(date.today()):
                self._date = data["date"]
                self._used = data.get("used", 0)
            else:
                self._reset()
        except Exception:
            self._reset()

    def _reset(self):
        self._date = str(date.today())
        self._used = 0
        self._persist()

    def _persist(self):
        try:
            with open(self._FILE, "w") as f:
                _json.dump({"date": self._date, "used": self._used}, f)
        except: pass

    def spend(self, n: int = 1) -> bool:
        with self._lock:
            if self._date != str(date.today()): self._reset()
            if self._used + n > TD_DAILY_BUDGET:
                logger.warning("TD budget exhausted: %d/%d", self._used, TD_DAILY_BUDGET)
                return False
            self._used += n
            self._persist()
            return True

    def can_afford(self, n: int = 1, reserve_ok: bool = False) -> bool:
        with self._lock:
            if self._date != str(date.today()): self._reset()
            limit = TD_DAILY_BUDGET if reserve_ok else (TD_DAILY_BUDGET - TD_RESERVE_BUDGET)
            return (self._used + n) <= limit

    @property
    def remaining(self) -> int:
        with self._lock:
            if self._date != str(date.today()): self._reset()
            return max(0, TD_DAILY_BUDGET - self._used)

    @property
    def used(self) -> int:
        with self._lock:
            if self._date != str(date.today()): self._reset()
            return self._used

_td_budget = _TdBudget()


# ══════════════════════════════════════════════════════════════════════════
# TWELVE DATA API — nested response parsers
# ══════════════════════════════════════════════════════════════════════════

def _td_request(path: str, cost: int = 1) -> Any:
    """Call Twelve Data API. Budget-tracked + per-minute rate limited."""
    if not TD_API_KEY:
        raise ValueError("TD_API_KEY not configured")
    if not _td_budget.spend(cost):
        raise TdBudgetExhausted(f"TD budget exhausted ({_td_budget.used}/{TD_DAILY_BUDGET})")
    # Per-minute rate limit: 8 req/min on free tier
    _td_rate_limit()
    sep = "&" if "?" in path else "?"
    url = f"https://api.twelvedata.com/{path}{sep}apikey={TD_API_KEY}"
    req = urllib.request.Request(url, headers={"User-Agent": _random_ua()})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        # TD returns errors inside 200 responses
        if isinstance(data, dict) and data.get("code") in (400, 401, 403, 404, 429):
            msg = data.get("message", "unknown error")
            logger.warning("TD API error: %s -> %s", path, msg)
            raise ValueError(f"TD error: {msg}")
        return data
    except urllib.error.HTTPError as e:
        logger.warning("TD request failed (%d): %s", e.code, path)
        raise


# Per-minute rate limiter for Twelve Data (8 req/min free tier)
_td_minute_lock = threading.Lock()
_td_minute_calls: list[float] = []

def _td_rate_limit():
    """Block if we've made 7+ calls in the last 60 seconds."""
    with _td_minute_lock:
        now = time.time()
        _td_minute_calls[:] = [t for t in _td_minute_calls if now - t < 60]
        if len(_td_minute_calls) >= 7:  # leave 1 buffer under the 8/min limit
            wait = 60 - (now - _td_minute_calls[0]) + 0.5
            if wait > 0:
                logger.debug("TD rate limit: waiting %.1fs", wait)
                time.sleep(wait)
        _td_minute_calls.append(time.time())



# ── Twelve Data: get_fast_info (FREE — uses /quote) ──────────────────────

def _td_get_fast_info(symbol: str) -> dict:
    """1 TD call: /quote -> fast_info dict."""
    q = _td_request(f"quote?symbol={symbol}")
    price = float(q["close"]) if q.get("close") else None
    prev = float(q["previous_close"]) if q.get("previous_close") else None
    return {
        "last_price": price,
        "previous_close": prev,
        "shares": None,      # not available on TD free tier
        "market_cap": None,   # not available on TD free tier
    }


# ── Twelve Data: get_history (FREE — uses /time_series) ──────────────────

def _td_get_history(symbol: str) -> pd.DataFrame:
    """1 TD call: /time_series -> OHLCV DataFrame."""
    data = _td_request(f"time_series?symbol={symbol}&interval=1day&outputsize=365")
    vals = data.get("values", [])
    if not vals:
        raise EmptyResponseError(f"TD returned no history for {symbol}")
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df


# ══════════════════════════════════════════════════════════════════════════
# YAHOO v8 CHART — only surviving Yahoo direct endpoint
# ══════════════════════════════════════════════════════════════════════════

def _yahoo_v8_quote(symbol: str) -> dict:
    """Get price from Yahoo v8 chart. No API key needed."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
    req = urllib.request.Request(url, headers={"User-Agent": _random_ua()})
    with urllib.request.urlopen(req, timeout=8) as resp:
        raw = _json.loads(resp.read())
    meta = raw["chart"]["result"][0]["meta"]
    price = meta.get("regularMarketPrice")
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    if price is None:
        raise EmptyResponseError(f"Yahoo v8 no price for {symbol}")
    return {"price": price, "previous_close": prev}


# ══════════════════════════════════════════════════════════════════════════
# FETCH PIPELINE: yfinance → Yahoo v8 → Twelve Data → stale cache
# ══════════════════════════════════════════════════════════════════════════

def _td_available(cost: int = 1, reserve_ok: bool = False) -> bool:
    return bool(TD_API_KEY) and _td_budget.can_afford(cost, reserve_ok)


def _fetch_with_fallback(yf_fn, yahoo_v8_fn=None, td_fn=None,
                         td_cost: int = 1, td_reserve_ok: bool = False, label: str = "data"):
    last_error = None

    # Tier 1: yfinance
    if not _is_banned():
        try:
            return _fetch_with_retry(lambda: _throttled_call(yf_fn), label=f"yf:{label}")
        except RateLimitError as e:
            last_error = e
        except Exception as e:
            last_error = e
    else:
        last_error = RateLimitError("ban active")

    # Tier 2: Yahoo v8 chart (prices only)
    if yahoo_v8_fn is not None:
        try:
            return yahoo_v8_fn()
        except Exception as e:
            last_error = e

    # Tier 3: Twelve Data (budget-gated)
    if td_fn is not None and _td_available(td_cost, td_reserve_ok):
        try:
            return td_fn()
        except TdBudgetExhausted as e:
            last_error = e
        except Exception as e:
            logger.warning("TD fallback failed for %s: %s", label, e)
            last_error = e

    raise last_error or ValueError(f"All sources failed for {label}")


# ══════════════════════════════════════════════════════════════════════════
# DATA FETCHERS — public API
# ══════════════════════════════════════════════════════════════════════════

def get_ticker_info(symbol: str) -> dict:
    sym = symbol.upper()
    key = f"{sym}:info"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key): return cached
    try:
        # yfinance is the ONLY source for company info (TD profile/statistics are paywalled)
        data = _fetch_with_fallback(
            yf_fn=lambda: _validate_info(yf.Ticker(sym).info, sym),
            label=f"{sym}:info")
        _cache.put(key, data)
        return data
    except Exception:
        if cached is not None:
            logger.info("Serving stale info for %s (age: %s)", sym, _cache._store[key].age_str if key in _cache._store else "?")
            return cached
        raise


def get_statements(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sym = symbol.upper()
    key = f"{sym}:stmts"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key): return cached
    def _yf():
        t = yf.Ticker(sym)
        return _validate_statements(t.financials, t.balancesheet, t.cashflow, sym)
    try:
        # yfinance is the ONLY source for statements (TD statements are paywalled)
        data = _fetch_with_fallback(yf_fn=_yf, label=f"{sym}:stmts")
        _cache.put(key, data)
        return data
    except Exception:
        if cached is not None:
            logger.info("Serving stale statements for %s (age: %s)", sym, _cache._store[key].age_str if key in _cache._store else "?")
            return cached
        raise


def get_fast_info(symbol: str) -> dict:
    sym = symbol.upper()
    key = f"{sym}:fast"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key): return cached
    def _yf():
        fi = yf.Ticker(sym).fast_info
        return _validate_fast_info({
            "last_price": getattr(fi, "last_price", None),
            "previous_close": getattr(fi, "previous_close", None),
            "shares": getattr(fi, "shares", None),
            "market_cap": getattr(fi, "market_cap", None),
        }, sym)
    def _v8():
        q = _yahoo_v8_quote(sym)
        return _validate_fast_info({
            "last_price": q["price"], "previous_close": q["previous_close"],
            "shares": None, "market_cap": None,
        }, sym)
    def _td():
        return _validate_fast_info(_td_get_fast_info(sym), sym)
    try:
        data = _fetch_with_fallback(yf_fn=_yf, yahoo_v8_fn=_v8, td_fn=_td, td_cost=1, label=f"{sym}:fast")
        _cache.put(key, data)
        return data
    except Exception:
        if cached is not None:
            logger.info("Serving stale fast_info for %s", sym)
            return cached
        raise


def get_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    sym = symbol.upper()
    key = f"{sym}:hist:{period}:{interval}"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key): return cached
    def _yf():
        return _validate_history(
            yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False), sym)
    def _td():
        return _validate_history(_td_get_history(sym), sym)
    try:
        data = _fetch_with_fallback(yf_fn=_yf, td_fn=_td, td_cost=1, label=f"{sym}:hist")
        _cache.put(key, data)
        return data
    except Exception:
        if cached is not None:
            logger.info("Serving stale history for %s", sym)
            return cached
        raise


# ══════════════════════════════════════════════════════════════════════════
# CLUSTER-SAFE FETCHERS — no Twelve Data (too expensive for 10 peers)
# ══════════════════════════════════════════════════════════════════════════

def get_ticker_info_no_td(symbol: str) -> dict:
    sym = symbol.upper()
    key = f"{sym}:info"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key): return cached
    try:
        data = _fetch_with_fallback(
            yf_fn=lambda: _validate_info(yf.Ticker(sym).info, sym),
            td_fn=None, label=f"{sym}:info:cluster")
        _cache.put(key, data)
        return data
    except Exception:
        if cached is not None: return cached
        raise

def get_statements_no_td(symbol: str) -> tuple:
    sym = symbol.upper()
    key = f"{sym}:stmts"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key): return cached
    def _yf():
        t = yf.Ticker(sym)
        return _validate_statements(t.financials, t.balancesheet, t.cashflow, sym)
    try:
        data = _fetch_with_fallback(yf_fn=_yf, td_fn=None, label=f"{sym}:stmts:cluster")
        _cache.put(key, data)
        return data
    except Exception:
        if cached is not None: return cached
        raise

def get_fast_info_no_td(symbol: str) -> tuple:
    sym = symbol.upper()
    key = f"{sym}:fast"
    cached = _cache.get(key)
    if cached is not None and _cache.is_fresh(key): return cached
    def _yf():
        fi = yf.Ticker(sym).fast_info
        return _validate_fast_info({
            "last_price": getattr(fi, "last_price", None),
            "previous_close": getattr(fi, "previous_close", None),
            "shares": getattr(fi, "shares", None),
            "market_cap": getattr(fi, "market_cap", None),
        }, sym)
    def _v8():
        q = _yahoo_v8_quote(sym)
        return _validate_fast_info({
            "last_price": q["price"], "previous_close": q["previous_close"],
            "shares": None, "market_cap": None,
        }, sym)
    try:
        data = _fetch_with_fallback(yf_fn=_yf, yahoo_v8_fn=_v8, td_fn=None, label=f"{sym}:fast:cluster")
        _cache.put(key, data)
        return data
    except Exception:
        if cached is not None: return cached
        raise


# ══════════════════════════════════════════════════════════════════════════
# TICKER QUOTES — Yahoo v8 first, then Twelve Data
# ══════════════════════════════════════════════════════════════════════════

def get_quotes(symbols: list[str]) -> list[dict]:
    results, need_fetch = [], []
    for sym in symbols:
        key = f"{sym.upper()}:quote"
        cached = _cache.get(key)
        if cached is not None and _cache.is_fresh(key):
            results.append(cached)
        else:
            need_fetch.append(sym)
            results.append(cached if cached else {
                "symbol": sym, "regularMarketPrice": None, "regularMarketChangePercent": None,
            })
    if need_fetch:
        delay_s = YF_QUOTE_DELAY_MS / 1000.0
        for i, sym in enumerate(need_fetch):
            if i > 0 and delay_s > 0: time.sleep(delay_s)
            try:
                quote = _fetch_single_quote(sym)
                _cache.put(f"{sym.upper()}:quote", quote)
                for j, r in enumerate(results):
                    if r["symbol"] == sym:
                        results[j] = quote
                        break
            except Exception as e:
                logger.debug("Quote fetch failed for %s: %s", sym, e)
    return results


def _fetch_single_quote(sym: str) -> dict:
    """Yahoo v8 -> Twelve Data -> fail."""
    # Yahoo v8 chart
    try:
        q = _yahoo_v8_quote(sym)
        price = round(q["price"], 2)
        prev = q["previous_close"]
        chg = round(((price - prev) / prev) * 100, 2) if price and prev else None
        return {"symbol": sym, "regularMarketPrice": price, "regularMarketChangePercent": chg}
    except Exception:
        pass

    # Twelve Data (uses reserve budget — quotes are highest priority)
    if _td_available(1, reserve_ok=True):
        try:
            data = _td_request(f"quote?symbol={sym}")
            price = float(data["close"]) if data.get("close") else None
            chg = float(data["percent_change"]) if data.get("percent_change") else None
            return {
                "symbol": sym,
                "regularMarketPrice": round(price, 2) if price else None,
                "regularMarketChangePercent": round(chg, 2) if chg else None,
            }
        except Exception as e:
            logger.debug("TD quote failed for %s: %s", sym, e)

    raise ValueError(f"All quote sources failed for {sym}")


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def clear_cache(symbol: Optional[str] = None):
    _cache.clear(symbol)

def get_ban_status() -> dict:
    with _ban_lock:
        if time.time() < _ban_until:
            ban = {"banned": True, "remaining_seconds": round(_ban_until - time.time(), 1)}
        else:
            ban = {"banned": False, "remaining_seconds": 0}
    return {
        **ban,
        "td_used": _td_budget.used,
        "td_remaining": _td_budget.remaining,
        "td_total": TD_DAILY_BUDGET,
    }


# ══════════════════════════════════════════════════════════════════════════
# RETRY WITH CLASSIFIED BACKOFF
# ══════════════════════════════════════════════════════════════════════════

def _fetch_with_retry(fn, max_retries: int = YF_MAX_RETRIES, label: str = ""):
    for attempt in range(max_retries):
        try:
            return fn()
        except (RateLimitError, TdBudgetExhausted):
            raise
        except EmptyResponseError as exc:
            if attempt == max_retries - 1: raise
            wait = YF_RETRY_BASE_SECONDS * (attempt + 1)
            logger.warning("[%s] Empty (%d/%d), retry in %.0fs", label, attempt+1, max_retries, wait)
            time.sleep(wait)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                _set_ban(YF_RATE_LIMIT_WAIT)
                raise RateLimitError(f"Reclassified: {exc}") from exc
            if attempt == max_retries - 1: raise
            wait = YF_RETRY_BASE_SECONDS * (attempt + 1)
            logger.warning("[%s] Failed (%d/%d), retry in %.0fs", label, attempt+1, max_retries, wait)
            time.sleep(wait)