"""
Persistent file-backed cache for Yahoo Finance data with Twelve Data fallback.

═══════════════════════════════════════════════════════════════════════════
DESIGN: CACHE-FIRST
═══════════════════════════════════════════════════════════════════════════

Every public fetch function follows this strict order:

    1. FRESH CACHE HIT   → return immediately, zero network calls
    2. LIVE FETCH        → yfinance → Yahoo v8 → Twelve Data
       (cache the result on success)
    3. STALE CACHE HIT   → serve stale data rather than fail
    4. NOTHING AVAILABLE → raise the underlying error

Cache files are NEVER silently deleted. Unreadable files are logged and
left in place for human inspection.

Two on-disk pickle formats are accepted automatically:
  A) Runtime format: a CacheEntry dataclass instance
  B) Prefetch format: a plain dict with keys {"data", "timestamp", "ttl"}

Public API is unchanged: get_ticker_info, get_statements, get_fast_info,
get_history, get_quotes, clear_cache, get_ban_status, and the _no_td
cluster-safe variants.
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
# ERROR CLASSES  (kept for external compatibility)
# ══════════════════════════════════════════════════════════════════════════

class RateLimitError(Exception):
    """Raised when we detect or enter a Yahoo rate-limit state."""
    pass


class EmptyResponseError(Exception):
    """Raised when an upstream returns structurally empty data."""
    pass


class TdBudgetExhausted(Exception):
    """Raised when Twelve Data's daily budget is used up."""
    pass


def _is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc).lower()
    if hasattr(exc, 'response') and hasattr(exc.response, 'status_code'):
        if exc.response.status_code in (429, 403):
            return True
    if any(m in s for m in ['429', 'too many requests', 'rate limit',
                            '403', 'forbidden', 'throttle', 'yfratelimiterror']):
        return True
    if isinstance(exc, urllib.error.HTTPError) and exc.code in (429, 403):
        return True
    return False


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
    """Serialize Yahoo calls with cooldown; auto-enter ban on 429/403."""
    if _is_banned():
        raise RateLimitError("Yahoo ban active")
    global _yf_last_call
    with _yf_lock:
        elapsed = time.time() - _yf_last_call
        if elapsed < YF_COOLDOWN_SECONDS:
            time.sleep(YF_COOLDOWN_SECONDS - elapsed)
        try:
            result = fn()
            return result
        except Exception as exc:
            if _is_rate_limit_error(exc):
                _set_ban(YF_RATE_LIMIT_WAIT)
                raise RateLimitError(f"Yahoo 429: {exc}") from exc
            raise
        finally:
            _yf_last_call = time.time()


# ══════════════════════════════════════════════════════════════════════════
# RESPONSE VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def _validate_info(data: Any, symbol: str) -> dict:
    if not isinstance(data, dict):
        raise EmptyResponseError(f"info for {symbol} not a dict")
    if not any(data.get(k) for k in ["longName", "shortName", "symbol",
                                      "currentPrice", "regularMarketPrice"]):
        raise EmptyResponseError(
            f"info for {symbol} has no core fields — keys: {list(data.keys())[:10]}"
        )
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
# CACHE ENTRY + PERSISTENT STORE
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
        if age < 60:
            return f"{age:.0f}s"
        if age < 3600:
            return f"{age/60:.0f}m"
        if age < 86400:
            return f"{age/3600:.1f}h"
        return f"{age/86400:.1f}d"


class YFCache:
    """
    Persistent file-backed cache.

    CRITICAL: this class NEVER deletes files on load errors. A broken file
    is logged and left on disk. The only way files disappear is through
    the explicit `clear()` method.
    """

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._load_from_disk()

    # ── Path mapping ─────────────────────────────────────────────────

    def _file_path(self, key: str) -> Path:
        safe = key.replace(":", "__").replace("/", "_").replace("\\", "_")
        return CACHE_DIR / f"{safe}.pkl"

    @staticmethod
    def _key_from_filename(stem: str) -> str:
        # "GOOGL__hist__5y__1mo" → "GOOGL:hist:5y:1mo"
        return stem.replace("__", ":")

    # ── Disk I/O ─────────────────────────────────────────────────────

    def _load_from_disk(self):
        """
        Load every .pkl in CACHE_DIR. Accepts two formats:
          A) CacheEntry instance
          B) Plain dict with keys {"data", "timestamp", "ttl"}
        Broken files are logged and LEFT ON DISK (never deleted).
        """
        loaded = 0
        skipped = 0
        errors = 0
        formats = {"dataclass": 0, "dict": 0}

        for f in CACHE_DIR.glob("*.pkl"):
            # Skip any hidden/meta files
            if f.name.startswith("_"):
                continue
            try:
                with open(f, "rb") as fp:
                    raw = pickle.load(fp)

                entry: Optional[CacheEntry] = None

                # Format A: already a CacheEntry
                if isinstance(raw, CacheEntry):
                    entry = raw
                    formats["dataclass"] += 1

                # Format B: plain dict from prefetch script
                elif (isinstance(raw, dict)
                      and "data" in raw
                      and "timestamp" in raw):
                    entry = CacheEntry(
                        data=raw["data"],
                        timestamp=float(raw["timestamp"]),
                        ttl=int(raw.get("ttl", YF_TTL_INFO)),
                    )
                    formats["dict"] += 1

                else:
                    logger.warning(
                        "Skipped %s (unknown pickle format: %s) — file preserved",
                        f.name, type(raw).__name__,
                    )
                    skipped += 1
                    continue

                key = self._key_from_filename(f.stem)
                self._store[key] = entry
                loaded += 1

            except Exception as e:
                logger.warning(
                    "Failed to load %s (%s) — file preserved for inspection",
                    f.name, e,
                )
                errors += 1

        if loaded:
            logger.info(
                "Cache loaded: %d entries (%d dataclass, %d dict). "
                "Skipped: %d. Errors: %d.",
                loaded, formats["dataclass"], formats["dict"], skipped, errors,
            )
        else:
            logger.warning(
                "Cache is empty! 0 entries loaded. %d skipped, %d errored.",
                skipped, errors,
            )

    def _save_to_disk(self, key: str, entry: CacheEntry):
        """Persist a single entry. Disk failures are logged, not raised."""
        try:
            path = self._file_path(key)
            tmp = path.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as fp:
                pickle.dump(entry, fp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)  # atomic on same filesystem
        except Exception as e:
            logger.warning("Disk write failed for %s: %s", key, e)

    # ── Public in-memory API ─────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return cached data regardless of freshness. None if absent."""
        with self._lock:
            entry = self._store.get(key)
            return entry.data if entry else None

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def is_fresh(self, key: str) -> bool:
        with self._lock:
            entry = self._store.get(key)
            return entry is not None and entry.is_fresh

    def get_age(self, key: str) -> Optional[float]:
        with self._lock:
            entry = self._store.get(key)
            return entry.age_seconds if entry else None

    def get_age_str(self, key: str) -> str:
        with self._lock:
            entry = self._store.get(key)
            return entry.age_str if entry else "none"

    def put(self, key: str, data: Any) -> None:
        """Store fresh data with a TTL inferred from the key prefix."""
        ttl = _ttl_for_key(key)
        with self._lock:
            entry = CacheEntry(data=data, timestamp=time.time(), ttl=ttl)
            self._store[key] = entry
            self._save_to_disk(key, entry)

    def clear(self, symbol: Optional[str] = None) -> None:
        """Explicit purge. Only path that actually deletes disk files."""
        with self._lock:
            if symbol:
                prefix = f"{symbol.upper()}:"
                for k in [k for k in self._store if k.startswith(prefix)]:
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


# Module-level singleton
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
        except Exception:
            pass

    def spend(self, n: int = 1) -> bool:
        with self._lock:
            if self._date != str(date.today()):
                self._reset()
            if self._used + n > TD_DAILY_BUDGET:
                logger.warning("TD budget exhausted: %d/%d", self._used, TD_DAILY_BUDGET)
                return False
            self._used += n
            self._persist()
            return True

    def can_afford(self, n: int = 1, reserve_ok: bool = False) -> bool:
        with self._lock:
            if self._date != str(date.today()):
                self._reset()
            limit = TD_DAILY_BUDGET if reserve_ok else (TD_DAILY_BUDGET - TD_RESERVE_BUDGET)
            return (self._used + n) <= limit

    @property
    def remaining(self) -> int:
        with self._lock:
            if self._date != str(date.today()):
                self._reset()
            return max(0, TD_DAILY_BUDGET - self._used)

    @property
    def used(self) -> int:
        with self._lock:
            if self._date != str(date.today()):
                self._reset()
            return self._used


_td_budget = _TdBudget()


# ══════════════════════════════════════════════════════════════════════════
# TWELVE DATA API
# ══════════════════════════════════════════════════════════════════════════

_td_minute_lock = threading.Lock()
_td_minute_calls: list[float] = []


def _td_rate_limit():
    """Block if we've made 7+ calls in the last 60 seconds (free tier is 8/min)."""
    with _td_minute_lock:
        now = time.time()
        _td_minute_calls[:] = [t for t in _td_minute_calls if now - t < 60]
        if len(_td_minute_calls) >= 7:
            wait = 60 - (now - _td_minute_calls[0]) + 0.5
            if wait > 0:
                logger.debug("TD rate limit: waiting %.1fs", wait)
                time.sleep(wait)
        _td_minute_calls.append(time.time())


def _td_request(path: str, cost: int = 1) -> Any:
    if not TD_API_KEY:
        raise ValueError("TD_API_KEY not configured")
    if not _td_budget.spend(cost):
        raise TdBudgetExhausted(f"TD budget exhausted ({_td_budget.used}/{TD_DAILY_BUDGET})")
    _td_rate_limit()
    # Yahoo uses hyphens for share classes (BRK-B), Twelve Data uses dots (BRK.B)
    if "-" in path:
        import re
        path = re.sub(r'([A-Z]{1,5})-([A-Z](?=[&,\s]|$))', r'\1.\2', path)
    sep = "&" if "?" in path else "?"
    url = f"https://api.twelvedata.com/{path}{sep}apikey={TD_API_KEY}"
    req = urllib.request.Request(url, headers={"User-Agent": _random_ua()})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        if isinstance(data, dict) and data.get("code") in (400, 401, 403, 404, 429):
            msg = data.get("message", "unknown error")
            logger.warning("TD API error: %s -> %s", path, msg)
            raise ValueError(f"TD error: {msg}")
        return data
    except urllib.error.HTTPError as e:
        logger.warning("TD request failed (%d): %s", e.code, path)
        raise


def _td_get_fast_info(symbol: str) -> dict:
    q = _td_request(f"quote?symbol={symbol}")
    price = float(q["close"]) if q.get("close") else None
    prev = float(q["previous_close"]) if q.get("previous_close") else None
    return {
        "last_price": price, "previous_close": prev,
        "shares": None, "market_cap": None,
    }


def _td_get_history(symbol: str) -> pd.DataFrame:
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
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    return df


# ══════════════════════════════════════════════════════════════════════════
# YAHOO v8 CHART — prices only
# ══════════════════════════════════════════════════════════════════════════

def _yahoo_v8_quote(symbol: str) -> dict:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{symbol}?interval=1d&range=5d")
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
# CACHE-FIRST FETCH TEMPLATE
# ══════════════════════════════════════════════════════════════════════════

def _td_available(cost: int = 1, reserve_ok: bool = False) -> bool:
    return bool(TD_API_KEY) and _td_budget.can_afford(cost, reserve_ok)


def _cache_first_fetch(
    key: str,
    yf_fn,
    *,
    yahoo_v8_fn=None,
    td_fn=None,
    td_cost: int = 1,
    td_reserve_ok: bool = False,
    use_td: bool = True,
    label: str = "",
):
    """
    Unified cache-first fetch flow used by every public data function.

    Order:
      1. Fresh cache         → return (no network)
      2. yfinance (if !banned) → cache + return
      3. Yahoo v8 (if given)   → cache + return
      4. Twelve Data (if given, budget allows, use_td)  → cache + return
      5. Stale cache (if any) → log + return
      6. Re-raise the last error
    """
    # 1. Fresh cache
    if _cache.is_fresh(key):
        return _cache.get(key)

    errors = []

    # 2. yfinance
    if not _is_banned():
        try:
            data = _fetch_with_retry(lambda: _throttled_call(yf_fn), label=f"yf:{label}")
            _cache.put(key, data)
            return data
        except (RateLimitError, EmptyResponseError) as e:
            errors.append(f"yfinance: {e}")
        except Exception as e:
            errors.append(f"yfinance: {e}")
    else:
        errors.append("yfinance: ban active")

    # 3. Yahoo v8 chart
    if yahoo_v8_fn is not None:
        try:
            data = yahoo_v8_fn()
            _cache.put(key, data)
            return data
        except Exception as e:
            errors.append(f"yahoo_v8: {e}")

    # 4. Twelve Data
    if td_fn is not None and use_td and _td_available(td_cost, td_reserve_ok):
        try:
            data = td_fn()
            _cache.put(key, data)
            return data
        except TdBudgetExhausted as e:
            errors.append(f"td: {e}")
        except Exception as e:
            errors.append(f"td: {e}")

    # 5. Stale cache fallback
    if _cache.has(key):
        logger.info(
            "Serving STALE cache for %s (age: %s) — live fetch failed: %s",
            key, _cache.get_age_str(key), "; ".join(errors[-2:]),
        )
        return _cache.get(key)

    # 6. Give up
    raise RuntimeError(
        f"All sources failed for {label or key}: {' | '.join(errors)}"
    )


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════

def get_ticker_info(symbol: str) -> dict:
    sym = symbol.upper()
    key = f"{sym}:info"
    return _cache_first_fetch(
        key,
        yf_fn=lambda: _validate_info(yf.Ticker(sym).info, sym),
        # TD profile/statistics are paywalled — no TD fallback for info
        label=key,
    )


def get_statements(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sym = symbol.upper()
    key = f"{sym}:stmts"

    def _yf():
        t = yf.Ticker(sym)
        return _validate_statements(t.financials, t.balancesheet, t.cashflow, sym)

    return _cache_first_fetch(key, yf_fn=_yf, label=key)


def get_fast_info(symbol: str) -> dict:
    sym = symbol.upper()
    key = f"{sym}:fast"

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

    return _cache_first_fetch(
        key, yf_fn=_yf, yahoo_v8_fn=_v8, td_fn=_td, td_cost=1, label=key,
    )


def get_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    sym = symbol.upper()
    key = f"{sym}:hist:{period}:{interval}"

    def _yf():
        return _validate_history(
            yf.download(sym, period=period, interval=interval,
                        auto_adjust=True, progress=False),
            sym,
        )

    def _td():
        return _validate_history(_td_get_history(sym), sym)

    return _cache_first_fetch(
        key, yf_fn=_yf, td_fn=_td, td_cost=1, label=key,
    )


# ── Cluster-safe variants: no Twelve Data (10 peers × 3 calls = too expensive) ──

def get_ticker_info_no_td(symbol: str) -> dict:
    sym = symbol.upper()
    key = f"{sym}:info"
    return _cache_first_fetch(
        key,
        yf_fn=lambda: _validate_info(yf.Ticker(sym).info, sym),
        use_td=False,
        label=f"{key}:cluster",
    )


def get_statements_no_td(symbol: str) -> tuple:
    sym = symbol.upper()
    key = f"{sym}:stmts"

    def _yf():
        t = yf.Ticker(sym)
        return _validate_statements(t.financials, t.balancesheet, t.cashflow, sym)

    return _cache_first_fetch(key, yf_fn=_yf, use_td=False, label=f"{key}:cluster")


def get_fast_info_no_td(symbol: str) -> dict:
    sym = symbol.upper()
    key = f"{sym}:fast"

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

    return _cache_first_fetch(
        key, yf_fn=_yf, yahoo_v8_fn=_v8, use_td=False, label=f"{key}:cluster",
    )


# ══════════════════════════════════════════════════════════════════════════
# TICKER QUOTES
# ══════════════════════════════════════════════════════════════════════════

def get_quotes(symbols: list[str]) -> list[dict]:
    results = []
    to_fetch = []

    for sym in symbols:
        key = f"{sym.upper()}:quote"
        if _cache.is_fresh(key):
            results.append(_cache.get(key))
        else:
            # Reserve slot in results so order is preserved, fill later
            to_fetch.append(sym)
            results.append(
                _cache.get(key) if _cache.has(key) else {
                    "symbol": sym,
                    "regularMarketPrice": None,
                    "regularMarketChangePercent": None,
                }
            )

    if to_fetch:
        delay_s = YF_QUOTE_DELAY_MS / 1000.0
        for i, sym in enumerate(to_fetch):
            if i > 0 and delay_s > 0:
                time.sleep(delay_s)
            try:
                quote = _fetch_single_quote(sym)
                _cache.put(f"{sym.upper()}:quote", quote)
                for j, r in enumerate(results):
                    if r["symbol"] == sym:
                        results[j] = quote
                        break
            except Exception as e:
                logger.debug("Quote fetch failed for %s: %s", sym, e)
                # leave the stale-or-null placeholder in results

    return results


def _fetch_single_quote(sym: str) -> dict:
    """Yahoo v8 → Twelve Data → fail."""
    # Yahoo v8 chart (no rate limit on this endpoint AFAICT)
    try:
        q = _yahoo_v8_quote(sym)
        price = round(q["price"], 2)
        prev = q["previous_close"]
        chg = round(((price - prev) / prev) * 100, 2) if price and prev else None
        return {"symbol": sym, "regularMarketPrice": price,
                "regularMarketChangePercent": chg}
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
        "cached_keys": len(_cache._store),
    }


# ══════════════════════════════════════════════════════════════════════════
# RETRY WITH CLASSIFIED BACKOFF
# ══════════════════════════════════════════════════════════════════════════

def _fetch_with_retry(fn, max_retries: int = YF_MAX_RETRIES, label: str = ""):
    """
    Retry wrapper used inside _cache_first_fetch for the yfinance tier only.
    Exits early on rate-limit (so we fall through to Yahoo v8 / TD quickly).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (RateLimitError, TdBudgetExhausted):
            raise
        except EmptyResponseError as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                raise
            wait = YF_RETRY_BASE_SECONDS * (attempt + 1)
            logger.warning("[%s] Empty (%d/%d), retry in %.0fs",
                           label, attempt + 1, max_retries, wait)
            time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc):
                _set_ban(YF_RATE_LIMIT_WAIT)
                raise RateLimitError(f"Reclassified: {exc}") from exc
            if attempt == max_retries - 1:
                raise
            wait = YF_RETRY_BASE_SECONDS * (attempt + 1)
            logger.warning("[%s] Failed (%d/%d), retry in %.0fs",
                           label, attempt + 1, max_retries, wait)
            time.sleep(wait)
    if last_exc:
        raise last_exc