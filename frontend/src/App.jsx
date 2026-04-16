import { useState, useEffect, useRef, useCallback } from "react";
import ChatBot from "./ChatBot";
import Dashboard from "./Dashboard";

const API = "";
const SYMBOLS = ["BRK-B","AAPL","TSLA","MSFT","GOOGL","AMZN","NVDA","META","AMD","NFLX"];

/* ── Ticker ─────────────────────────────────────────────────────────── */

function TickerBar() {
  const [prices, setPrices] = useState({});

  useEffect(() => {
    // Quotes now use direct HTTP (not yfinance) — safe to fetch early
    const t = setTimeout(fetchPrices, 8000);
    const id = setInterval(fetchPrices, 60000);
    return () => { clearTimeout(t); clearInterval(id); };
  }, []);

  async function fetchPrices() {
    try {
      const r = await fetch(`${API}/api/quotes?symbols=${SYMBOLS.join(",")}`);
      if (!r.ok) return;
      const d = await r.json();
      const q = d?.quoteResponse?.result ?? [];
      const u = {};
      for (const x of q) {
        u[x.symbol] = {
          price: x.regularMarketPrice?.toFixed(2) ?? "—",
          chg: x.regularMarketChangePercent != null
            ? (x.regularMarketChangePercent >= 0 ? "+" : "") + x.regularMarketChangePercent.toFixed(2) + "%"
            : "",
          up: (x.regularMarketChangePercent ?? 0) >= 0,
        };
      }
      setPrices(u);
    } catch {}
  }

  const items = SYMBOLS.map(s => ({
    sym: s.replace("-", "."),
    val: prices[s]?.price ?? "···",
    chg: prices[s]?.chg ?? "",
    up: prices[s]?.up ?? true,
  }));
  const track = [...items, ...items, ...items];

  return (
    <div className="tk-bar">
      <div className="tk-track">
        {track.map((t, i) => (
          <span key={i} className="tk-item">
            <span className="tk-sym">{t.sym}</span>
            <span className="tk-price">{t.val}</span>
            {t.chg && <span className={`tk-chg ${t.up ? "tk-up" : "tk-dn"}`}>{t.chg}</span>}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ── Resizable chat panel ────────────────────────────────────────────── */

function ChatPanel({ visible, onClose }) {
  const [width, setWidth] = useState(420);
  const drag = useRef(null);

  const onPointerDown = useCallback((e) => {
    e.preventDefault();
    drag.current = { startX: e.clientX, startW: width };
    const onMove = (ev) => {
      if (!drag.current) return;
      setWidth(Math.max(340, Math.min(700, drag.current.startW + (drag.current.startX - ev.clientX))));
    };
    const onUp = () => { drag.current = null; window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp); };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, [width]);

  return (
    <aside className={`chat-panel ${visible ? "chat-panel--open" : ""}`} style={visible ? { width, minWidth: width } : undefined}>
      <div className="cp-resize" onPointerDown={onPointerDown} />
      <div className="cp-inner">
        <div className="cp-head">
          <div className="cp-head-l">
            <div className="brand brand--sm">B<sup>2</sup></div>
            <span className="cp-head-title">Ask the Oracle</span>
          </div>
          <button className="cp-close" onClick={onClose}><svg width="10" height="10" viewBox="0 0 12 12" fill="none"><path d="M1 1l10 10M11 1 1 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg></button>
        </div>
        <div className="cp-body"><ChatBot /></div>
      </div>
    </aside>
  );
}

/* ── App ────────────────────────────────────────────────────────────── */

export default function App() {
  const [chatOpen, setChatOpen] = useState(false);
  const [time, setTime] = useState(new Date());

  useEffect(() => { const id = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(id); }, []);

  const ts = time.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  const ds = time.toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" }).toUpperCase();

  return (
    <div className="app">
      {/* Fixed header — not sticky, just flex-shrink: 0 inside a locked viewport */}
      <header className="hdr">
        <div className="hdr-l">
          <div className="brand">B<sup>2</sup></div>
          <div className="hdr-title-group">
            <span className="hdr-t">BUFFETT BUREAU</span>
            <span className="hdr-sub">Value Intelligence Terminal</span>
          </div>
        </div>
        <div className="hdr-r">
          <span className="hdr-clock">{ds}</span>
          <div className="hdr-sep" />
          <span className="hdr-clock hdr-time">{ts}</span>
        </div>
      </header>

      <TickerBar />

      {/* Workspace fills remaining viewport height — overflow hidden isolates scroll contexts */}
      <div className="workspace">
        <main className="main"><Dashboard /></main>
        <ChatPanel visible={chatOpen} onClose={() => setChatOpen(false)} />
      </div>

      {/* FAB — hidden when chat is open */}
      {!chatOpen && (
        <button className="fab" onClick={() => setChatOpen(true)} title="Ask the Oracle">
          <span className="fab-b">B<sup>2</sup></span>
        </button>
      )}

      <style>{CSS}</style>
    </div>
  );
}

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg-0:#080b12; --bg-1:#0d1117; --bg-2:#141a24; --bg-3:#1b2332; --bg-4:#243044;
  --bd:#1e2a3a; --bd-l:#2d3f56;
  --tx-1:#e4ddd0; --tx-2:#9b927f; --tx-3:#6b6358;
  --accent:#c9a84c; --accent-l:#dfc06a;
  --accent-bg:rgba(201,168,76,.07); --accent-bd:rgba(201,168,76,.22);
  --ok:#4a9e6d; --ok-bg:rgba(74,158,109,.08); --ok-bd:rgba(74,158,109,.22);
  --err:#c4463a; --err-bg:rgba(196,70,58,.08); --err-bd:rgba(196,70,58,.22);
  --warn:#c9a84c; --warn-bg:rgba(201,168,76,.07); --warn-bd:rgba(201,168,76,.22);
  --blue:#5b8db8; --blue-bg:rgba(91,141,184,.08); --blue-bd:rgba(91,141,184,.22);
  --serif:'Libre Baskerville',Georgia,'Times New Roman',serif;
  --sans:'DM Sans',system-ui,-apple-system,sans-serif;
  --mono:'JetBrains Mono','SF Mono',Consolas,monospace;
  --ease:cubic-bezier(.16,1,.3,1);
}

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{font-family:var(--sans);background:var(--bg-0);color:var(--tx-1);-webkit-font-smoothing:antialiased}
::selection{background:rgba(201,168,76,.3);color:#fff}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:var(--bg-0)}
::-webkit-scrollbar-thumb{background:var(--bd-l)}

/* ===== CRITICAL: app is EXACTLY viewport height, body never scrolls ===== */
.app{height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* Header + ticker are flex-shrink:0 — they take their natural height */
.hdr{
  background:var(--bg-1);border-bottom:1px solid var(--bd);
  height:54px;display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;flex-shrink:0;z-index:50;
}
.hdr::after{content:'';position:absolute;bottom:-1px;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--accent-bd),transparent)}
.hdr-l{display:flex;align-items:center;gap:14px}
.hdr-r{display:flex;align-items:center;gap:14px}
.hdr-title-group{display:flex;flex-direction:column}
.hdr-t{font-family:var(--serif);font-size:14px;font-weight:700;color:var(--tx-1);letter-spacing:2px;line-height:1.2}
.hdr-sub{font-size:9px;color:var(--tx-3);letter-spacing:1.5px;text-transform:uppercase;font-weight:500}
.hdr-clock{font-size:11px;color:var(--tx-3);letter-spacing:.5px;font-family:var(--mono);font-weight:500}
.hdr-time{color:var(--accent)}
.hdr-sep{width:1px;height:24px;background:var(--bd)}

.brand{background:var(--accent);color:var(--bg-0);font-family:var(--serif);font-weight:700;font-size:14px;padding:5px 9px 4px;line-height:1;letter-spacing:.3px}
.brand sup{font-size:9px}
.brand--sm{font-size:11px;padding:3px 7px 2px}

/* Ticker */
.tk-bar{background:var(--bg-0);border-bottom:1px solid var(--bd);height:30px;overflow:hidden;display:flex;align-items:center;flex-shrink:0;z-index:49}
.tk-track{display:flex;animation:tkr 80s linear infinite;white-space:nowrap;will-change:transform}
.tk-item{font-size:11px;font-family:var(--mono);font-weight:500;padding:0 20px;border-right:1px solid var(--bd);display:inline-flex;align-items:center;gap:8px;height:30px;flex-shrink:0}
.tk-sym{color:var(--accent);font-weight:700;letter-spacing:.5px}
.tk-price{color:var(--tx-2)}
.tk-chg{font-weight:600}
.tk-up{color:var(--ok)}
.tk-dn{color:var(--err)}
@keyframes tkr{0%{transform:translateX(0)}100%{transform:translateX(-33.333%)}}

/* ===== Workspace: takes ALL remaining height, overflow hidden ===== */
.workspace{flex:1;display:flex;overflow:hidden;min-height:0}

/* Main scrolls independently within its own box */
.main{flex:1;overflow-y:auto;min-width:0}

/* Chat panel — slides in from right, squeezes main */
.chat-panel{width:0;min-width:0;overflow:hidden;display:flex;flex-shrink:0;transition:width .35s var(--ease),min-width .35s var(--ease);position:relative}
.cp-resize{position:absolute;left:0;top:0;bottom:0;width:5px;cursor:ew-resize;z-index:10;transition:background .15s}
.cp-resize:hover,.cp-resize:active{background:var(--accent)}
.cp-inner{flex:1;display:flex;flex-direction:column;border-left:1px solid var(--bd);background:var(--bg-1);min-width:0;overflow:hidden}
.cp-head{padding:12px 16px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--bd);background:var(--bg-2);flex-shrink:0}
.cp-head-l{display:flex;align-items:center;gap:10px}
.cp-head-title{font-family:var(--serif);font-size:13px;font-weight:700;color:var(--tx-1);letter-spacing:.5px}
.cp-close{width:26px;height:26px;border:1px solid var(--bd);background:var(--bg-3);color:var(--tx-3);cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .12s}
.cp-close:hover{background:var(--bg-4);color:var(--tx-1)}
/* ===== cp-body: flex:1 + overflow:hidden creates the scroll boundary for ChatBot ===== */
.cp-body{flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:0}

/* FAB */
.fab{position:fixed;bottom:24px;right:24px;z-index:1000;width:52px;height:52px;border:none;cursor:pointer;background:var(--accent);color:var(--bg-0);display:flex;align-items:center;justify-content:center;box-shadow:0 4px 24px rgba(201,168,76,.3),0 0 0 1px rgba(201,168,76,.15),inset 0 1px 0 rgba(255,255,255,.1);transition:transform .2s var(--ease),box-shadow .2s ease,background .2s ease}
.fab:hover{transform:scale(1.06);box-shadow:0 6px 32px rgba(201,168,76,.4),0 0 0 1px rgba(201,168,76,.2)}
.fab:active{transform:scale(.94)}
.fab-b{font-family:var(--serif);font-weight:700;font-size:16px;letter-spacing:.3px}
.fab-b sup{font-size:10px}

@media(max-width:900px){
  .chat-panel--open{position:fixed!important;top:84px;right:0;bottom:0;width:100%!important;min-width:100%!important;z-index:100}
  .cp-resize{display:none}
  .fab{bottom:16px;right:16px}
}
`;