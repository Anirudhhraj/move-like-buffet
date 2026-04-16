import { useState, useEffect, useCallback, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid, BarChart, Bar, Cell, AreaChart, Area } from "recharts";

const API = "";
const SERIF = "'Libre Baskerville', Georgia, serif";
const SANS = "'DM Sans', system-ui, sans-serif";
const MONO = "'JetBrains Mono', 'SF Mono', monospace";

const C = {
  tx1:"#e4ddd0", tx2:"#9b927f", tx3:"#6b6358",
  bg0:"#080b12", bg1:"#0d1117", bg2:"#141a24", bg3:"#1b2332", bg4:"#243044",
  bd:"#1e2a3a", bdl:"#2d3f56",
  accent:"#c9a84c", accentL:"#dfc06a",
  ok:"#4a9e6d", err:"#c4463a", warn:"#c9a84c", blue:"#5b8db8",
};

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/* ══════════════════════════════════════════════════════════════════════
   useFetch — FIXED:
   1. AbortController cancels in-flight HTTP when a new call starts
   2. Race-safe: only applies results if still the active request
   3. Keeps stale data visible during reload (no blank flash)
   ══════════════════════════════════════════════════════════════════════ */
function useFetch() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const controllerRef = useRef(null);
  const activeUrl = useRef(null);

  const run = useCallback(async (url) => {
    // ABORT any in-flight request — this is the key rate-limit fix
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    activeUrl.current = url;

    setLoading(true);
    setError(null);
    try {
      const r = await fetch(url, { signal: controller.signal });
      if (!r.ok) {
        const b = await r.json().catch(() => ({}));
        throw new Error(b.detail || `HTTP ${r.status}`);
      }
      const j = await r.json();
      if (j.detail) throw new Error(j.detail);
      // Only apply if this is still the active request
      if (activeUrl.current === url && !controller.signal.aborted) {
        setData(j);
        setLoading(false);
      }
      return j;
    } catch (e) {
      // Silently ignore aborted requests — they're intentional
      if (e.name === "AbortError") return null;
      if (activeUrl.current === url && !controller.signal.aborted) {
        setError(e.message);
        setData(null);
        setLoading(false);
      }
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    if (controllerRef.current) controllerRef.current.abort();
    activeUrl.current = null;
    setData(null);
    setLoading(false);
    setError(null);
  }, []);

  return { data, loading, error, run, reset };
}

/* ── Shared UI ───────────────────────────────────────────────────────── */

function Card({ children, style = {} }) {
  return <div style={{ background: C.bg2, border: `1px solid ${C.bd}`, padding: 20, ...style }}>{children}</div>;
}
function Sec({ children }) {
  return <div style={{ fontFamily: SERIF, fontSize: 12, fontWeight: 700, color: C.accent, marginBottom: 14, letterSpacing: "1px", textTransform: "uppercase" }}>{children}</div>;
}
function Stat({ label, value, sub, tone = "default" }) {
  const tones = {
    default: { bg: C.bg3, color: C.tx1, bd: C.bd },
    primary: { bg: "rgba(91,141,184,0.08)", color: C.blue, bd: "rgba(91,141,184,0.2)" },
    success: { bg: "rgba(74,158,109,0.06)", color: C.ok, bd: "rgba(74,158,109,0.2)" },
    warning: { bg: "rgba(201,168,76,0.06)", color: C.warn, bd: "rgba(201,168,76,0.2)" },
    danger:  { bg: "rgba(196,70,58,0.06)", color: C.err, bd: "rgba(196,70,58,0.2)" },
    neutral: { bg: C.bg3, color: C.tx2, bd: C.bd },
  };
  const t = tones[tone] || tones.default;
  return (
    <div style={{ minWidth: 130, flex: 1, padding: 16, background: t.bg, border: `1px solid ${t.bd}` }}>
      <div style={{ fontSize: 10, color: C.tx3, marginBottom: 6, textTransform: "uppercase", letterSpacing: "1px", fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: t.color, fontFamily: MONO }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: C.tx3, marginTop: 4, fontFamily: MONO }}>{sub}</div>}
    </div>
  );
}
function Pill({ text, tone }) {
  const m = {
    success: { bg: "rgba(74,158,109,0.1)", color: C.ok, bd: "rgba(74,158,109,0.2)" },
    warning: { bg: "rgba(201,168,76,0.1)", color: C.warn, bd: "rgba(201,168,76,0.2)" },
    danger:  { bg: "rgba(196,70,58,0.1)", color: C.err, bd: "rgba(196,70,58,0.2)" },
    neutral: { bg: C.bg3, color: C.tx2, bd: C.bd },
  };
  const s = m[tone] || m.neutral;
  return <span style={{ display: "inline-flex", alignItems: "center", padding: "5px 10px", fontSize: 12, fontWeight: 600, fontFamily: MONO, background: s.bg, color: s.color, border: `1px solid ${s.bd}` }}>{text}</span>;
}

function InfoBar({ data, sym }) {
  if (!data) return null;
  const chips = [
    { l: "PRICE", v: data.currentPrice ? `$${data.currentPrice.toFixed(2)}` : "N/A" },
    { l: "P/E", v: data.trailingPE ? data.trailingPE.toFixed(1) : "N/A" },
    { l: "MKT CAP", v: data.marketCap ? `$${(data.marketCap / 1e9).toFixed(1)}B` : "N/A" },
    { l: "52W HIGH", v: data.fiftyTwoWeekHigh ? `$${data.fiftyTwoWeekHigh.toFixed(2)}` : "N/A" },
    { l: "52W LOW", v: data.fiftyTwoWeekLow ? `$${data.fiftyTwoWeekLow.toFixed(2)}` : "N/A" },
    { l: "BETA", v: data.beta ? data.beta.toFixed(2) : "N/A" },
  ];
  return (
    <Card style={{ marginBottom: 2, borderBottom: "none" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
        <div style={{ minWidth: 200, flex: 1 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
            <span style={{ fontSize: 28, fontWeight: 700, color: C.tx1, fontFamily: SERIF }}>{sym}</span>
            {data.currentPrice && <span style={{ fontSize: 20, fontWeight: 700, color: C.accent, fontFamily: MONO }}>${data.currentPrice.toFixed(2)}</span>}
          </div>
          <div style={{ fontSize: 14, color: C.tx2, marginBottom: 3 }}>{data.longName}</div>
          <div style={{ fontSize: 12, color: C.tx3, letterSpacing: "0.5px" }}>{data.sector} · {data.industry}</div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(90px, 1fr))", gap: 4, flex: 2, minWidth: 260 }}>
          {chips.map(c => (
            <div key={c.l} style={{ padding: "10px 12px", background: C.bg3, border: `1px solid ${C.bd}` }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: C.tx1, fontFamily: MONO }}>{c.v}</div>
              <div style={{ fontSize: 9, color: C.tx3, marginTop: 3, textTransform: "uppercase", letterSpacing: "1px", fontWeight: 700 }}>{c.l}</div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

const ttStyle = { border: `1px solid ${C.bd}`, fontSize: 12, fontFamily: SANS, background: C.bg3, color: C.tx1, borderRadius: 0 };
const axStyle = { fontSize: 10, fill: C.tx3, fontFamily: MONO };

/* ── Panels (unchanged logic, consistent design) ─────────────────────── */

const RATIO_META = [
  { key: "gross_margin", label: "Gross Margin", rule: "≥ 40%", pass: v => v >= 40, sec: "Income" },
  { key: "sga_margin", label: "SG&A Margin", rule: "≤ 30%", pass: v => v <= 30, sec: "Income" },
  { key: "net_margin", label: "Net Margin", rule: "≥ 20%", pass: v => v >= 20, sec: "Income" },
  { key: "eps_growth", label: "EPS Growth", rule: "> 1.0×", pass: v => v > 1, fmt: v => v != null ? `${v.toFixed(2)}×` : "N/A", sec: "Income" },
  { key: "cash_gt_debt", label: "Cash / Debt", rule: "> 1.0×", pass: v => v > 1, fmt: v => v != null ? `${v.toFixed(2)}×` : "N/A", sec: "Balance Sheet" },
  { key: "adj_debt_to_equity", label: "Debt / Equity", rule: "< 0.80", pass: v => v < 0.8, fmt: v => v != null ? v.toFixed(2) : "N/A", sec: "Balance Sheet" },
  { key: "capex_margin", label: "CapEx Margin", rule: "< 25%", pass: v => v < 25, sec: "Cash Flow" },
];
function fmtR(v, m) { if (v == null) return "N/A"; if (m.fmt) return m.fmt(v); return `${v.toFixed(1)}%`; }

function Scorecard({ data }) {
  const ps = data.ratios, latest = ps[0];
  const passed = RATIO_META.filter(m => latest[m.key] != null && m.pass(latest[m.key])).length;
  const failed = RATIO_META.filter(m => latest[m.key] != null && !m.pass(latest[m.key])).length;
  const score = passed + failed > 0 ? Math.round((passed / (passed + failed)) * 100) : 0;
  const st = score >= 70 ? "success" : score >= 45 ? "warning" : "danger";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Stat label="Buffett Score" value={`${score}%`} tone={st} />
        <Stat label="Pass" value={passed} tone="success" />
        <Stat label="Fail" value={failed} tone="danger" />
      </div>
      {["Income", "Balance Sheet", "Cash Flow"].map(sec => (
        <div key={sec}>
          <Sec>{sec}</Sec>
          <div style={{ border: `1px solid ${C.bd}`, overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead><tr style={{ background: C.bg3 }}>
                <th style={{ textAlign: "left", padding: "10px 14px", color: C.tx3, fontWeight: 700, fontSize: 10, letterSpacing: "1px", textTransform: "uppercase" }}>Metric</th>
                <th style={{ textAlign: "center", padding: "10px", color: C.tx3, fontWeight: 700, fontSize: 10, letterSpacing: "1px", textTransform: "uppercase" }}>Threshold</th>
                {ps.map(p => <th key={p.period} style={{ textAlign: "center", padding: "10px", color: C.tx3, fontWeight: 700, fontSize: 10, fontFamily: MONO }}>{p.period.slice(0, 7)}</th>)}
              </tr></thead>
              <tbody>
                {RATIO_META.filter(m => m.sec === sec).map((m, i) => (
                  <tr key={m.key} style={{ background: i % 2 === 0 ? C.bg2 : C.bg3, borderTop: `1px solid ${C.bd}` }}>
                    <td style={{ textAlign: "left", padding: "10px 14px", color: C.tx1, fontSize: 13 }}>{m.label}</td>
                    <td style={{ textAlign: "center", padding: "10px" }}><span style={{ padding: "3px 8px", background: C.bg4, color: C.tx3, fontSize: 11, fontFamily: MONO }}>{m.rule}</span></td>
                    {ps.map(p => { const v = p[m.key]; const ok = v != null && m.pass(v); const na = v == null; return <td key={p.period} style={{ textAlign: "center", padding: "10px" }}><Pill text={fmtR(v, m)} tone={na ? "neutral" : ok ? "success" : "danger"} /></td>; })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}

function DCFPanel({ data }) {
  const vt = data.verdict === "Undervalued" ? "success" : data.verdict === "Overvalued" ? "danger" : "warning";
  const cd = data.pv_fcf_by_year.map((v, i) => ({ year: `Y${i + 1}`, pv: Math.round(v / 1e6) }));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Stat label="Intrinsic / Share" value={`$${data.intrinsic_per_share?.toLocaleString() || "N/A"}`} tone={vt} />
        <Stat label="Current Price" value={`$${data.current_price ?? "N/A"}`} tone="primary" />
        <Stat label="Margin of Safety" value={data.margin_of_safety_pct != null ? `${data.margin_of_safety_pct > 0 ? "+" : ""}${data.margin_of_safety_pct}%` : "N/A"} tone={vt} />
      </div>
      <Pill text={data.verdict} tone={vt} />
      <Sec>PV of Free Cash Flow ($M)</Sec>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={cd} barSize={24}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke={C.bd} />
          <XAxis dataKey="year" tick={axStyle} axisLine={false} tickLine={false} />
          <YAxis tick={axStyle} axisLine={false} tickLine={false} tickFormatter={v => `$${v}M`} />
          <Tooltip contentStyle={ttStyle} formatter={v => [`$${v}M`, "PV FCF"]} />
          <Bar dataKey="pv" radius={[0,0,0,0]}>{cd.map((_, i) => <Cell key={i} fill={i % 2 === 0 ? C.blue : "#7aa8c9"} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function AltmanPanel({ data }) {
  const tone = data.zone === "Safe Zone" ? "success" : data.zone === "Grey Zone" ? "warning" : "danger";
  const comps = [
    { l: "Working Capital / Assets", v: data.components.X1_working_capital_ratio },
    { l: "Retained Earnings / Assets", v: data.components.X2_retained_earnings_ratio },
    { l: "EBIT / Assets", v: data.components.X3_ebit_ratio },
    { l: "Market Cap / Liabilities", v: data.components.X4_market_cap_to_liabilities },
    { l: "Revenue / Assets", v: data.components.X5_asset_turnover },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Stat label="Z-Score" value={data.z_score} tone={tone} />
        <Stat label="Zone" value={data.zone} tone={tone} sub={data.description} />
      </div>
      <Sec>Components</Sec>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {comps.map(c => (
          <div key={c.l} style={{ padding: 14, border: `1px solid ${C.bd}`, background: C.bg3 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <span style={{ fontSize: 13, color: C.tx1 }}>{c.l}</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: c.v >= 0 ? C.ok : C.err, fontFamily: MONO }}>{c.v?.toFixed(4)}</span>
            </div>
            <div style={{ height: 4, background: C.bg4, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${clamp(Math.abs(c.v) * 60, 0, 100)}%`, background: c.v >= 0 ? C.ok : C.err }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PiotroskiPanel({ data }) {
  const tone = data.f_score >= 8 ? "success" : data.f_score >= 5 ? "warning" : "danger";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Stat label="F-Score" value={`${data.f_score} / 9`} tone={tone} />
        <Stat label="Strength" value={data.strength} tone={tone} />
      </div>
      {["Profitability", "Leverage", "Efficiency"].map(cat => (
        <div key={cat}><Sec>{cat}</Sec>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {Object.entries(data.signals).filter(([, v]) => v.category === cat).map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 14px", background: C.bg3, border: `1px solid ${C.bd}` }}>
                <span style={{ fontSize: 13, color: C.tx1 }}>{v.label}</span>
                <Pill text={v.score ? "PASS" : "FAIL"} tone={v.score ? "success" : "danger"} />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function ClusterPanel({ data }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Sec>Peer Groups — {data.sector}</Sec>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {Object.entries(data.cluster_members).map(([cl, mem]) => {
          const isTgt = Number(cl) === data.target_cluster;
          return (
            <div key={cl} style={{ flex: 1, minWidth: 200, padding: 16, background: isTgt ? "rgba(91,141,184,0.06)" : C.bg3, border: `1px solid ${isTgt ? "rgba(91,141,184,0.25)" : C.bd}` }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: isTgt ? C.blue : C.tx1, marginBottom: 8, textTransform: "uppercase", letterSpacing: "1px" }}>Cluster {Number(cl) + 1}{isTgt ? " · YOUR POSITION" : ""}</div>
              <div style={{ fontSize: 13, color: C.tx2, lineHeight: 1.7 }}>{mem.join(", ")}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TechnicalPanel({ data }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Stat label="RSI (14)" value={data.rsi_14.toFixed(1)} sub={data.rsi_signal} tone={data.rsi_14 > 70 ? "danger" : data.rsi_14 < 30 ? "success" : "warning"} />
        <Stat label="MA Signal" value={data.ma_signal.split(" (")[0]} sub={`50: $${data.ma_50} · 200: $${data.ma_200}`} tone={data.ma_signal.includes("Bull") ? "success" : "danger"} />
        <Stat label="Price" value={`$${data.price}`} sub={`52W: $${data.low_52w} – $${data.high_52w}`} tone="primary" />
      </div>
      <Sec>90-Day Price Action</Sec>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data.price_series}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke={C.bd} />
          <XAxis dataKey="date" tick={axStyle} tickFormatter={d => d.slice(5)} interval={2} axisLine={false} tickLine={false} />
          <YAxis tick={axStyle} axisLine={false} tickLine={false} tickFormatter={v => `$${v}`} />
          <Tooltip contentStyle={ttStyle} formatter={v => [`$${v}`, "Price"]} />
          <ReferenceLine y={data.ma_50} stroke={C.blue} strokeDasharray="5 3" />
          <ReferenceLine y={data.ma_200} stroke={C.accent} strokeDasharray="5 3" />
          <Line type="monotone" dataKey="price" stroke={C.tx1} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function MonteCarloPanel({ data }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Stat label="Current" value={`$${data.current_price}`} tone="primary" />
        <Stat label="Median" value={`$${data.forecast.p50}`} tone="neutral" />
        <Stat label="Bull P90" value={`$${data.forecast.p90}`} tone="success" />
        <Stat label="Bear P10" value={`$${data.forecast.p10}`} tone="danger" />
        <Stat label="% Positive" value={`${data.pct_paths_positive}%`} tone={data.pct_paths_positive >= 60 ? "success" : data.pct_paths_positive >= 40 ? "warning" : "danger"} />
      </div>
      <Sec>1Y Forecast — {data.simulations.toLocaleString()} Simulations</Sec>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data.chart_data}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke={C.bd} />
          <XAxis dataKey="day" tick={axStyle} axisLine={false} tickLine={false} />
          <YAxis tick={axStyle} axisLine={false} tickLine={false} tickFormatter={v => `$${v}`} />
          <Tooltip contentStyle={ttStyle} />
          <Area type="monotone" dataKey="p90" stroke="none" fill="rgba(91,141,184,0.08)" name="P90" />
          <Area type="monotone" dataKey="p75" stroke="none" fill="rgba(74,158,109,0.08)" name="P75" />
          <Area type="monotone" dataKey="p25" stroke="none" fill="rgba(201,168,76,0.08)" name="P25" />
          <Area type="monotone" dataKey="p10" stroke="none" fill="rgba(196,70,58,0.08)" name="P10" />
          <Line type="monotone" dataKey="p50" stroke={C.tx1} strokeWidth={2} dot={false} name="Median" />
          <ReferenceLine y={data.current_price} stroke={C.accent} strokeDasharray="5 3" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function FinTable({ data, label }) {
  const [search, setSearch] = useState("");
  if (!data || !Object.keys(data).length) return <p style={{ color: C.tx3, fontSize: 13 }}>No data available.</p>;
  const periods = Object.keys(data);
  let rows = [...new Set(periods.flatMap(p => Object.keys(data[p])))];
  if (search) rows = rows.filter(r => r.toLowerCase().includes(search.toLowerCase()));
  function fmt(v) { if (v == null) return "—"; if (Math.abs(v) > 1e9) return `$${(v / 1e9).toFixed(2)}B`; if (Math.abs(v) > 1e6) return `$${(v / 1e6).toFixed(2)}M`; return typeof v === "number" ? v.toFixed(2) : v; }
  return (
    <div>
      <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Filter metrics..." style={inputSt} />
      <div style={{ overflowX: "auto", border: `1px solid ${C.bd}`, marginTop: 12 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead><tr style={{ background: C.bg3 }}><th style={{ textAlign: "left", padding: "10px 14px", color: C.tx3, fontWeight: 700, fontSize: 10, textTransform: "uppercase", letterSpacing: "1px" }}>{label}</th>
            {periods.map(p => <th key={p} style={{ textAlign: "right", padding: "10px 14px", color: C.tx3, fontWeight: 700, fontSize: 10, fontFamily: MONO }}>{p.slice(0, 7)}</th>)}</tr></thead>
          <tbody>{rows.map((row, i) => (
            <tr key={row} style={{ background: i % 2 === 0 ? C.bg2 : C.bg3, borderTop: `1px solid ${C.bd}` }}>
              <td style={{ textAlign: "left", padding: "10px 14px", color: C.tx1 }}>{row}</td>
              {periods.map(p => { const v = data[p][row]; return <td key={p} style={{ textAlign: "right", padding: "10px 14px", color: typeof v === "number" && v < 0 ? C.err : C.tx1, fontFamily: MONO, fontSize: 12 }}>{fmt(v)}</td>; })}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Tabs ─────────────────────────────────────────────────────────────── */

const TABS = [
  { id: "scorecard", label: "Buffett Score", ep: s => `/stock/buffett-ratios?symbol=${s}` },
  { id: "dcf", label: "DCF", ep: s => `/stock/dcf?symbol=${s}` },
  { id: "altman", label: "Altman Z", ep: s => `/stock/altman-z?symbol=${s}` },
  { id: "piotroski", label: "Piotroski F", ep: s => `/stock/piotroski?symbol=${s}` },
  { id: "cluster", label: "Peers", ep: s => `/stock/cluster?symbol=${s}` },
  { id: "technical", label: "Technical", ep: s => `/stock/technical?symbol=${s}` },
  { id: "montecarlo", label: "Monte Carlo", ep: s => `/stock/monte-carlo?symbol=${s}` },
  { id: "income", label: "Income Stmt", ep: s => `/stock/financials?symbol=${s}` },
  { id: "balance", label: "Balance Sheet", ep: s => `/stock/financials?symbol=${s}` },
  { id: "cashflow", label: "Cash Flow", ep: s => `/stock/financials?symbol=${s}` },
];

/* ══════════════════════════════════════════════════════════════════════
   Dashboard — FIXED:
   1. fetchTab uses stable tabDataRun ref (no re-render loop)
   2. Tab switch is debounced 80ms to absorb rapid clicks
   3. fetchSymbol guarded by fetchingRef to prevent overlapping calls
   4. StrictMode mount guard
   5. null data shows loading spinner instead of blank
   ══════════════════════════════════════════════════════════════════════ */

export default function Dashboard() {
  const [inputSym, setInputSym] = useState("AAPL");
  const [activeSym, setActiveSym] = useState("");
  const [subTab, setSubTab] = useState("scorecard");
  const [fetching, setFetching] = useState(false);

  const info = useFetch();
  const tabData = useFetch();

  const lastFetch = useRef({ tab: null, sym: null, url: null });
  const fetchingRef = useRef(false);
  const tabDebounce = useRef(null);

  const tabDataRun = tabData.run;  // stable
  const infoRun = info.run;        // stable

  const fetchTab = useCallback(async (tab, symbol) => {
    const tabDef = TABS.find(t => t.id === tab);
    if (!tabDef || !symbol) return;
    const url = `${API}${tabDef.ep(symbol)}`;
    if (lastFetch.current.url === url && lastFetch.current.tab === tab && lastFetch.current.sym === symbol) return;
    lastFetch.current = { tab, sym: symbol, url };
    await tabDataRun(url);
  }, [tabDataRun]);

  async function fetchSymbol(sym) {
    const symbol = sym.trim().toUpperCase();
    if (!symbol || fetchingRef.current) return;
    fetchingRef.current = true;
    setActiveSym(symbol);
    setFetching(true);
    lastFetch.current = { tab: null, sym: null, url: null };

    // Step 1: fetch company info
    await infoRun(`${API}/stock/info?symbol=${symbol}`);

    // Step 2: fetch the active tab data (backend's own 0.4s cooldown handles yfinance pacing)
    await fetchTab(subTab, symbol);

    setFetching(false);
    fetchingRef.current = false;
  }

  // Debounced tab switch — 200ms absorbs rapid clicking, reduces yfinance pressure
  useEffect(() => {
    if (!activeSym) return;
    if (tabDebounce.current) clearTimeout(tabDebounce.current);
    tabDebounce.current = setTimeout(() => {
      fetchTab(subTab, activeSym);
    }, 200);
    return () => { if (tabDebounce.current) clearTimeout(tabDebounce.current); };
  }, [subTab, activeSym, fetchTab]);

  // Auto-load AAPL immediately — no delay, ticker is held back so yfinance is free
  useEffect(() => {
    fetchSymbol("AAPL");
  }, []);

  function renderPanel() {
    if (tabData.loading) return <LoadingState tab={subTab} />;
    if (tabData.error) return (
      <div style={errSt}>
        <div style={{ fontSize: 13, fontWeight: 700, color: C.err, marginBottom: 4, fontFamily: SERIF }}>Unable to load data</div>
        <div style={{ fontSize: 12, color: C.tx3 }}>{tabData.error}</div>
        <button onClick={() => { lastFetch.current = { tab: null, sym: null, url: null }; fetchTab(subTab, activeSym); }} style={retryBtnSt}>Retry</button>
      </div>
    );
    if (!tabData.data) return <LoadingState tab={subTab} />;
    const d = tabData.data;
    if (subTab === "scorecard") return <Scorecard data={d} />;
    if (subTab === "dcf") return <DCFPanel data={d} />;
    if (subTab === "altman") return <AltmanPanel data={d} />;
    if (subTab === "piotroski") return <PiotroskiPanel data={d} />;
    if (subTab === "cluster") return <ClusterPanel data={d} />;
    if (subTab === "technical") return <TechnicalPanel data={d} />;
    if (subTab === "montecarlo") return <MonteCarloPanel data={d} />;
    if (subTab === "income") return <FinTable data={d?.income_statement} label="Metric" />;
    if (subTab === "balance") return <FinTable data={d?.balance_sheet} label="Metric" />;
    if (subTab === "cashflow") return <FinTable data={d?.cash_flow} label="Metric" />;
    return null;
  }

  return (
    <div style={{ fontFamily: SANS, color: C.tx1, minHeight: "100%" }}>
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "24px 24px 100px" }}>
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontFamily: SERIF, fontSize: 30, fontWeight: 700, letterSpacing: "-0.3px", color: C.tx1 }}>Equity Research Desk</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
            <div style={{ width: 28, height: 1, background: C.accent }} />
            <span style={{ fontSize: 12, color: C.tx3, letterSpacing: "1px", textTransform: "uppercase", fontWeight: 500 }}>Fundamental & Technical Intelligence</span>
          </div>
        </div>

        <Card style={{ marginBottom: 2, borderBottom: "none" }}>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ position: "relative", flex: "0 0 auto", maxWidth: 260, width: "100%" }}>
              <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", fontSize: 11, color: C.tx3, fontFamily: MONO, fontWeight: 700 }}>SYM</span>
              <input value={inputSym} onChange={e => setInputSym(e.target.value.toUpperCase())} onKeyDown={e => e.key === "Enter" && fetchSymbol(inputSym)} placeholder="AAPL" style={{ ...inputSt, paddingLeft: 48, maxWidth: 260 }} />
            </div>
            <button onClick={() => fetchSymbol(inputSym)} disabled={fetching} style={{ padding: "10px 28px", border: "none", background: fetching ? C.bg4 : C.accent, color: fetching ? C.tx3 : C.bg0, fontSize: 12, fontWeight: 700, letterSpacing: "1px", textTransform: "uppercase", cursor: fetching ? "default" : "pointer", fontFamily: SANS }}>{fetching ? "LOADING..." : "ANALYZE"}</button>
            {activeSym && !fetching && <span style={{ fontSize: 12, color: C.tx3, fontFamily: MONO }}>Showing: <strong style={{ color: C.accent }}>{activeSym}</strong></span>}
          </div>
        </Card>

        {info.loading && !info.data && <div style={{ ...centerSt, height: 60 }}>Loading company info...</div>}
        <InfoBar data={info.data} sym={activeSym} />

        <div style={{ display: "flex", flexWrap: "wrap", gap: 0, marginBottom: 2, background: C.bg2, border: `1px solid ${C.bd}`, borderBottom: "none", overflowX: "auto" }}>
          {TABS.map(tab => {
            const active = subTab === tab.id;
            return (
              <button key={tab.id} onClick={() => setSubTab(tab.id)} style={{
                padding: "11px 16px", border: "none",
                borderBottom: active ? `2px solid ${C.accent}` : "2px solid transparent",
                background: active ? C.bg3 : "transparent",
                color: active ? C.accent : C.tx3,
                fontSize: 11, fontWeight: 700, cursor: "pointer", fontFamily: SANS,
                letterSpacing: "0.5px", textTransform: "uppercase", whiteSpace: "nowrap",
              }}>{tab.label}</button>
            );
          })}
        </div>

        <Card>{renderPanel()}</Card>
      </div>
    </div>
  );
}

function LoadingState({ tab }) {
  return (
    <div style={centerSt}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
        <div style={{ width: 20, height: 20, border: `2px solid ${C.bd}`, borderTopColor: C.accent, borderRadius: "50%", animation: "spin .8s linear infinite" }} />
        <span style={{ fontSize: 13, color: C.tx3 }}>Loading {tab} data...</span>
        <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
      </div>
    </div>
  );
}

const inputSt = { width: "100%", padding: "10px 12px", border: `1px solid ${C.bd}`, fontSize: 14, outline: "none", boxSizing: "border-box", fontFamily: MONO, color: C.tx1, background: C.bg3 };
const centerSt = { minHeight: 180, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, color: C.tx3 };
const errSt = { padding: 16, background: "rgba(196,70,58,0.06)", border: "1px solid rgba(196,70,58,0.15)" };
const retryBtnSt = { marginTop: 10, padding: "6px 14px", border: `1px solid ${C.bd}`, background: C.bg3, color: C.tx2, fontSize: 12, cursor: "pointer", fontFamily: SANS };