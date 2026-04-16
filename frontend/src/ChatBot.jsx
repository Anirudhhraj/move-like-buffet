import { useState, useRef, useEffect, useCallback } from "react";
import { Send } from "lucide-react";

const API = "";

const SUGGESTIONS = [
  "What is Buffett's margin of safety?",
  "Why does Buffett avoid leverage?",
  "How did See's Candies change Buffett's thinking?",
  "What does Buffett look for in management?",
  "How does Buffett think about risk?",
];

/* ── Citation with tooltip ───────────────────────────────────────────── */

function CiteBadge({ num, sources }) {
  const [show, setShow] = useState(false);
  const src = sources?.find(s => s.ref_idx === num);
  return (
    <span className="cite-wrap" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      <span className="cite">[{num}]</span>
      {show && src && (
        <div className="cite-tip">
          <div className="cite-tip-type">{src.source_type === "qa_pair" ? "CURATED Q&A" : "SOURCE PASSAGE"}</div>
          <div className="cite-tip-label">{[src.label, src.sublabel].filter(Boolean).join(" / ")}</div>
          {src.source_file && <div className="cite-tip-file">{src.source_file}</div>}
          <div className="cite-tip-sim">Similarity: {(src.similarity * 100).toFixed(1)}%</div>
        </div>
      )}
    </span>
  );
}

function renderMarkdown(text) {
  if (!text) return null;
  // Split on **bold** markers
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    const boldMatch = part.match(/^\*\*(.+)\*\*$/);
    if (boldMatch) return <strong key={i} style={{ fontWeight: 700, color: "var(--tx-1)" }}>{boldMatch[1]}</strong>;
    return <span key={i}>{part}</span>;
  });
}

function renderWithCitations(text, sources) {
  if (!text) return null;
  // First split on citations [1], [2], etc.
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (m) return <CiteBadge key={i} num={parseInt(m[1])} sources={sources} />;
    // Then render markdown within each text segment
    return <span key={i}>{renderMarkdown(part)}</span>;
  });
}

/* ══════════════════════════════════════════════════════════════════════
   Research Log — FIXED:
   - Expanded by default (not collapsed)
   - Color-coded steps: search=blue, analyze=accent, generating=green, etc.
   - Full query shown prominently
   - Proper monospace terminal-style log formatting
   ══════════════════════════════════════════════════════════════════════ */

const STEP_STYLES = {
  hyde:       { icon: "HYD", color: "var(--blue)" },
  hyde_done:  { icon: "HYD", color: "var(--blue)" },
  search:     { icon: "SRC", color: "var(--blue)" },
  query:      { icon: "QRY", color: "var(--accent-l)" },
  analyze:    { icon: "ANL", color: "var(--accent-l)" },
  found:      { icon: "FND", color: "var(--ok)" },
  gap:        { icon: "GAP", color: "var(--warn)" },
  sufficient: { icon: " OK", color: "var(--ok)" },
  done:       { icon: "DON", color: "var(--ok)" },
  generating: { icon: "GEN", color: "var(--ok)" },
  citation:   { icon: "CIT", color: "var(--tx-3)" },
};

function ResearchLog({ events }) {
  const [collapsed, setCollapsed] = useState(false);
  if (!events || !events.length) return null;

  return (
    <div className="rl-wrap">
      <button onClick={() => setCollapsed(c => !c)} className="rl-toggle">
        <span className="rl-toggle-arrow">{collapsed ? "▶" : "▼"}</span>
        <span className="rl-toggle-label">RESEARCH LOG</span>
        <span className="rl-toggle-count">{events.length} steps</span>
      </button>
      {!collapsed && (
        <div className="rl-panel">
          {events.map((e, i) => {
            const st = STEP_STYLES[e.step] || { icon: "···", color: "var(--tx-3)" };
            return (
              <div key={i} className="rl-line">
                <span className="rl-step" style={{ color: st.color }}>{st.icon}</span>
                <span className="rl-detail">{e.detail}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ── Source map ───────────────────────────────────────────────────────── */

function SourceMap({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources?.length) return null;
  return (
    <div style={{ marginTop: 6 }}>
      <button onClick={() => setOpen(o => !o)} className="sm-btn">
        <span style={{ fontSize: 9 }}>{open ? "▼" : "▶"}</span>
        Sources ({sources.length})
      </button>
      {open && (
        <div className="sm-panel">
          {sources.map((s, i) => (
            <div key={i} className="sm-row">
              <span className="sm-idx">[{s.ref_idx}]</span>{" "}
              <span className="sm-type">{s.source_type}</span>{" "}
              {[s.label, s.sublabel].filter(Boolean).join(" / ")}
              {s.source_file ? ` (${s.source_file})` : ""}{" "}
              <span className="sm-sim">{(s.similarity * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Meta badges ─────────────────────────────────────────────────────── */

function MetaBadge({ strategy, confidence, duration, enrichedQuery }) {
  if (!strategy || strategy === "reject") return null;
  return (
    <div className="mb-row">
      <span className="mb-tag">{strategy}</span>
      <span className={`mb-tag mb-conf mb-conf-${confidence}`}>{confidence}</span>
      <span className="mb-tag">{duration}ms</span>
      {enrichedQuery && <div className="mb-eq">Enriched query: <strong>{enrichedQuery}</strong></div>}
    </div>
  );
}

function TypingDots() {
  return (
    <div className="typing">
      {[0, 1, 2].map(i => <div key={i} className="tdot" style={{ animationDelay: `${i * 0.18}s` }} />)}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   ChatBot — FIXED:
   - scrollTop on the messages container, NOT scrollIntoView
     (scrollIntoView propagates to all scrollable ancestors = left side moves)
   - Research log expanded by default with proper formatting
   ══════════════════════════════════════════════════════════════════════ */

export default function ChatBot() {
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const msgsRef = useRef(null);   // ← ref on the CONTAINER, not a bottom sentinel
  const inp = useRef(null);
  const tokBuf = useRef("");
  const rafId = useRef(null);

  // FIXED: scroll the container directly — never touches parent scroll contexts
  const scrollToBottom = useCallback(() => {
    const el = msgsRef.current;
    if (el) {
      requestAnimationFrame(() => {
        el.scrollTop = el.scrollHeight;
      });
    }
  }, []);

  useEffect(() => { scrollToBottom(); }, [msgs, scrollToBottom]);

  const flush = useCallback(() => {
    if (tokBuf.current) {
      const c = tokBuf.current; tokBuf.current = "";
      setMsgs(p => { const cp = [...p]; const l = cp[cp.length - 1]; if (l?.streaming) cp[cp.length - 1] = { ...l, text: l.text + c }; return cp; });
    }
    rafId.current = null;
  }, []);

  async function send(q) {
    const question = (q || input).trim();
    if (!question || loading) return;
    setInput("");
    if (inp.current) inp.current.style.height = "auto";

    const history = msgs.filter(m => m.role && m.text && !m.streaming).map(m => ({ role: m.role === "user" ? "user" : "assistant", content: m.text }));

    setMsgs(p => [...p, { role: "user", text: question }]);
    setLoading(true);
    setMsgs(p => [...p, { role: "assistant", text: "", streaming: true, events: [], sources: [], strategy: "", confidence: "", duration_ms: 0, enriched_query: "" }]);

    try {
      const res = await fetch(`${API}/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: question, history }) });
      if (!res.ok) throw new Error(`Server ${res.status}`);

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n\n");
        buf = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let data; try { data = JSON.parse(line.slice(6)); } catch { continue; }

          if (data.type === "research") {
            setMsgs(p => { const cp = [...p]; const l = cp[cp.length - 1]; if (l?.streaming) cp[cp.length - 1] = { ...l, events: [...(l.events || []), { step: data.step, detail: data.detail }] }; return cp; });
          } else if (data.type === "token") {
            tokBuf.current += data.content;
            if (!rafId.current) rafId.current = requestAnimationFrame(flush);
          } else if (data.type === "done") {
            if (rafId.current) { cancelAnimationFrame(rafId.current); rafId.current = null; }
            flush();
            const d = data.data;
            setMsgs(p => { const cp = [...p]; const l = cp[cp.length - 1]; if (l?.streaming) cp[cp.length - 1] = { ...l, text: d.answer || l.text, streaming: false, strategy: d.strategy, confidence: d.confidence, sources: d.sources || [], duration_ms: d.duration_ms, enriched_query: d.enriched_query || "" }; return cp; });
          } else if (data.type === "error") {
            setMsgs(p => { const cp = [...p]; const l = cp[cp.length - 1]; if (l?.streaming) cp[cp.length - 1] = { ...l, text: "Error: " + data.detail, streaming: false }; return cp; });
          }
        }
      }
    } catch (err) {
      setMsgs(p => { const cp = [...p]; const l = cp[cp.length - 1]; if (l?.streaming) cp[cp.length - 1] = { ...l, text: "Could not reach the server. Is the backend running?\n\n" + err.message, streaming: false }; return cp; });
    } finally {
      setLoading(false);
      inp.current?.focus();
    }
  }

  const empty = msgs.length === 0;

  return (
    <div className="cb-root">
      {empty && (
        <div className="cb-empty">
          <div className="cb-empty-icon"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></div>
          <p className="cb-empty-title">Ask the Oracle</p>
          <div className="cb-empty-rule" />
          <p className="cb-empty-text">Investing philosophy, strategies, and decisions — grounded in Buffett's own words and primary sources.</p>
          <div className="cb-sug-list">
            {SUGGESTIONS.map(t => <button key={t} onClick={() => send(t)} className="cb-sug">{t}</button>)}
          </div>
        </div>
      )}

      {!empty && (
        <div className="cb-msgs" ref={msgsRef}>
          {msgs.map((m, i) => (
            <div key={i} className={`msg msg-${m.role}`}>
              {m.role === "assistant" && <div className="av av-bot"><span>B</span></div>}
              <div className={`msg-c msg-c-${m.role}`}>
                {m.role === "assistant" && m.events?.length > 0 && (
                  <ResearchLog events={m.events} />
                )}
                <div className={`bub bub-${m.role}`}>
                  {m.role === "assistant"
                    ? (m.text ? renderWithCitations(m.text, m.sources) : (m.streaming ? <TypingDots /> : "No response."))
                    : m.text}
                </div>
                {m.role === "assistant" && !m.streaming && m.sources?.length > 0 && <SourceMap sources={m.sources} />}
                {m.role === "assistant" && !m.streaming && m.strategy && <MetaBadge strategy={m.strategy} confidence={m.confidence} duration={m.duration_ms} enrichedQuery={m.enriched_query} />}
              </div>
              {m.role === "user" && <div className="av av-user"><span>U</span></div>}
            </div>
          ))}
        </div>
      )}

      <div className={`cb-input ${empty ? "cb-input--empty" : ""}`}>
        <div className="cb-input-wrap">
          <textarea ref={inp} value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder="Ask about Buffett..." rows={1} className="cb-ta"
            onInput={e => { e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 100) + "px"; }}
          />
          <button onClick={() => send()} disabled={loading || !input.trim()} className="cb-send">
            <Send size={13} strokeWidth={2.5} />
          </button>
        </div>
      </div>

      <style>{`
.cb-root{display:flex;flex-direction:column;height:100%;background:var(--bg-1);font-family:var(--sans);overflow:hidden}

.cb-empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:28px 20px;gap:8px}
.cb-empty-icon{color:var(--accent);opacity:.6}
.cb-empty-title{font-family:var(--serif);font-size:17px;font-weight:700;color:var(--tx-1);letter-spacing:.5px}
.cb-empty-rule{width:32px;height:1px;background:var(--accent);opacity:.5;margin:4px 0}
.cb-empty-text{font-size:12px;color:var(--tx-3);text-align:center;line-height:1.6;max-width:280px}
.cb-sug-list{display:flex;flex-direction:column;gap:4px;width:100%;margin-top:10px}
.cb-sug{padding:9px 14px;border:1px solid var(--bd);background:var(--bg-2);cursor:pointer;font-size:12px;color:var(--tx-2);text-align:left;line-height:1.4;font-family:inherit;transition:all .15s}
.cb-sug:hover{border-color:var(--accent-bd);background:var(--accent-bg);color:var(--accent-l)}

/* Messages container — THIS is the scroll boundary. Nothing outside scrolls. */
.cb-msgs{flex:1;overflow-y:auto;padding:16px 14px 8px;display:flex;flex-direction:column;gap:16px;min-height:0}

.msg{display:flex;gap:8px;align-items:flex-start}
.msg-user{justify-content:flex-end}
.msg-assistant{justify-content:flex-start}
.av{width:26px;height:26px;flex-shrink:0;display:flex;align-items:center;justify-content:center;margin-top:2px;font-size:10px;font-weight:800;font-family:var(--serif)}
.av-bot{background:var(--accent);color:var(--bg-0)}
.av-user{background:var(--bg-4);color:var(--tx-2)}
.msg-c{max-width:85%;display:flex;flex-direction:column}
.msg-c-user{align-items:flex-end}
.msg-c-assistant{align-items:flex-start}
.bub{padding:10px 14px;font-size:13px;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.bub-user{background:var(--accent);color:var(--bg-0);font-weight:500}
.bub-assistant{background:var(--bg-2);color:var(--tx-1);border:1px solid var(--bd)}

/* ── Research Log — terminal-style, expanded by default ─── */
.rl-wrap{margin-bottom:6px}
.rl-toggle{display:flex;align-items:center;gap:6px;padding:6px 10px;border:1px solid var(--bd);background:var(--bg-3);color:var(--tx-2);font-size:10px;font-weight:700;cursor:pointer;font-family:var(--mono);letter-spacing:.5px;width:100%;text-align:left}
.rl-toggle:hover{background:var(--bg-4)}
.rl-toggle-arrow{font-size:8px;color:var(--tx-3);width:10px}
.rl-toggle-label{text-transform:uppercase;letter-spacing:1px}
.rl-toggle-count{color:var(--tx-3);margin-left:auto}

.rl-panel{
  border:1px solid var(--bd);border-top:none;
  background:var(--bg-0);
  padding:8px 0;
  max-height:400px;overflow-y:auto;
  font-family:var(--mono);font-size:11px;line-height:1.8;
}
.rl-line{display:flex;gap:0;padding:2px 10px;border-left:2px solid transparent;align-items:flex-start}
.rl-line:hover{background:rgba(255,255,255,.02);border-left-color:var(--accent-bd)}
.rl-step{
  flex-shrink:0;width:36px;
  font-weight:700;letter-spacing:.5px;
  text-align:right;padding-right:8px;padding-top:1px;
}
.rl-detail{color:var(--tx-2);white-space:pre-wrap;word-break:break-word;flex:1;min-width:0}

/* Citations */
.cite-wrap{position:relative;display:inline}
.cite{display:inline-block;padding:0 4px;margin:0 1px;font-size:10px;font-weight:700;font-family:var(--mono);background:var(--accent-bg);color:var(--accent-l);border:1px solid var(--accent-bd);line-height:18px;vertical-align:middle;cursor:help;transition:background .12s}
.cite:hover{background:rgba(201,168,76,.2)}
.cite-tip{position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:var(--bg-4);border:1px solid var(--bd-l);padding:10px 12px;min-width:200px;max-width:260px;box-shadow:0 8px 24px rgba(0,0,0,.5);z-index:100;font-size:11px;line-height:1.5;pointer-events:none}
.cite-tip-type{color:var(--accent-l);font-weight:700;margin-bottom:3px;font-size:9px;letter-spacing:1px}
.cite-tip-label{color:var(--tx-1);font-weight:500}
.cite-tip-file{color:var(--tx-3);font-family:var(--mono);font-size:10px;margin-top:2px}
.cite-tip-sim{color:var(--tx-3);font-family:var(--mono);font-size:10px;margin-top:4px}

/* Sources */
.sm-btn{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border:1px solid var(--blue-bd);background:var(--blue-bg);color:var(--blue);font-size:11px;font-weight:600;cursor:pointer;font-family:inherit}
.sm-btn:hover{background:rgba(91,141,184,.12)}
.sm-panel{margin-top:4px;padding:8px 10px;background:var(--blue-bg);border:1px solid var(--blue-bd);font-size:11px;line-height:1.7;color:var(--tx-2);font-family:var(--mono)}
.sm-row{margin-bottom:3px}
.sm-idx{font-weight:700;color:var(--blue)}
.sm-type{color:var(--tx-3)}
.sm-sim{color:var(--tx-3)}

/* Meta */
.mb-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;font-size:11px}
.mb-tag{padding:2px 7px;background:var(--bg-3);border:1px solid var(--bd);color:var(--tx-3);font-family:var(--mono)}
.mb-conf{font-weight:600}
.mb-conf-high{color:var(--ok)}
.mb-conf-medium{color:var(--warn)}
.mb-conf-low{color:var(--err)}
.mb-eq{width:100%;padding:5px 8px;background:var(--bg-3);border:1px solid var(--bd);color:var(--tx-2);font-family:var(--mono);font-size:11px;font-style:normal}
.mb-eq strong{color:var(--accent-l);font-weight:600}

.typing{display:flex;gap:4px;padding:4px 0;align-items:center}
.tdot{width:5px;height:5px;background:var(--tx-3);animation:bnc 1.2s infinite}
@keyframes bnc{0%,80%,100%{transform:translateY(0);opacity:.4}40%{transform:translateY(-4px);opacity:1}}

.cb-input{padding:10px 14px 14px;border-top:1px solid var(--bd);margin-top:auto;background:var(--bg-1);flex-shrink:0}
.cb-input--empty{border-top:none}
.cb-input-wrap{display:flex;gap:8px;align-items:flex-end;border:1px solid var(--bd);padding:7px 7px 7px 13px;background:var(--bg-2);transition:border-color .15s}
.cb-input-wrap:focus-within{border-color:var(--accent)}
.cb-ta{flex:1;border:none;outline:none;resize:none;font-size:13px;line-height:1.5;color:var(--tx-1);background:transparent;font-family:inherit;max-height:100px;overflow-y:auto}
.cb-ta::placeholder{color:var(--tx-3)}
.cb-send{width:30px;height:30px;border:none;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s;cursor:pointer}
.cb-send:not(:disabled){background:var(--accent);color:var(--bg-0)}
.cb-send:not(:disabled):hover{background:var(--accent-l)}
.cb-send:disabled{background:var(--bg-3);color:var(--tx-3);cursor:default}
      `}</style>
    </div>
  );
}