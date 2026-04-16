# Move Like Buffett

A full-stack investment analysis platform that applies Warren Buffett's value investing principles to any publicly traded company. Features a RAG-powered AI chatbot trained on Buffett's letters, speeches, and essays, alongside real-time stock analysis with Buffett-style financial metrics.

---

## Tech Stack

**Backend** — Python 3.11+ / FastAPI / Uvicorn  
**Frontend** — React 19 / Vite 8 / Recharts / Lucide Icons  
**AI/RAG** — DeepSeek LLM / Sentence-Transformers / FAISS  
**Data** — yfinance / Yahoo Finance v8 / Twelve Data API  

---

## Project Structure

```
move-like-buffett/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # All configuration (reads .env)
│   ├── requirements.txt
│   ├── build_indices.py         # One-time FAISS index builder
│   ├── rag/
│   │   ├── agent.py             # BuffettAgent — orchestrates RAG pipeline
│   │   ├── researcher.py        # Multi-source retrieval + citation
│   │   ├── retriever.py         # FAISS similarity search
│   │   ├── router.py            # Query classification
│   │   └── indexer.py           # Index builder
│   ├── stock/
│   │   ├── cache.py             # Data fetching + caching + fallback chain
│   │   ├── analysis.py          # Buffett ratios, DCF, Altman Z, Piotroski
│   │   └── endpoints.py         # FastAPI routes for stock analysis
│   └── data/
│       ├── csvs/                # QA pairs from Buffett sources
│       ├── chunks/              # Text chunks from source documents
│       ├── indices/             # FAISS index files (auto-generated)
│       └── cache/               # Persistent stock data cache (auto-generated)
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Router — Chat vs Dashboard
│   │   ├── ChatBot.jsx          # RAG chatbot interface
│   │   └── Dashboard.jsx        # Stock analysis dashboard
│   ├── package.json
│   └── vite.config.js           # Dev proxy → backend:8000
└── pipeline/                    # Data processing notebooks (offline)
```

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm**
- **DeepSeek API key** — [platform.deepseek.com](https://platform.deepseek.com)
- **Twelve Data API key** (free) — [twelvedata.com/register](https://twelvedata.com/register)

---

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd move-like-buffett
```

### 2. Backend

```bash
cd backend
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment variables

Create `backend/.env`:

```env
DEEPSEEK_API_KEY=your-deepseek-key-here
TD_API_KEY=your-twelvedata-key-here
```

That's the minimum. Full options:

```env
# DeepSeek LLM
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Twelve Data (free fallback for quotes + price history)
TD_API_KEY=your-twelvedata-key
TD_DAILY_BUDGET=800
TD_RESERVE_BUDGET=50

# Server
CORS_ORIGINS=http://localhost:5173
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

### 4. Build FAISS indices (one-time)

```bash
cd backend
python build_indices.py
```

This reads the QA CSVs and text chunks, builds the FAISS vector indices, and saves them to `data/indices/`. Only needs to run once unless source data changes.

### 5. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

### 7. Open the app

```
http://localhost:5173
```

---

## Features

### AI Chatbot
Ask any question about Warren Buffett's investment philosophy. The RAG pipeline retrieves relevant passages from his shareholder letters (1957-2012), Cunningham's essays, the Florida speech, Notre Dame lectures, and the Ivey 2008 Q&A, then generates a cited answer via DeepSeek.

### Stock Analysis Dashboard
Enter any ticker symbol to get:

| Analysis | Description |
|---|---|
| **Company Info** | Sector, industry, market cap, PE, beta |
| **Buffett Ratios** | Gross margin, ROE, debt/equity, FCF yield, EPS consistency |
| **DCF Valuation** | Discounted cash flow with margin of safety |
| **Altman Z-Score** | Bankruptcy probability assessment |
| **Piotroski F-Score** | Financial strength (0-9 scale) |
| **Technical Analysis** | RSI, MACD, Bollinger Bands, moving averages |
| **Monte Carlo** | Price simulation with confidence intervals |
| **Peer Clustering** | K-means comparison against sector peers |

---

## Data Architecture

Stock data flows through a three-tier fallback system:

```
Tier 1:  yfinance            ← primary (all data types, rate-limited)
Tier 2:  Yahoo v8 chart API  ← prices only (no API key, shares IP ban)
Tier 3:  Twelve Data API     ← quotes + price history only (800 req/day free)
Tier 4:  Disk cache           ← stale data served when all live sources fail
```

**Important:** Company info and financial statements are only available from yfinance. No free API provides fundamentals anymore. The cache stores statements for 24 hours, so after one successful fetch, temporary rate limits don't affect the app.

**If you see 502 errors on first load:** yfinance is rate-limited. Restart your router to get a new IP, then reload. After the first successful fetch, data is cached and rate limits won't cause errors for hours.

---

## API Endpoints

All stock endpoints are under `/stock/`. The Vite dev server proxies these automatically.

| Endpoint | Method | Description |
|---|---|---|
| `/stock/quotes?symbols=AAPL,MSFT` | GET | Batch price quotes |
| `/stock/info?symbol=GOOGL` | GET | Company profile + metrics |
| `/stock/financials?symbol=GOOGL` | GET | Income, balance sheet, cash flow |
| `/stock/buffett-ratios?symbol=GOOGL` | GET | Buffett-style ratio analysis |
| `/stock/dcf?symbol=GOOGL` | GET | DCF intrinsic value estimate |
| `/stock/altman-z?symbol=GOOGL` | GET | Altman Z-Score |
| `/stock/piotroski?symbol=GOOGL` | GET | Piotroski F-Score |
| `/stock/technical?symbol=GOOGL` | GET | Technical indicators |
| `/stock/monte-carlo?symbol=GOOGL` | GET | Monte Carlo simulation |
| `/stock/cluster?symbol=GOOGL` | GET | Peer cluster analysis |
| `/stock/health` | GET | Yahoo ban status + TD budget |
| `/stock/clear-cache` | POST | Clear cached data |
| `/chat` | POST | RAG chatbot (streaming) |

---

## Troubleshooting

**502 Bad Gateway on stock endpoints**  
yfinance is rate-limited by Yahoo. Restart your router for a new IP. The app auto-recovers once the ban lifts (typically 15-60 minutes).

**Ticker bar shows prices but analysis tabs fail**  
Quotes use Yahoo v8 (separate rate pool). Fundamentals need yfinance. Wait for the ban to clear or restart router.

**"TD budget exhausted"**  
You've used all 800 Twelve Data API calls for the day. Resets at midnight. Quotes and price history fall back to Yahoo v8.

**FAISS index errors on startup**  
Run `python build_indices.py` from the backend directory. The indices must be built before the first launch.

**Chat returns empty or generic responses**  
Verify `DEEPSEEK_API_KEY` is set in `.env` and the key is active at [platform.deepseek.com](https://platform.deepseek.com).

---

## Knowledge Sources

The RAG chatbot is trained on:

- Buffett Partnership Letters (1957-1970)
- Berkshire Hathaway Shareholder Letters (2000-2012)
- Lawrence Cunningham's *The Essays of Warren Buffett*
- University of Florida Speech (1998)
- Notre Dame Lectures
- Ivey Business School Q&A (2008)

---

## License

For academic and portfolio demonstration purposes.