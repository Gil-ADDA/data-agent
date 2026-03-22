# Data Agent

An AI-powered data collection, enrichment, and querying agent — built with a ReAct architecture on top of Groq + LLaMA 3.3 70B.

The agent can search the web, scrape pages, read PDFs, import Excel files, store everything in a local SQLite database, and answer natural language questions about the data — all through a simple CLI.

---

## What Was Built

### Phase 1 — Basic LLM Agent
A minimal agent that sends user requests to an LLM and returns a text response.
No tools, no memory, no data storage. Proof of concept only.

### Phase 2 — ReAct Agent with Tools
Replaced the basic call with a **ReAct loop** (Reasoning + Acting):
- The LLM outputs a JSON action (`tool` + `args`)
- The agent executes the tool and feeds the result back
- Loop continues until the LLM outputs `{"done": true, "answer": "..."}`

Tools added in this phase:
| Tool | What it does |
|------|-------------|
| `search_web` | DuckDuckGo search, returns top N results |
| `fetch_url` | Fetches a webpage and returns clean text |
| `parse_table` | Extracts HTML tables from a URL |
| `save_data` | Saves any JSON to `data/json/` |
| `read_file` | Reads a JSON file from `data/json/` |

### Phase 3 — SQLite Database Layer
Added persistent structured storage so data survives between sessions:
| Tool | What it does |
|------|-------------|
| `save_to_db` | Saves a list of records into a named SQLite table |
| `list_tables` | Shows all tables in the DB with column names and row counts |
| `query_db` | Accepts a natural language question → generates SQL → returns results |
| `export_table` | Exports any DB table to CSV or JSON in `data/exports/` |

### Phase 4 — File Import & PDF Reading
| Tool | What it does |
|------|-------------|
| `import_xlsx` | Imports `.xlsx` or `.csv` from `data/csv/` into SQLite (handles metadata header rows) |
| `read_pdf` | Extracts text from any PDF in `data/pdf/` using pdfplumber |
| `list_data_files` | Shows all files across json/, pdf/, exports/, and DB tables |

### Phase 5 — Agent Reliability Fixes
Two bugs found and fixed during 10-test evaluation:
- **Rate limit handling** — retry loop with 15s/30s waits on Groq 429 errors
- **Tool repetition guard** — each tool limited to 2 calls per request to prevent loops

### Phase 6 — DB-First Decision Logic
Rewrote `SYSTEM_PROMPT` with explicit decision rules:
1. Always check DB first before going to the web
2. If relevant tables exist → use `query_db`, not `search_web`
3. Only go to the web if DB has no relevant data
4. After fetching from web → always save with `save_to_db`

Additionally: the DB schema (all tables + columns + row counts) is **injected automatically** into every request before the agent starts, so it knows what data is available without needing to call `list_tables` first.

---

## Folder Structure

```
data-agent/
├── agent_with_tool.py   # Core agent + all 12 tools + ReAct loop
├── cli.py               # CLI interface (interactive + one-shot mode)
├── agent.py             # Early prototype (no tools — kept for reference)
├── data/
│   ├── csv/             # Drop xlsx/csv files here for import
│   ├── db/              # SQLite database (local only, not in git)
│   ├── pdf/             # Drop PDF files here (local only, not in git)
│   ├── json/            # Intermediate JSON data (local only)
│   └── exports/         # Exported tables as CSV/JSON
└── .env                 # API keys (never committed)
```

---

## Requirements

### API Keys
```
GROQ_API_KEY=your_key_here   # Free tier: 100K tokens/day
```

### Python Dependencies
```bash
pip install groq python-dotenv requests beautifulsoup4 ddgs pdfplumber pandas openpyxl
```

### Python Version
Python 3.9+

### Full Setup
```bash
git clone https://github.com/Gil-ADDA/data-agent.git
cd data-agent
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install groq python-dotenv requests beautifulsoup4 ddgs pdfplumber pandas openpyxl
echo "GROQ_API_KEY=your_key_here" > .env
python cli.py
```

---

## Usage

**Interactive mode:**
```bash
python cli.py
```

**One-shot mode:**
```bash
python cli.py "Search for top AI companies in Israel and save to database"
python cli.py "What data do we have in the database?"
python cli.py "Import companies.xlsx and show me the first 5 rows"
python cli.py "Read report.pdf and save key data to the database"
python cli.py "Export the results table to CSV"
```

**Import an Excel file:**
Place the file in `data/csv/` then:
```bash
python cli.py "Import myfile.xlsx into a table called my_table"
```
If the file has metadata rows before the actual headers, specify:
```bash
python cli.py "Import myfile.xlsx with header at row 9"
```

---

## Architecture — ReAct Loop

```
User request
    │
    ▼
[Auto-inject DB schema] — all tables + columns sent to LLM upfront
    │
    ▼
┌─────────────────────────────────────┐
│           ReAct Loop (max 10 steps) │
│                                     │
│  LLM → JSON action                  │
│    ├── tool call → execute → result │
│    │      └── feed result back      │
│    └── {"done": true} → return      │
└─────────────────────────────────────┘
    │
    ▼
Decision rules (enforced in system prompt):
  1. Data in DB? → query_db (no web search)
  2. No data?   → search_web → save_to_db → answer
  3. Each tool: max 2 calls per request
  4. Rate limit: auto-retry with backoff
```

---

## Is This RAG?

**Not quite — but close.**

| Capability | Current System | Full RAG |
|---|---|---|
| Data storage | SQLite (structured) | Vector DB (Chroma, Pinecone) |
| Search method | SQL `LIKE` keyword match | Semantic / cosine similarity |
| Finds by meaning | ❌ | ✅ |
| Finds by exact keyword | ✅ | ✅ |
| Works on PDFs/docs | ✅ partial | ✅ |
| Grounded answers | ✅ | ✅ |

**The gap:** searching `LIKE "%fleet%"` misses companies whose description says *"vehicle operations optimization"* — same concept, different words. RAG solves this with embeddings.

---

## Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| Groq free tier: 100K tokens/day | Rate limit on heavy use | Upgrade plan or switch to Claude API |
| SQL search is keyword-only | Misses semantic matches | Add vector search (see Roadmap) |
| PDF parsing: text-based only | Scanned PDFs not supported | Use OCR (Tesseract / AWS Textract) |
| No memory between sessions | Agent forgets past conversations | Add conversation history to DB |
| LLM sometimes ignores rules | Occasional wrong tool choice | Tune system prompt or switch model |

---

## Roadmap — Build Phases

### ✅ Phase 7 — RAG / Semantic Search (COMPLETED)
- ChromaDB as persistent vector store (`data/vector/`)
- `paraphrase-multilingual-MiniLM-L12-v2` — supports Hebrew + English
- Auto-translation: Hebrew queries → English before search (transparent)
- 1,462 documents indexed from 12 DB tables
- Auto-index rebuild on every `import_xlsx`
- Tools added: `semantic_search`, `build_vector_index`

### Phase 8 — Web UI (Streamlit)
Replace CLI with a browser-based chat interface:
- Chat window with conversation history
- File upload (drag & drop xlsx/pdf)
- Table viewer for query results with export button
- Sidebar showing DB tables and vector index status

### Phase 9 — Scheduled Data Collection
Keep data fresh automatically:
- Cron-like jobs to re-scrape sources on a schedule
- Detect new companies vs. existing ones (delta updates)
- Weekly email/Slack summary report

### Phase 10 — Multi-domain Knowledge Base
Extend beyond autotech:
- Smart Cities, Smart Transportation, Advanced Industry, VC Funds
- Each domain: dedicated DB schema + vector collection
- Domain routing: agent detects which domain the question is about

### Phase 11 — Claude API Integration
Replace Groq/LLaMA with Claude (Anthropic) for:
- Higher quality reasoning and instruction following
- Longer context window (better for large PDFs)
- Native tool-use API (no JSON parsing workaround)
- Multilingual understanding without translation step

---

## Continuous Improvement — How to Get Better Results

The system is designed to be improved iteratively. If results are not satisfactory, follow this process:

### Step 1 — Diagnose the problem
| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Results are irrelevant | Query too vague | Use more specific English terms |
| Missing companies | Not in DB | Import missing data source |
| Wrong companies returned | Descriptions too short | Enrich DB with fuller descriptions |
| Similarity scores too low (<50%) | Topic not well represented | Add more data on that topic |
| Semantic search and SQL disagree | Use both and compare | Run `query_db` + `semantic_search` and merge |

### Step 2 — Improve the data
```bash
# Add new xlsx/pdf files to data/csv/ or data/pdf/
python cli.py "Import newfile.xlsx into table new_table"
# Index is rebuilt automatically
```

### Step 3 — Improve the query
```bash
# Bad (too short):
python cli.py "fleet companies"

# Good (descriptive):
python cli.py "companies that provide fleet management, vehicle tracking, and telematics solutions for commercial vehicles"
```

### Step 4 — Tune search results
```bash
# Increase result count to see more candidates
python cli.py "semantic_search fleet management 20"

# Combine with SQL filter for precision
python cli.py "from semantic search results on fleet management, show only companies with more than 50 employees"
```

### Step 5 — Upgrade the model (when ready)
Current model: `paraphrase-multilingual-MiniLM-L12-v2` (fast, local, free)

Better alternatives (require more RAM/compute):
| Model | Size | Quality | Languages |
|-------|------|---------|-----------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 120MB | Good | 50+ |
| `paraphrase-multilingual-mpnet-base-v2` | 280MB | Better | 50+ |
| OpenAI `text-embedding-3-small` | API | Best | All |
| Claude Embeddings | API | Best | All |

To switch model: change `EMBEDDING_MODEL` in `agent_with_tool.py`, delete `data/vector/`, run `build_vector_index`.

---

## Recommendations

1. **Switch to Claude API** — better reasoning, longer context, native tools (no JSON workaround)
2. **Build Streamlit UI** — makes the tool accessible without terminal
3. **Standardize DB schema per domain** — define fixed columns per sector for cleaner queries
4. **Add `imported_at` timestamp** — track when data was added for freshness monitoring
5. **Run `build_vector_index` after every manual DB change** — keeps semantic search in sync
6. **If results are bad → diagnose before changing code** — follow the Continuous Improvement steps above
