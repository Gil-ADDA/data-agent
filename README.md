# Data Agent

AI agent for web data collection, storage, and querying — powered by Groq + LLaMA.

## Features

- **Web search** via DuckDuckGo
- **Web scraping** — fetch URLs, extract tables from HTML
- **PDF reading** — extract text from PDF files
- **SQLite database** — auto-creates tables, stores structured data
- **Natural language queries** — ask questions about your data in plain language
- **Export** — save tables to JSON or CSV
- **CLI** — interactive or one-shot mode

## Folder structure

```
data-agent/
├── agent_with_tool.py   # Core agent + all tools
├── cli.py               # CLI interface
├── data/
│   ├── db/              # SQLite database
│   ├── pdf/             # Drop PDF files here
│   ├── json/            # Collected JSON data
│   └── exports/         # Exported tables (CSV/JSON)
└── .env                 # GROQ_API_KEY (not committed)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install groq python-dotenv requests beautifulsoup4 ddgs pdfplumber
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

## Usage

**Interactive mode:**
```bash
python cli.py
```

**One-shot mode:**
```bash
python cli.py "Search for top AI companies in Israel and save to database"
python cli.py "What data do we have in the database?"
python cli.py "Read report.pdf and save key data to the database"
python cli.py "Export the results table to CSV"
```

---

## Architecture — Is this RAG?

### What we built (RAG-adjacent)

The agent follows a **DB-first ReAct loop**:

```
User request
    │
    ▼
[DB context injected] ── list of all tables sent to LLM before step 1
    │
    ▼
LLM decides:
    ├── Data exists in DB? → query_db (SQL via natural language)
    └── No data? → search_web → save_to_db → answer
```

**Tools available:** `search_web`, `fetch_url`, `parse_table`, `read_pdf`,
`save_to_db`, `query_db`, `list_tables`, `import_xlsx`, `export_table`, `save_data`, `read_file`, `list_data_files`

### What RAG adds (next step)

**RAG = Retrieval-Augmented Generation** — instead of keyword SQL search,
the agent finds documents by *meaning*, not exact words.

| | Current system | Full RAG |
|---|---|---|
| Search | `WHERE description LIKE "%fleet%"` | Semantic vector search |
| Finds | Exact keyword matches only | Conceptually similar results |
| Storage | SQLite | Vector DB (Chroma, Pinecone) |
| "Vehicle ops optimization" matches "fleet"? | ❌ | ✅ |

### Steps to upgrade to full RAG

**Step 1 — Embeddings**
- Take every company description
- Convert to a vector (array of numbers representing meaning)
- Store in a vector database (e.g. ChromaDB locally)

**Step 2 — Semantic Search**
- Incoming question → convert to vector
- Find the N descriptions closest in meaning (cosine similarity)

**Step 3 — Generate**
- Pass the retrieved results to the LLM → grounded answer

**Tools needed:**
```bash
pip install chromadb sentence-transformers
```
