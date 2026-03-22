import json
import os
import re
import sqlite3
import requests
import pandas as pd
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from ddgs import DDGS
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction, Embeddings

load_dotenv()

client = Groq()

# ── Data folder structure ─────────────────────────────────────────────────────
DATA_DIR    = "data"
DB_DIR      = os.path.join(DATA_DIR, "db")
PDF_DIR     = os.path.join(DATA_DIR, "pdf")
JSON_DIR    = os.path.join(DATA_DIR, "json")
EXPORT_DIR  = os.path.join(DATA_DIR, "exports")
VECTOR_DIR  = os.path.join(DATA_DIR, "vector")

for _dir in [DB_DIR, PDF_DIR, JSON_DIR, EXPORT_DIR, VECTOR_DIR]:
    os.makedirs(_dir, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "data_agent.db")

# ── Vector DB setup (multilingual model — Hebrew + English) ───────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_chroma_client     = None
_chroma_collection = None
_embed_fn          = None

class MultilingualEmbedder(EmbeddingFunction):
    """Wraps sentence-transformers for ChromaDB — supports Hebrew + English."""
    def __init__(self):
        print(f"[RAG] Loading multilingual model '{EMBEDDING_MODEL}' (first time only)...")
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def __call__(self, input: list) -> Embeddings:
        return self._model.encode(input, normalize_embeddings=True).tolist()

def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = MultilingualEmbedder()
    return _embed_fn

def _get_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)
        # Use a versioned collection name so old English-only index is replaced
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="companies_v2",
            embedding_function=_get_embed_fn(),
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection

# ── Tool implementations ──────────────────────────────────────────────────────

def fetch_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Return clean text instead of raw HTML
        text = soup.get_text(separator="\n", strip=True)
        return text[:3000]
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

def search_web(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error searching: {str(e)}"

def parse_table(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return "No tables found"
        all_tables = []
        for i, table in enumerate(tables):
            headers_row = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if cells:
                    rows.append(cells)
            all_tables.append({"table_index": i, "headers": headers_row, "rows": rows[:20]})
        return json.dumps(all_tables, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error parsing table: {str(e)}"

def read_file(filename: str) -> str:
    try:
        files = os.listdir(JSON_DIR)
        match = next((f for f in files if filename in f), None)
        if not match:
            return f"File not found. Available: {', '.join(files)}"
        with open(os.path.join(JSON_DIR, match), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def read_pdf(filename: str) -> str:
    """Extract text from a PDF in data/pdf/"""
    try:
        files = os.listdir(PDF_DIR)
        match = next((f for f in files if filename.lower() in f.lower()), None)
        if not match:
            return f"PDF not found. Available: {', '.join(files) or 'none'}"
        path = os.path.join(PDF_DIR, match)
        text_pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_pages.append(f"[Page {i+1}]\n{text}")
        full_text = "\n\n".join(text_pages)
        return full_text[:5000] if full_text else "No text found in PDF"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def list_data_files() -> str:
    """Show all files organized by type"""
    result = {}
    for label, folder in [("json", JSON_DIR), ("pdf", PDF_DIR), ("exports", EXPORT_DIR)]:
        files = os.listdir(folder)
        result[label] = files if files else []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    result["db_tables"] = [r[0] for r in cursor.fetchall()]
    conn.close()
    return json.dumps(result, ensure_ascii=False, indent=2)

def build_vector_index(table_name: str = None) -> str:
    """Embed company descriptions from DB tables into ChromaDB vector store."""
    try:
        collection = _get_collection()

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()

        if table_name:
            tables = [table_name]
        else:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall() if r[0] != "sqlite_sequence"]

        total_added = 0
        for tbl in tables:
            cur.execute(f'PRAGMA table_info("{tbl}")')
            cols = [r[1] for r in cur.fetchall()]

            name_col = next((c for c in cols if c.lower() in
                             ["short name", "name", "company english name", "company hebrew name"]), None)
            desc_col = next((c for c in cols if "description" in c.lower()), None)

            if not name_col or not desc_col:
                continue

            cur.execute(f'SELECT "{name_col}", "{desc_col}" FROM "{tbl}"')
            rows = cur.fetchall()

            docs, ids, metas = [], [], []
            for row in rows:
                name = str(row[0] or "").strip()
                desc = str(row[1] or "").strip()
                if not name or not desc or desc == "None":
                    continue
                import hashlib
                doc_id = hashlib.md5(f"{tbl}_{name}".encode()).hexdigest()
                docs.append(f"{name}: {desc}")
                ids.append(doc_id)
                metas.append({"name": name, "table": tbl, "description": desc[:500]})

            if not docs:
                continue

            # Upsert in batches of 100 — ChromaDB handles embeddings internally
            for i in range(0, len(docs), 100):
                collection.upsert(
                    ids=ids[i:i+100],
                    documents=docs[i:i+100],
                    metadatas=metas[i:i+100]
                )
                print(f"  [RAG] Indexed {min(i+100, len(docs))}/{len(docs)} from '{tbl}'")
            total_added += len(docs)

        conn.close()
        return f"Vector index built: {total_added} documents indexed from {len(tables)} table(s)"
    except Exception as e:
        return f"Error building vector index: {str(e)}"


def _is_hebrew(text: str) -> bool:
    """Returns True if the text contains Hebrew characters."""
    return any('\u05d0' <= c <= '\u05ea' for c in text)


def _translate_to_english(hebrew_query: str) -> str:
    """Translate a Hebrew query to English using the LLM."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a translator. Translate the user's Hebrew search query to English. Output ONLY the English translation — no explanations, no quotes, no punctuation at the end."},
            {"role": "user", "content": hebrew_query}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def semantic_search(query: str, n_results: int = 10) -> str:
    """Search company descriptions by meaning using vector similarity.
    Automatically translates Hebrew queries to English before searching."""
    try:
        collection = _get_collection()

        if collection.count() == 0:
            return "Vector index is empty. Run build_vector_index first."

        # Auto-translate Hebrew queries
        search_query = query
        translation_note = ""
        if _is_hebrew(query):
            search_query = _translate_to_english(query)
            translation_note = f"[🔤 Translated: \"{query}\" → \"{search_query}\"]"
            print(f"\n{translation_note}")

        results = collection.query(
            query_texts=[search_query],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = round((1 - dist) * 100, 1)
            output.append({
                "rank": i + 1,
                "name": meta.get("name", ""),
                "table": meta.get("table", ""),
                "similarity_%": similarity,
                "description": meta.get("description", "")[:200]
            })

        result_json = json.dumps(output, ensure_ascii=False, indent=2)
        if translation_note:
            return f"{translation_note}\n{result_json}"
        return result_json

    except Exception as e:
        return f"Error in semantic search: {str(e)}"


def import_xlsx(filename: str, table_name: str = None, header_row: int = 0) -> str:
    """Import an xlsx/csv file from data/csv/ directly into SQLite"""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        CSV_DIR = os.path.join(DATA_DIR, "csv")
        files = os.listdir(CSV_DIR)
        match = next((f for f in files if filename.lower() in f.lower()), None)
        if not match:
            return f"File not found. Available: {', '.join(files)}"
        path = os.path.join(CSV_DIR, match)
        ext = os.path.splitext(match)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, header=header_row)
        else:
            df = pd.read_csv(path, header=header_row)
        # Clean up
        df = df.dropna(how='all').reset_index(drop=True)
        df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]
        df.columns = [str(c).strip() for c in df.columns]
        # Table name = filename without extension if not provided
        if not table_name:
            table_name = os.path.splitext(match)[0].lower().replace(" ", "_").replace("-", "_")[:40]
        conn = sqlite3.connect(DB_PATH)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        # Auto-build vector index for this table
        build_vector_index(table_name)
        return f"Imported {len(df)} rows, {len(df.columns)} columns into table '{table_name}'\nColumns: {list(df.columns)}"
    except Exception as e:
        return f"Error importing file: {str(e)}"

def export_table(table_name: str, fmt: str = "json") -> str:
    """Export a DB table to data/exports/ as JSON or CSV"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM "{table_name}"')
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if fmt == "csv":
            import csv
            filepath = os.path.join(EXPORT_DIR, f"{table_name}_{timestamp}.csv")
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            filepath = os.path.join(EXPORT_DIR, f"{table_name}_{timestamp}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        return f"Exported {len(rows)} rows to {filepath}"
    except Exception as e:
        return f"Error exporting: {str(e)}"

def save_data(filename: str, data) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(JSON_DIR, f"{filename}_{timestamp}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"Saved to: {filepath}"

def save_to_db(table_name: str, records: list) -> str:
    try:
        if not records:
            return "No records to save"
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        flat_records = []
        for rec in records:
            flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in rec.items()}
            flat_records.append(flat)
        columns = list(flat_records[0].keys())
        cols_def = ", ".join(f'"{c}" TEXT' for c in columns)
        cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols_def})')
        cols_str = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        rows = [tuple(r.get(c, "") for c in columns) for r in flat_records]
        cursor.executemany(f'INSERT INTO "{table_name}" ({cols_str}) VALUES ({placeholders})', rows)
        conn.commit()
        conn.close()
        return f"Inserted {len(rows)} records into '{table_name}'"
    except Exception as e:
        return f"Error saving to DB: {str(e)}"

def list_tables() -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            return "Database is empty"
        result = {}
        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
            cols = [row[1] for row in cursor.fetchall()]
            cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            count = cursor.fetchone()[0]
            result[table] = {"columns": cols, "row_count": count}
        conn.close()
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error listing tables: {str(e)}"

def query_db(question: str) -> str:
    try:
        schema_info = list_tables()
        sql_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are a SQL expert. Schema:\n{schema_info}\nReturn ONLY a valid SQLite SELECT query, nothing else."},
                {"role": "user", "content": question}
            ]
        )
        sql = sql_response.choices[0].message.content.strip().strip("```sql").strip("```").strip()
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return json.dumps({"sql": sql, "results": rows}, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error querying DB: {str(e)}"


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "search_web": {
        "fn": lambda args: search_web(args["query"], args.get("max_results", 5)),
        "desc": 'search_web(query, max_results=5) → Search DuckDuckGo'
    },
    "fetch_url": {
        "fn": lambda args: fetch_url(args["url"]),
        "desc": 'fetch_url(url) → Get clean text from a webpage'
    },
    "parse_table": {
        "fn": lambda args: parse_table(args["url"]),
        "desc": 'parse_table(url) → Extract tables from a webpage'
    },
    "save_to_db": {
        "fn": lambda args: save_to_db(args["table_name"], args["records"]),
        "desc": 'save_to_db(table_name, records) → Save list of dicts to SQLite (data/db/)'
    },
    "list_tables": {
        "fn": lambda args: list_tables(),
        "desc": 'list_tables() → Show all DB tables and row counts'
    },
    "query_db": {
        "fn": lambda args: query_db(args["question"]),
        "desc": 'query_db(question) → Ask a natural language question about stored data'
    },
    "save_data": {
        "fn": lambda args: save_data(args["filename"], args["data"]),
        "desc": 'save_data(filename, data) → Save JSON file to data/json/'
    },
    "read_file": {
        "fn": lambda args: read_file(args["filename"]),
        "desc": 'read_file(filename) → Read a JSON file from data/json/'
    },
    "read_pdf": {
        "fn": lambda args: read_pdf(args["filename"]),
        "desc": 'read_pdf(filename) → Extract text from a PDF in data/pdf/'
    },
    "list_data_files": {
        "fn": lambda args: list_data_files(),
        "desc": 'list_data_files() → Show all files and DB tables in the data/ folder'
    },
    "export_table": {
        "fn": lambda args: export_table(args["table_name"], args.get("format", "json")),
        "desc": 'export_table(table_name, format="json"|"csv") → Export DB table to data/exports/'
    },
    "import_xlsx": {
        "fn": lambda args: import_xlsx(args["filename"], args.get("table_name"), args.get("header_row", 0)),
        "desc": 'import_xlsx(filename, table_name, header_row=0) → Import xlsx/csv from data/csv/ into SQLite'
    },
    "semantic_search": {
        "fn": lambda args: semantic_search(args["query"], args.get("n_results", 10)),
        "desc": 'semantic_search(query, n_results=10) → Search companies by meaning (not just keywords) using vector similarity'
    },
    "build_vector_index": {
        "fn": lambda args: build_vector_index(args.get("table_name")),
        "desc": 'build_vector_index(table_name=None) → Embed all company descriptions into vector DB (run once, or after new imports)'
    },
}


# ── ReAct Agent loop ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a data collection and analysis agent with access to a local SQLite database and the web.

Available tools:
{tools}

To use a tool, output EXACTLY this JSON (nothing before or after):
{{"tool": "tool_name", "args": {{"arg1": "value1"}}}}

When you have finished and have an answer for the user, output EXACTLY:
{{"done": true, "answer": "your final answer here"}}

## Decision rules — follow this order every time:

1. ALWAYS check the DB context provided before taking any action.
2. For conceptual or descriptive questions ("companies that solve X", "who works on Y") → use semantic_search FIRST.
3. For structured queries (filters, counts, specific fields) → use query_db.
4. Only use search_web or fetch_url if:
   - The database has no relevant data, AND
   - The user explicitly asks for new/live information from the internet.
5. After fetching new data from the web → always save it with save_to_db for future use.
6. If vector index seems empty or outdated → call build_vector_index before semantic_search.

## Output rules:
- Output ONLY valid JSON, one object per response
- Never explain yourself before calling a tool
- After getting tool results, decide the next step based on the rules above
""".strip()


def run_agent(user_request: str) -> str:
    import time
    tools_desc = "\n".join(f"  - {v['desc']}" for v in TOOL_REGISTRY.values())
    system = SYSTEM_PROMPT.format(tools=tools_desc)

    # Inject current DB state so agent knows what's available before step 1
    db_context = list_tables()
    if db_context and db_context != "Database is empty":
        db_hint = f"[Current database tables]\n{db_context}"
    else:
        db_hint = "[Database is currently empty]"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{db_hint}\n\nUser request: {user_request}"}
    ]

    tool_call_counts = {}  # track how many times each tool is called

    for step in range(10):
        # Retry loop for rate limit
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 15 * (attempt + 1)
                    print(f"[Rate limit hit — waiting {wait}s...]")
                    time.sleep(wait)
                else:
                    return f"Error: {str(e)}"

        raw = response.choices[0].message.content.strip()

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            return raw  # Plain text answer

        try:
            action = json.loads(json_match.group())
        except json.JSONDecodeError:
            return raw

        if action.get("done"):
            return action.get("answer", raw)

        tool_name = action.get("tool")
        tool_args = action.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            return f"Unknown tool: {tool_name}"

        # Block repetitive tool calls (max 2 per tool)
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
        if tool_call_counts[tool_name] > 2:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"You already used '{tool_name}' {tool_call_counts[tool_name]-1} times. Use the results you have or finish with a done answer."})
            continue

        print(f"\n[Step {step+1}] {tool_name}({', '.join(f'{k}={repr(v)[:50]}' for k,v in tool_args.items())})")
        result = TOOL_REGISTRY[tool_name]["fn"](tool_args)

        # Keep context lean — only last result in detail
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Tool result:\n{str(result)[:1500]}"})

    return "Max steps reached"


# ── Test run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test 1: Collect & save to DB ===")
    result = run_agent(
        "Search for 'top programming languages 2025', get 5 results, and save them to the database in a table called 'search_results'"
    )
    print("\n--- Response ---\n", result)

    print("\n\n=== Test 2: Ask a question about the DB ===")
    result = run_agent("What data do we have in the database? List the tables and show me the first few rows.")
    print("\n--- Response ---\n", result)
