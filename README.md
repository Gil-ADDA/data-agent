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
