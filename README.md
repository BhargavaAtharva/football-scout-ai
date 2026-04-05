# ⚽ Football Scout AI

A natural language interface for scouting and querying FIFA 24 player data — powered by a hybrid RAG + Pandas query engine and Llama 3.3 70B.

![Stack](https://img.shields.io/badge/LLM-Llama%203.3%2070B-blue) ![Stack](https://img.shields.io/badge/Vector%20DB-ChromaDB-green) ![Stack](https://img.shields.io/badge/Frontend-React-cyan)

---

## What It Does

Instead of manually filtering spreadsheets, you ask questions in plain English:

- *"Who has the best dribbling?"*
- *"Find me a creative midfielder who can press hard"*
- *"Top 5 players by finishing"*
- *"Best goalkeepers by reflexes"*

The system intelligently routes each query to the right engine and returns a grounded answer.

---

## Architecture
```
User Query
    ↓
LLM Classifier (Groq)
    ↓
STAT query          SEMANTIC query
    ↓                     ↓
Pandas nlargest     RAG Pipeline
    ↓                     ↓
Top 5 players       ChromaDB retrieval
                          ↓
                    Llama 3.3 70B
                          ↓
                    Grounded answer
```

**The key engineering decision:** RAG alone fails for stat-based queries like "who has the highest dribbling" because vector similarity doesn't compare numerical values. The hybrid router solves this by detecting query intent first and using the right tool for each query type.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Backend | Flask + Python |
| Frontend | React |
| Dataset | FIFA 24 — 5,682 players, 41 attributes |

---

## Known Limitations

- **Named entity queries** like "Messi vs Ronaldo" don't work reliably — vector search retrieves semantically similar chunks rather than specific named players. Fix: add a third route that extracts player names and fetches them directly from the dataframe.
- **Vocabulary mismatch** — querying "PSG" won't match "Paris Saint-Germain" in the embeddings. Fix: abbreviation expansion at preprocessing time.
- **ChromaDB is local only** — not suitable for production scale. Would be replaced with Qdrant or Pinecone for deployment.

---

## Evaluation (RAGAS)
| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Answer Relevancy | 0.37 |
| Context Recall | 0.75 |

Answer relevancy is lower due to terse single-word answers. 
Context recall failure on stat-based queries is expected — 
these are routed to pandas in production use.

## Setup
```bash
# 1. Clone the repo
git clone https://github.com/BhargavaAtharva/football-scout-ai
cd football-scout-ai

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
export GROQ_API_KEY="your-key-here"

# 5. Run the backend
python api.py

# 6. In a new terminal, run the frontend
cd frontend
npm install
npm start
```

---

## Project Structure
```
football-scout-ai/
├── scout.py          # Core RAG + hybrid query engine
├── api.py            # Flask REST API
├── player_stats.csv  # FIFA 24 dataset
└── frontend/
    └── src/
        └── App.js    # React chat interface
```