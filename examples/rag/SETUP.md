# Complete Setup Guide for LangGraph RAG Example

This example uses **PgDistRagRetriever** from PR #1 (feature/pg-dist-rag-retriever branch) of langchain-yugabytedb.

## Prerequisites

You need a YugabyteDB instance with:
- `pg_dist_rag` extension enabled
- A vector index already created (e.g., `test_index`)
- Documents embedded and ready for retrieval

## Step 1: Clone LangGraph with RAG Example

```bash
# Navigate to your code directory
cd ~/code

# Clone the LangGraph repository with the RAG example branch
git clone -b examples/yugabytedb-rag git@github.com:krishna-yb/langgraph.git

# Navigate to the RAG example directory
cd langgraph/examples/rag

# Verify files are present
ls -la
# Should show: vanilla_langgraph_pgvector_rag.py, run.sh, SETUP.md, etc.
```

## Step 2: Clone and Install langchain-yugabytedb (PgDistRagRetriever)

```bash
# Navigate to your code directory
cd ~/code

# Clone the repository with the PgDistRagRetriever feature branch
git clone -b feature/pg-dist-rag-retriever git@github.com:krishna-yb/langchain-yugabytedb.git

# Verify it's cloned
ls ~/code/langchain-yugabytedb
# Should show: setup.py, langchain_yugabytedb/, pyproject.toml, etc.

# Check the branch
cd ~/code/langchain-yugabytedb
git branch
# Should show: * feature/pg-dist-rag-retriever

# Verify PgDistRagRetriever commit
git log --oneline -1
# Should show: ac4dc1a Add PgDistRagRetriever for pg_dist_rag extension...
```

## Step 3: Create Virtual Environment and Install Dependencies

```bash
# Navigate to the RAG example directory
cd ~/code/langgraph/examples/rag

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install base dependencies
pip install --upgrade pip
pip install langgraph langchain-core langchain-openai

# Install langchain-yugabytedb in editable mode
# This allows you to modify the code and see changes immediately
pip install -e ~/code/langchain-yugabytedb

# Verify installation
python -c "from langchain_yugabytedb import PgDistRagRetriever; print('✓ PgDistRagRetriever installed')"
```

## Step 4: Set OpenAI API Key

```bash
export OPENAI_API_KEY="sk-proj-your-key-here"
```

## Step 5: View the Workflow Graph

```bash
cd ~/code/langgraph/examples/rag

# View ASCII workflow graph
./show_graph.sh

# Generate Mermaid diagram (paste output to mermaid.live)
./show_mermaid.sh
```

## Step 6: Run the RAG Agent

```bash
cd ~/code/langgraph/examples/rag

# Activate virtual environment if not already active
source venv/bin/activate

# Run a query (assumes 'test_index' exists in your database)
./run.sh --index-name test_index --query "What are the key coaching points for soccer passing drills?"

# Test with more retrieved documents
./run.sh --index-name test_index --query "How do you set up a breaking ball drill?" --k 5

# Test self-correction loop (intentionally bad query)
./run.sh --index-name test_index --query "What basketball techniques?" --max-retries 2
```

## Database Connection

The app connects to YugabyteDB using:
- **Host:** 127.0.0.1
- **Port:** 5433
- **Database:** yugabyte
- **User:** yugabyte
- **Password:** yugabyte (default)

You can modify the connection string in `vanilla_langgraph_pgvector_rag.py` if needed.

## Troubleshooting

### "relation test_index does not exist"

**Cause:** Vector index table not created in your database

**Fix:** Ensure your YugabyteDB instance has a vector index created using `dist_rag.init_vector_index()` and `dist_rag.build_index()`. See YugabyteDB pg_dist_rag documentation.

### "ModuleNotFoundError: No module named 'langchain_yugabytedb'"

**Cause:** Virtual environment not activated or langchain-yugabytedb not installed

**Fix:**
```bash
cd ~/code/langgraph/examples/rag
source venv/bin/activate
pip install -e ~/code/langchain-yugabytedb
```

### "OpenAI API key not set"

**Cause:** OPENAI_API_KEY environment variable not exported

**Fix:**
```bash
export OPENAI_API_KEY="sk-proj-your-actual-key-here"
```

## Quick Reference

### Key Paths
- **LangGraph Example:** `~/code/langgraph/examples/rag/`
- **PgDistRagRetriever Source:** `~/code/langchain-yugabytedb/`
- **Main Script:** `vanilla_langgraph_pgvector_rag.py`

### Useful Commands
```bash
# View workflow graph
./show_graph.sh

# Generate Mermaid diagram
./show_mermaid.sh

# Run with custom parameters
python vanilla_langgraph_pgvector_rag.py \
  --index-name test_index \
  --query "Your question here" \
  --k 3 \
  --max-retries 3

# List available indexes in your database
python -c "from langchain_yugabytedb import PgDistRagRetriever; from langchain_openai import OpenAIEmbeddings; r = PgDistRagRetriever(connection_string='postgresql+psycopg://yugabyte:yugabyte@127.0.0.1:5433/yugabyte', index_name='dummy', embeddings=OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)); print(r.list_available_indexes())"
```

## What This Example Demonstrates

1. **Stateful Workflow:** Uses LangGraph's `StateGraph` to manage RAG pipeline state
2. **Self-Correction:** Grades answer quality and rewrites queries if needed
3. **Query Analysis:** Analyzes user queries before retrieval
4. **Document Retrieval:** Uses YugabyteDB pg_dist_rag for vector similarity search
5. **Conditional Routing:** Dynamic workflow paths based on quality grades
6. **Loop Protection:** Configurable retry limits to prevent infinite loops
7. **Observability:** Detailed workflow logging and graph visualization
