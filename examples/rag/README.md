# RAG Example with LangGraph

Self-correcting RAG agent using YugabyteDB pg_dist_rag and LangGraph.

## Prerequisites

1. Clone and install `langchain-yugabytedb` PR #1 (feature/pg-dist-rag-retriever branch)
2. YugabyteDB with pg_dist_rag extension
3. OpenAI API key

**See [SETUP.md](SETUP.md) for complete installation steps.**

## Database Contents (test_index)

The `test_index` contains **169 chunks** from **3 soccer drill documents**:
- `soccer_drills_page_1.json` (62 chunks)
- `soccer_drills_page_2.json` (62 chunks)  
- `soccer_drills_page_3.json` (45 chunks)

Documents stored in: `s3://proto-automated-embedding/playgentic_documents/sample/`

## Setup

```bash
cd /home/krishna/code/langgraph/examples/rag
export OPENAI_API_KEY="sk-proj-your-openai-key-here"
```

## Usage

### View Workflow Graph

**ASCII Graph (in terminal):**
```bash
./show_graph.sh
```

**Mermaid Diagram (for visualization):**
```bash
./show_mermaid.sh
```
This outputs a Mermaid diagram that you can paste into https://mermaid.live/ to visualize the workflow graph.

Or directly:
```bash
source venv/bin/activate
python vanilla_langgraph_pgvector_rag.py --show-graph --index-name test_index --query "dummy"
python vanilla_langgraph_pgvector_rag.py --mermaid --index-name test_index --query "dummy"
```

### Run Query

```bash
./run.sh --index-name test_index --query "What are the key coaching points for soccer passing drills?"
```

### Example Questions for test_index

```bash
# About coaching points
python vanilla_langgraph_pgvector_rag.py --index-name test_index --query "What are the key coaching points for soccer passing drills?"

# About drill setup
python vanilla_langgraph_pgvector_rag.py --index-name test_index --query "How do you set up a breaking ball drill?"

# About variations
python vanilla_langgraph_pgvector_rag.py --index-name test_index --query "What are some drill variations for defensive pressure training?"

# About attacking drills
python vanilla_langgraph_pgvector_rag.py --index-name test_index --query "Explain the setup for attacking vs defending drills"

# Get more documents
python vanilla_langgraph_pgvector_rag.py --index-name test_index --query "What are the objectives of soccer training drills?" --k 5
```

## Quick Run

```bash
chmod +x run.sh show_graph.sh show_mermaid.sh
export OPENAI_API_KEY='your-key'
./run.sh --index-name test_index --query "Your question here"
```

## Files

- `vanilla_langgraph_pgvector_rag.py` - Main RAG agent implementation
- `run.sh` - Quick run script
- `show_graph.sh` - Display ASCII workflow graph
- `show_mermaid.sh` - Generate Mermaid diagram
- `workflow.mermaid` - Saved Mermaid diagram (paste to mermaid.live)
- `QUICKSTART.txt` - Quick start commands
- `GRAPH.txt` - Detailed workflow documentation
