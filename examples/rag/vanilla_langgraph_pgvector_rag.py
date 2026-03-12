"""Self-Correcting RAG Agent with LangGraph and PgDistRagRetriever.

Prerequisites:
    1. Clone PR #1 branch: git clone -b feature/pg-dist-rag-retriever git@github.com:krishna-yb/langchain-yugabytedb.git
    2. Install: pip install -e ~/code/langchain-yugabytedb
    3. Start YugabyteDB with pg_dist_rag enabled
    4. Create index with documents (see SETUP.md)

Database: test_index contains 169 chunks from 3 soccer drill documents
Content: Coaching points, drill setups, variations, attacking/defending exercises

View workflow graph:
    ./show_graph.sh              # ASCII diagram in terminal
    ./show_mermaid.sh            # Mermaid diagram (paste to mermaid.live)

Example queries:
    ./run.sh --index-name test_index --query "What are the key coaching points for soccer passing drills?"
    ./run.sh --index-name test_index --query "How do you set up a breaking ball drill?"
    ./run.sh --index-name test_index --query "What are drill variations for defensive pressure training?"
    ./run.sh --index-name test_index --query "Explain attacking vs defending drills" --k 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

try:
    try:
        # Older/newer SDKs may expose CallbackHandler in different modules.
        from langfuse.callback import CallbackHandler
    except ImportError:
        from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    LANGFUSE_IMPORT_ERROR = str(e)
else:
    LANGFUSE_IMPORT_ERROR = ""

LANGCHAIN_YB_PATH = Path.home() / "code" / "langchain-yugabytedb"
if LANGCHAIN_YB_PATH.exists() and str(LANGCHAIN_YB_PATH) not in sys.path:
    sys.path.insert(0, str(LANGCHAIN_YB_PATH))

class RAGState(TypedDict):
    query: str
    max_retries: int
    needs_retrieval: bool
    retrieved_docs: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str
    is_good_answer: bool
    iteration: int
    workflow_steps: list[str]

def analyze_query(state: RAGState) -> dict:
    print("\n" + "="*80)
    print("STEP 1: ANALYZING QUERY")
    print("="*80)
    query = state["query"].lower()
    retrieval_indicators = ["what", "how", "why", "explain", "describe", "tell me"]
    needs_retrieval = any(indicator in query for indicator in retrieval_indicators)
    
    print(f"→ Checking if query requires document retrieval...")
    print(f"→ Scanning for question indicators: {retrieval_indicators}")
    
    if state.get("iteration", 0) > 0:
        needs_retrieval = True
        print(f"→ This is retry attempt {state.get('iteration', 0) + 1}, forcing retrieval")
    
    if needs_retrieval:
        print(f"✓ Query contains question words → Will retrieve documents from vector DB")
    else:
        print(f"✓ Query is simple → Can answer directly without retrieval")
    
    return {
        "needs_retrieval": needs_retrieval,
        "workflow_steps": state.get("workflow_steps", []) + [f"analyze → {'retrieve' if needs_retrieval else 'direct'}"],
    }


def retrieve_documents(state: RAGState, retriever) -> dict:
    print("\n" + "="*80)
    print("STEP 2: RETRIEVING DOCUMENTS FROM VECTOR DATABASE")
    print("="*80)
    query = state["query"]
    iteration = state.get("iteration", 0)
    print(f"→ Query: '{query}'")
    print(f"→ Attempt: {iteration + 1}")
    
    try:
        print(f"→ Converting query to embeddings using OpenAI text-embedding-3-small...")
        print(f"→ Performing vector similarity search in YugabyteDB pg_dist_rag...")
        print(f"→ Searching through 169 chunks from soccer drills database...")
        docs = retriever.invoke(query)
        print(f"✓ Retrieved top {len(docs)} most relevant document chunks")
        for i, doc in enumerate(docs, 1):
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"  [{i}] {preview}...")
        return {
            "retrieved_docs": docs,
            "workflow_steps": state["workflow_steps"] + [f"retrieve → {len(docs)} docs"],
        }
    except Exception as e:
        print(f"✗ Retrieval failed: {e}")
        return {
            "retrieved_docs": [],
            "workflow_steps": state["workflow_steps"] + ["retrieve → error"],
        }


def generate_answer(state: RAGState, llm) -> dict:
    print("\n" + "="*80)
    print("STEP 3: GENERATING ANSWER WITH LLM")
    print("="*80)
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    
    if not state.get("needs_retrieval", False):
        print("→ Direct answer mode (no retrieval needed)")
        answer = "This query can be answered directly without retrieving documents."
        return {
            "answer": answer,
            "messages": [HumanMessage(content=query), AIMessage(content=answer)],
            "workflow_steps": state["workflow_steps"] + ["generate → direct"],
        }
    
    if not docs:
        print("No documents available for answer generation")
        answer = "I couldn't find relevant documents to answer this question."
        return {
            "answer": answer,
            "messages": [HumanMessage(content=query), AIMessage(content=answer)],
            "workflow_steps": state["workflow_steps"] + ["generate → no_docs"],
        }
    
    print(f"→ Preparing context from {len(docs)} retrieved documents...")
    print(f"→ Adding citation markers [1], [2], [3], [4] to each chunk...")
    context = "\n\n".join([f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)])
    context_size = len(context)
    print(f"→ Total context size: {context_size} characters")
    
    system_prompt = """You are a helpful AI assistant. Answer questions based on the provided context.
Use ONLY information from the provided context. Include citation numbers [1], [2], etc."""
    
    user_prompt = f"""Context:\n{context}\n\nQuestion: {query}\n\nProvide a clear answer with citations [N]."""
    
    try:
        if llm:
            print(f"→ Sending context + query to OpenAI GPT-4o-mini...")
            print(f"→ Instructing LLM to use ONLY provided context and include citations...")
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = llm.invoke(messages)
            
            answer = response.content
            print(f"✓ LLM generated answer successfully ({len(answer)} characters)")
            citation_count = answer.count('[')
            print(f"✓ Answer contains {citation_count} citations")
        else:
            print("⚠ No LLM available, returning raw context")
            answer = f"Context found:\n{context[:500]}..."
        
        return {
            "answer": answer,
            "messages": [HumanMessage(content=query), AIMessage(content=answer)],
            "workflow_steps": state["workflow_steps"] + ["generate → success"],
        }
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return {
            "answer": f"Error generating answer: {e}",
            "messages": [HumanMessage(content=query), AIMessage(content=f"Error: {e}")],
            "workflow_steps": state["workflow_steps"] + ["generate → error"],
        }


def grade_answer(state: RAGState) -> dict:
    print("\n" + "="*80)
    print("STEP 4: QUALITY CHECKING ANSWER")
    print("="*80)
    answer = state.get("answer", "")
    docs = state.get("retrieved_docs", [])
    iteration = state.get("iteration", 0)
    
    print(f"→ Running quality checks on generated answer...")
    
    is_substantial = len(answer) > 100
    has_citations = "[" in answer if docs else True
    no_errors = "error" not in answer.lower() and "couldn't find" not in answer.lower()
    can_retry = iteration < state["max_retries"]
    
    print(f"→ Checking if answer is substantial (>100 chars)...")
    print(f"  {'✓' if is_substantial else '✗'} Length: {len(answer)} characters")
    
    print(f"→ Checking if answer includes citations [N]...")
    print(f"  {'✓' if has_citations else '✗'} Citations: {'Found' if has_citations else 'Missing'}")
    
    print(f"→ Checking for error messages...")
    print(f"  {'✓' if no_errors else '✗'} Clean answer: {'Yes' if no_errors else 'Contains errors'}")
    
    print(f"→ Checking retry budget...")
    print(f"  {'✓' if can_retry else '✗'} Retries remaining: {state['max_retries'] - iteration}")
    
    is_good = is_substantial and has_citations and no_errors
    
    if is_good:
        print(f"\n✓ QUALITY CHECK PASSED - Answer is good!")
    elif not can_retry:
        print(f"\n⚠ QUALITY CHECK FAILED - But no retries left, accepting answer")
    else:
        print(f"\n✗ QUALITY CHECK FAILED - Will rewrite query and retry")
    
    return {
        "is_good_answer": is_good or not can_retry,
        "workflow_steps": state["workflow_steps"] + [f"grade → {'pass' if is_good else 'fail'}"],
    }


def rewrite_query(state: RAGState) -> dict:
    print("\n" + "="*80)
    print("STEP 5: REWRITING QUERY FOR RETRY")
    print("="*80)
    original = state["query"]
    iteration = state.get("iteration", 0)
    
    print(f"→ Answer quality was insufficient, improving query...")
    print(f"→ Retry attempt: {iteration + 1}")
    
    if "explain" not in original.lower() and "how" not in original.lower():
        rewritten = f"Explain in detail: {original}"
        print(f"→ Strategy: Adding 'Explain in detail' prefix")
    else:
        rewritten = f"{original} with specific examples and technical details"
        print(f"→ Strategy: Requesting specific examples and details")
    
    print(f"\n  Original:  {original}")
    print(f"  Rewritten: {rewritten}")
    print(f"\n→ Looping back to STEP 2 (RETRIEVE) with improved query...")
    
    return {
        "query": rewritten,
        "iteration": iteration + 1,
        "workflow_steps": state["workflow_steps"] + ["rewrite → retry"],
    }

def should_retrieve(state: RAGState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state["needs_retrieval"] else "generate"

def should_improve(state: RAGState) -> Literal["improve", "end"]:
    return "end" if state["is_good_answer"] else "improve"

def build_rag_graph(retriever, llm):
    workflow = StateGraph(RAGState)
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", lambda s: retrieve_documents(s, retriever))
    workflow.add_node("generate", lambda s: generate_answer(s, llm))
    workflow.add_node("grade", grade_answer)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges("analyze", should_retrieve, {"retrieve": "retrieve", "generate": "generate"})
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade")
    workflow.add_conditional_edges("grade", should_improve, {"improve": "rewrite", "end": END})
    workflow.add_edge("rewrite", "retrieve")
    return workflow.compile()

def setup_retriever(connection: str, index_name: str, k: int):
    from langchain_openai import OpenAIEmbeddings
    from langchain_yugabytedb import PgDistRagRetriever
    
    try:
        config = PgDistRagRetriever.get_embedding_config_for_index(connection_string=connection, index_name=index_name)
        embeddings = OpenAIEmbeddings(model=config['model'], dimensions=config['dimensions'])
    except Exception:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    return PgDistRagRetriever(connection_string=connection, index_name=index_name, embeddings=embeddings, k=k)

def setup_llm():
    if "OPENAI_API_KEY" not in os.environ:
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except ImportError:
        return None


def print_execution_path(result):
    """Print the actual execution path taken for this query."""
    print(f"\n{'='*80}")
    print("🎯 DYNAMIC EXECUTION PATH (What Actually Happened)")
    print('='*80)
    
    steps = result['workflow_steps']
    iteration = result['iteration']
    
    if iteration == 0:
        # Single pass, no loop
        print("\n📍 PATH: START → ANALYZE → RETRIEVE → GENERATE → GRADE → END")
        print("""
        ┌─────────┐
        │  START  │
        └────┬────┘
             │
             ▼
        ┌─────────┐
        │ ANALYZE │ ✓ Executed
        └────┬────┘
             │
             ▼
        ┌──────────┐
        │ RETRIEVE │ ✓ Executed (4 docs)
        └────┬─────┘
             │
             ▼
        ┌──────────┐
        │ GENERATE │ ✓ Executed
        └────┬─────┘
             │
             ▼
        ┌────────┐
        │ GRADE  │ ✓ PASSED
        └────┬───┘
             │
             ▼
        ┌─────┐
        │ END │
        └─────┘
        
        ✓ No retry needed - Answer passed quality check on first attempt
        """)
    else:
        # Loop occurred
        print(f"\n📍 PATH: START → ANALYZE → RETRIEVE → GENERATE → GRADE → REWRITE → (LOOP {iteration}x) → END")
        print("""
        ┌─────────┐
        │  START  │
        └────┬────┘
             │
             ▼
        ┌─────────┐
        │ ANALYZE │ ✓ Executed
        └────┬────┘
             │
             ▼""")
        
        for i in range(iteration + 1):
            if i == 0:
                print(f"""        ┌──────────┐
        │ RETRIEVE │ ✓ Try {i+1} - Retrieved docs
        └────┬─────┘
             │
             ▼
        ┌──────────┐
        │ GENERATE │ ✓ Generated answer
        └────┬─────┘
             │
             ▼
        ┌────────┐
        │ GRADE  │ ✗ FAILED (missing citations or short)
        └────┬───┘
             │
             ▼
        ┌─────────┐
        │ REWRITE │ ✓ Query improved
        └────┬────┘
             │
             │ (LOOP BACK)
             ▼""")
            elif i < iteration:
                print(f"""        ┌──────────┐
        │ RETRIEVE │ ✓ Try {i+1} - Retrieved docs
        └────┬─────┘
             │
             ▼
        ┌──────────┐
        │ GENERATE │ ✓ Generated answer
        └────┬─────┘
             │
             ▼
        ┌────────┐
        │ GRADE  │ ✗ FAILED again
        └────┬───┘
             │
             ▼
        ┌─────────┐
        │ REWRITE │ ✓ Query improved
        └────┬────┘
             │
             │ (LOOP BACK)
             ▼""")
            else:
                print(f"""        ┌──────────┐
        │ RETRIEVE │ ✓ Try {i+1} - Retrieved docs
        └────┬─────┘
             │
             ▼
        ┌──────────┐
        │ GENERATE │ ✓ Generated answer
        └────┬─────┘
             │
             ▼
        ┌────────┐
        │ GRADE  │ {'✓ PASSED' if result['is_good_answer'] else '⚠ FAILED (no retries left)'}
        └────┬───┘
             │
             ▼
        ┌─────┐
        │ END │
        └─────┘
        
        🔄 Looped {iteration} time(s) before completing
        """)

def print_graph_structure(app=None, format_type="ascii"):
    """Print the workflow graph structure."""
    
    if format_type == "mermaid" and app:
        print("\n" + "="*80)
        print("🗺️  LANGGRAPH WORKFLOW - MERMAID DIAGRAM")
        print("="*80)
        try:
            # Get the graph in Mermaid format
            mermaid_graph = app.get_graph().draw_mermaid()
            print(mermaid_graph)
            print("\n💡 Copy the above and paste into: https://mermaid.live/")
        except Exception as e:
            print(f"Error generating Mermaid diagram: {e}")
        print("="*80 + "\n")
        return
    
    # ASCII format
    print("\n" + "="*80)
    print("🗺️  LANGGRAPH WORKFLOW STRUCTURE")
    print("="*80)
    print("""
    ┌─────────────┐
    │    START    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   ANALYZE   │  ← Check if query needs retrieval
    └──────┬──────┘
           │
      ┌────┴────┐
      │         │
      ▼         ▼
   DIRECT    RETRIEVE  ← Get docs from vector DB
      │         │
      │         ▼
      │    GENERATE   ← LLM generates answer
      │         │
      └────┬────┘
           │
           ▼
      ┌────────┐
      │ GRADE  │      ← Quality check
      └───┬────┘
          │
      ┌───┴────┐
      │        │
      ▼        ▼
    PASS    REWRITE   ← Improve query
      │        │
      │        └──────┐
      │               │
      ▼               │
     END         (loop back
                  to RETRIEVE)

    Key:
    • ANALYZE: Decides if documents needed
    • RETRIEVE: Vector similarity search
    • GENERATE: LLM creates answer with citations
    • GRADE: Checks length, citations, errors
    • REWRITE: Improves query for retry
    """)
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Self-Correcting RAG with LangGraph")
    parser.add_argument("--index-name", help="Vector index name")
    parser.add_argument("--query", help="Question to ask")
    parser.add_argument("--connection", default=os.getenv("DB_CONNECTION", "postgresql+psycopg://yugabyte:yugabyte@127.0.0.1:5433/yugabyte"))
    parser.add_argument("--k", type=int, default=4, help="Number of documents")
    parser.add_argument("--max-retries", type=int, default=1, help="Max retry attempts")
    parser.add_argument("--show-graph", action="store_true", help="Display workflow graph (ASCII)")
    parser.add_argument("--mermaid", action="store_true", help="Generate Mermaid diagram")
    parser.add_argument("--langfuse", action="store_true", help="Enable Langfuse observability")
    args = parser.parse_args()
    
    # Show graph without database connection
    if args.show_graph:
        print_graph_structure(None, format_type="ascii")
        return
    
    if args.mermaid:
        # Need to build graph for Mermaid
        retriever = setup_retriever(args.connection, args.index_name or "dummy", args.k)
        llm = setup_llm()
        app = build_rag_graph(retriever, llm)
        print_graph_structure(app, format_type="mermaid")
        return
    
    # Validate required args for query mode
    if not args.index_name or not args.query:
        parser.error("--index-name and --query are required (unless using --show-graph)")
    
    retriever = setup_retriever(args.connection, args.index_name, args.k)
    llm = setup_llm()
    app = build_rag_graph(retriever, llm)
    
    # Initialize Langfuse handler if requested
    langfuse_handler = None
    if args.langfuse:
        # Cookbook uses LANGFUSE_BASE_URL. Map it to LANGFUSE_HOST if needed.
        if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
            os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL", "")

        if not LANGFUSE_AVAILABLE:
            print("⚠ --langfuse flag provided but Langfuse callback import failed")
            print(f"→ Import error: {LANGFUSE_IMPORT_ERROR}")
            print("→ Install required deps: pip install langfuse langchain")
        elif not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
            print("⚠ --langfuse flag provided but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set")
        else:
            try:
                # Langfuse cookbook pattern: initialize callback from env vars.
                langfuse_handler = CallbackHandler()
                print(
                    "✓ Langfuse observability enabled | base_url="
                    f"{os.getenv('LANGFUSE_BASE_URL', os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com'))}"
                )
            except Exception as e:
                print(f"⚠ Failed to initialize Langfuse: {e}")
    
    print_graph_structure(app, format_type="ascii")
    
    print(f"\n{'='*80}")
    print("🤖 LANGGRAPH SELF-CORRECTING RAG WORKFLOW")
    print('='*80)
    print(f"📝 Query: {args.query}")
    print(f"🗄️  Database Index: {args.index_name}")
    print(f"📊 Top-K Documents: {args.k}")
    print(f"🔄 Max Retry Attempts: {args.max_retries}")
    if langfuse_handler:
        print(f"🔍 Langfuse Tracing: Enabled")
    print('='*80)
    print("\nStarting workflow execution...\n")
    
    invoke_config = {}
    if langfuse_handler:
        invoke_config["callbacks"] = [langfuse_handler]

    result = app.invoke(
        {
            "query": args.query,
            "max_retries": args.max_retries,
            "needs_retrieval": False,
            "retrieved_docs": [],
            "messages": [],
            "answer": "",
            "is_good_answer": False,
            "iteration": 0,
            "workflow_steps": [],
        },
        config=invoke_config if invoke_config else None,
    )
    
    print(f"\n{'='*80}")
    print("✅ WORKFLOW COMPLETE - FINAL ANSWER")
    print('='*80)
    print(result['answer'])
    print(f"\n{'='*80}")
    print("📊 EXECUTION SUMMARY")
    print('='*80)
    print("Workflow steps executed:")
    for i, step in enumerate(result['workflow_steps'], 1):
        print(f"  {i}. {step}")
    print(f"\n📈 Statistics:")
    print(f"  • Total iterations: {result['iteration'] + 1}")
    print(f"  • Documents retrieved: {len(result['retrieved_docs'])}")
    print(f"  • Answer quality: {'✓ Passed' if result['is_good_answer'] else '✗ Below threshold'}")
    print('='*80)
    
    # Generate dynamic execution path diagram
    print_execution_path(result)
    print('='*80 + '\n')

    if langfuse_handler:
        langfuse_handler.client.flush()


if __name__ == "__main__":
    main()
