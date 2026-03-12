#!/bin/bash

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    echo "Usage: export OPENAI_API_KEY='your-key' && ./run.sh"
    exit 1
fi

source venv/bin/activate

# Langfuse local (tracing-only project in local Langfuse instance)
export LANGFUSE_BASE_URL="${LANGFUSE_BASE_URL:-http://localhost:3000}"
export LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY:-pk-lf-local-tracing-project}"
export LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY:-sk-lf-local-tracing-project-secret}"

# Group all traces in this run under one session.
# Override by setting LANGFUSE_SESSION_ID before calling run.sh.
export LANGFUSE_SESSION_ID="${LANGFUSE_SESSION_ID:-$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')}"

# Automatically enable Langfuse callbacks unless explicitly disabled.
if [[ " $* " == *" --langfuse "* ]]; then
  python vanilla_langgraph_pgvector_rag.py "$@"
else
  python vanilla_langgraph_pgvector_rag.py --langfuse "$@"
fi
