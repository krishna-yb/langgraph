#!/bin/bash

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    echo "Usage: export OPENAI_API_KEY='your-key' && ./run.sh"
    exit 1
fi

source venv/bin/activate
python vanilla_langgraph_pgvector_rag.py "$@"
