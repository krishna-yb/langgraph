#!/bin/bash

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set before running this script}"
export LANGFUSE_BASE_URL="${LANGFUSE_BASE_URL:-http://localhost:3000}"
export LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY:-pk-lf-local-tracing-project}"
export LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY:-sk-lf-local-tracing-project-secret}"

INDEX="soccer_drills_test"

# 3 relevant + 2 irrelevant queries
QUERIES=(
  "What are the best passing drills for youth players?"
  "How do you set up a rondo drill?"
  "What is the capital of France?"
  "How do you improve shooting accuracy in soccer?"
  "What is the speed of light?"
)

source venv/bin/activate

for i in "${!QUERIES[@]}"; do
  N=$((i + 1))
  Q="${QUERIES[$i]}"
  echo ""
  echo "=== Run $N/5 ==="
  echo "Query: $Q"
  echo "----------------------------------------"
  python vanilla_langgraph_pgvector_rag.py \
    --langfuse \
    --index-name "$INDEX" \
    --query "$Q"
  echo ""
  echo "--- Done $N/5 ---"
done

echo ""
echo "All 5 runs complete. Check Langfuse at http://localhost:3000"
