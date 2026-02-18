#!/bin/bash

cd /home/krishna/code/langgraph/examples/rag
source venv/bin/activate
python vanilla_langgraph_pgvector_rag.py --mermaid --index-name test_index --query "dummy"
