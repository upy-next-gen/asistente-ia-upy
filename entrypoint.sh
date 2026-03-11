#!/bin/sh
set -e
python scripts/index_docs_pplx.py
chainlit run app.py --host 0.0.0.0 --port 8000 -h