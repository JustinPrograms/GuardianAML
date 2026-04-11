uvicorn endpoints_db:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload \
    --reload-dir . \
    --reload-dir ../../frontend \
    --reload-dir ../../models \