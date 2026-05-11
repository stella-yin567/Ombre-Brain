#!/bin/bash
nginx -c /app/nginx.conf
OMBRE_BUCKETS_DIR=/app/buckets/kk OMBRE_PORT=8001 python server.py &
OMBRE_BUCKETS_DIR=/app/buckets/deepseek OMBRE_PORT=8002 python server.py &
wait
