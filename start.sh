#!/bin/bash
OMBRE_BUCKETS_DIR=/data/buckets_kk OMBRE_PORT=8000 python server.py &
OMBRE_BUCKETS_DIR=/data/buckets_deepseek OMBRE_PORT=8001 python server.py &
wait
