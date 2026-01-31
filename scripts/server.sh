#!/bin/bash
MAX_CPU_JOBS=8  ANNOTATOR_LOG=annotator_server.log uvicorn annotator_server.main:app --reload --host 0.0.0.0 --port 8000
