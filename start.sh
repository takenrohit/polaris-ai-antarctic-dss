#!/bin/bash
# POLARIS-AI POSIX One-Command Startup Script
set -e

echo "Starting POLARIS-AI Antarctic Decision Support System..."
python3 run_demo.py

echo ""
echo "Launching Backend and Frontend Services..."
(cd backend && python3 run_backend.py) &
BACKEND_PID=$!

(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "========================================================"
echo "Services are running:"
echo "  - Backend:  http://127.0.0.1:8000"
echo "  - Swagger:  http://127.0.0.1:8000/docs"
echo "  - Frontend: http://localhost:5173"
echo "========================================================"
echo "Press CTRL+C to stop all services."

trap "kill $BACKEND_PID $FRONTEND_PID" SIGINT SIGTERM EXIT
wait
