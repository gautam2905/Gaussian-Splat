# Interactive AI Museums — top-level targets
#
# Most paths assume conda env `museums` and Node 20.18+. See README.md.

.PHONY: help dev backend frontend test test-backend test-frontend smoke clean install

CONDA_ENV ?= museums
DATA_DIR  ?= /tmp/museums-data

help:
	@echo "Interactive AI Museums -- Make targets"
	@echo ""
	@echo "  make install       Install backend + frontend deps"
	@echo "  make dev           Boot backend (uvicorn) + frontend (vite) in parallel"
	@echo "  make backend       Boot backend only"
	@echo "  make frontend      Boot frontend only"
	@echo "  make test          Run all tests (backend + frontend)"
	@echo "  make test-backend  Run backend pytest"
	@echo "  make test-frontend Run frontend vitest"
	@echo "  make smoke         Curl the backend pipeline end-to-end"
	@echo "  make clean         Remove caches, lock files, temp data"
	@echo ""
	@echo "  Env vars: CONDA_ENV (default: museums), DATA_DIR (default: /tmp/museums-data)"

install:
	conda run -n $(CONDA_ENV) --no-capture-output pip install -e 'backend/[dev]'
	cd frontend && npm install

dev:
	@mkdir -p $(DATA_DIR)
	@echo "Booting backend (uvicorn) on :8000 and frontend (vite) on :5173..."
	@echo "Press Ctrl+C to stop both."
	@(DATA_DIR=$(DATA_DIR) conda run -n $(CONDA_ENV) --no-capture-output uvicorn --app-dir backend app.main:app --port 8000 --reload &) ; \
	 (cd frontend && npm run dev -- --port 5173 &) ; \
	 wait

backend:
	@mkdir -p $(DATA_DIR)
	DATA_DIR=$(DATA_DIR) conda run -n $(CONDA_ENV) --no-capture-output uvicorn --app-dir backend app.main:app --port 8000 --reload

frontend:
	cd frontend && npm run dev -- --port 5173

test: test-backend test-frontend

test-backend:
	cd backend && conda run -n $(CONDA_ENV) --no-capture-output pytest tests/ -v

test-frontend:
	cd frontend && npx vitest run

smoke:
	@mkdir -p $(DATA_DIR)
	@DATA_DIR=$(DATA_DIR) conda run -n $(CONDA_ENV) --no-capture-output uvicorn --app-dir backend app.main:app --port 8000 & \
	 SERVER_PID=$$! ; \
	 sleep 3 ; \
	 echo "--- POST /api/exhibits ---" ; \
	 curl -fsS -X POST http://localhost:8000/api/exhibits -H 'content-type: application/json' -d '{"prompt":"an anatomy exhibit"}' && echo ; \
	 echo "--- SSE /api/exhibits/events ---" ; \
	 curl -fsS -N "http://localhost:8000/api/exhibits/events?prompt=an%20anatomy%20exhibit" 2>&1 | head -25 ; \
	 kill $$SERVER_PID 2>/dev/null ; \
	 echo ; echo "smoke OK"

clean:
	rm -rf backend/__pycache__ backend/.pytest_cache backend/app/__pycache__ backend/app/**/__pycache__
	rm -rf frontend/node_modules/.vite frontend/dist
	rm -rf $(DATA_DIR)
	@echo "Cleaned caches + $(DATA_DIR). To fully nuke: rm -rf backend/.venv frontend/node_modules"
