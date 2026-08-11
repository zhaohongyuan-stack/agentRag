.PHONY: install dev up down test test-unit test-integration lint format typecheck run-agent run-retrieval clean

# 安装
install:
	pip install -e ".[dev]"

dev:
	pip install -e ".[dev]"

# 基础设施
up:
	docker-compose up -d

down:
	docker-compose down

# 测试
test:
	pytest

test-unit:
	pytest agent_platform/tests/unit knowledge_platform/tests/unit

test-integration:
	pytest agent_platform/tests/workflow knowledge_platform/tests/integration

# 代码质量
lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy agent_platform knowledge_platform

# 服务
run-agent:
	uvicorn services.agent_api.main:app --reload --host 0.0.0.0 --port 8000

run-retrieval:
	uvicorn services.retrieval_api.main:app --reload --host 0.0.0.0 --port 8001

# 脚本
validate-data:
	python scripts/validate_parsed_data/main.py

build-indexes:
	python scripts/build_indexes/main.py

run-eval:
	python scripts/run_evaluation/main.py

smoke-test:
	python scripts/smoke_test/main.py

# 清理
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name "*.pyc" -delete; \
	rm -rf .pytest_cache .mypy_cache .ruff_cache
