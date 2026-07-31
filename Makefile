.PHONY: install dev up down test test-unit test-integration lint format typecheck run-agent run-retrieval clean

# 安装（核心 + 开发 + LLM + Redis）
install:
	pip install -e ".[dev,llm,redis]"

dev:
	pip install -e ".[dev,llm,redis]"

# 基础设施（仅 Redis，可选）
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
# A组检索服务（端口 8000）
run-retrieval:
	cd knowledge_platform/retrieval && python -m retrieval_service.server

# B组 Agent 服务（端口 8002，避免与 A组冲突）
run-agent:
	AGENT_PORT=8002 python -m agent_platform.server

# 一键启动 A组 + B组（推荐）
run-all:
	python scripts/start_servers.py

# 检查服务状态
check-services:
	python scripts/start_servers.py --check

# 联调测试
smoke-test:
	python scripts/real_integration_test.py "银行业总资产是多少"

# 清理
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name "*.pyc" -delete; \
	rm -rf .pytest_cache .mypy_cache .ruff_cache
