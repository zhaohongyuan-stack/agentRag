# ACE-RAG Dockerfile — Python 3.11 轻量级镜像 (国内加速)
FROM python:3.11-slim AS base

# 替换 apt 源为阿里云镜像 (Debian Trixie 兼容)
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && \
        sed -i 's|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list && \
        sed -i 's|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list; \
    fi

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 配置 pip 阿里云镜像
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# 安装核心依赖
RUN pip install --no-cache-dir \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.6.0" \
    "numpy>=1.26.0" \
    "pyyaml>=6.0" \
    "openai>=1.30.0" \
    "httpx>=0.27.0" \
    "redis[hiredis]>=5.0.0" \
    "jieba>=0.42.1" \
    "python-dotenv>=1.0.0" \
    "requests>=2.31.0" \
    "python-multipart>=0.0.9" \
    "openpyxl>=3.1.0" \
    "xlrd>=2.0.1" \
    "pdfplumber>=0.10.0" \
    "pypdf>=3.17.0" \
    "python-docx>=1.1.0" \
    "langgraph>=0.2.0" \
    "langchain-core>=0.3.0"

# 复制项目代码
COPY agent_platform/ ./agent_platform/
COPY knowledge_platform/ ./knowledge_platform/
COPY contracts/ ./contracts/
COPY scripts/ ./scripts/
COPY .env ./.env

# 环境变量默认值
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_EMBED_API=true \
    USE_RERANKER=true \
    REDIS_HOST=redis \
    REDIS_PORT=6379

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认启动检索服务
CMD ["python", "-m", "retrieval_service.server"]
