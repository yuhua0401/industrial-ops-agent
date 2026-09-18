FROM python:3.11-slim

WORKDIR /app

# 系统依赖（gcc: torch/transformers 等编译需要；curl: 健康检查）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖层（源码变动时不重建此层）
# 注：如需减小镜像体积，torch 可改用 CPU 源：
#   pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY backend/ backend/
# 数据（诊断树 / 维修实例）与脚本：诊断 Agent 运行时按 Path(__file__) 相对路径加载 data/，必须打包
COPY data/ data/
COPY scripts/ scripts/

# 非 root 运行（最小权限）
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 容器健康检查（轻量存活探针）
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
