FROM python:3.11-slim

WORKDIR /app

# 系统依赖（gcc: torch/transformers 等编译需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖层（源码变动时不重建此层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY backend/ backend/
# 数据（诊断树 / 维修实例）与脚本：诊断 Agent 运行时按 Path(__file__) 相对路径加载 data/，必须打包
COPY data/ data/
COPY scripts/ scripts/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
