# 部署指南

## 环境要求

| 组件 | 最低配置 | 推荐配置 |
|---|---|---|
| CPU | 8 核 | 16 核 |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 100 GB SSD | 200 GB SSD |
| GPU（可选） | NVIDIA T4 | NVIDIA A10 |
| 操作系统 | Ubuntu 22.04 | Ubuntu 22.04 |
| Docker | 24.0+ | 24.0+ |

## 本地开发部署

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（复制模板到项目根目录的 .env.local）
cp deploy/.env.example .env.local
vim .env.local   # 填入 DEEPSEEK_API_KEY、DB_PASSWORD、JWT_SECRET_KEY 等

# 3. 初始化数据库（幂等迁移，服务启动时也会自动执行）
python scripts/migrate.py

# 4. 启动服务
uvicorn backend.main:app --reload --port 8000
```

验证：`curl http://localhost:8000/health`

## Docker 部署（私有化）

```bash
# 1. 克隆代码
git clone <your-repo-url> /opt/equipment-cs
cd /opt/equipment-cs

# 2. 配置环境变量
cp deploy/.env.example .env.local
vim .env.local   # 填入 API Key 等配置

# 3. 启动全部服务（postgres / etcd / minio / milvus / backend / nginx）
docker-compose -f deploy/docker-compose.yml up -d

# 4. 验证
curl http://localhost:8000/health
```

> 说明：
> - backend 容器名为 `equipment_cs_backend`，进入容器调试：`docker exec -it equipment_cs_backend bash`
> - 镜像内已打包 `data/` 与 `scripts/`，可在容器内运行 `python scripts/migrate.py`
> - 调试 Milvus 面板（attu）：`docker-compose --profile debug -f deploy/docker-compose.yml up -d`，访问 http://localhost:30000

## 企业微信接入（规划中）

> 当前代码尚未实现企业微信回调端点。待接入后：
> 1. 登录企业微信管理后台 → 应用管理 → 自建 → 创建应用
> 2. 配置回调 URL 与 corpid、secret、token、aes_key
> 3. 将密钥填入 `.env.local` 并重启服务
