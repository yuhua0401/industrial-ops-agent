"""
preflight_check.py - 演示环境预检

一条命令检查全链路依赖是否就绪，浏览器演示前必跑：
    ✅/❌ Docker 容器（postgres / milvus 等）
    ✅/❌ PostgreSQL 连接 + 关键表（users/devices/parts/tickets）
    ✅/❌ 种子数据（登录账号 engineer、设备 SN-AC-0001、备件 BRG-6204）
    ✅/❌ Milvus collection（CNC-1000 示例知识库）
    ✅/❌ LLM 配置（DeepSeek API Key 已填）
    ℹ️  附带自动修复建议（该跑哪条命令）

用法：
    python scripts/preflight_check.py
"""
import asyncio
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL, INFO = "✅", "❌", "ℹ️ "
results: list[tuple[str, str, str]] = []   # (icon, item, hint)


def record(ok: bool, item: str, hint: str = "") -> None:
    results.append((PASS if ok else FAIL, item, hint))


def check_tcp(host: str, port: int, item: str, hint: str) -> None:
    try:
        with socket.create_connection((host, port), timeout=3):
            record(True, item)
    except OSError:
        record(False, item, hint)


async def check_db() -> None:
    """PG 连接 + 关键表 + 种子数据。"""
    from sqlalchemy import text

    from backend.dependencies import AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as session:
            tables = {
                r[0] for r in (await session.execute(text(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public'"))).fetchall()
            }
            for t in ("users", "devices", "parts", "tickets", "after_sale_appointments"):
                record(t in tables, f"PG 表 {t}",
                       "python scripts/migrate.py（服务启动时也会自动建表）")

            row = (await session.execute(text(
                "SELECT COUNT(*) FROM users WHERE username='engineer'"))).fetchone()
            record(row and row[0] > 0, "种子用户 engineer",
                   "python scripts/seed_dev_data.py")

            row = (await session.execute(text(
                "SELECT COUNT(*) FROM devices WHERE device_sn='SN-AC-0001'"))).fetchone()
            record(row and row[0] > 0, "种子设备 SN-AC-0001",
                   "python scripts/seed_dev_data.py")

            row = (await session.execute(text(
                "SELECT COUNT(*) FROM parts"))).fetchall()
            record(bool(row and row[0][0] > 0), "备件种子数据",
                   "python scripts/seed_dev_data.py")
    except Exception as e:
        record(False, "PostgreSQL 连接",
               f"{e} → docker compose -f deploy/docker-compose.yml up -d postgres")


def check_milvus() -> None:
    try:
        from pymilvus import MilvusClient

        from backend.config import get_settings
        s = get_settings()
        client = MilvusClient(uri=f"http://{s.milvus_host}:{s.milvus_port}")
        if not client.has_collection("equipment_knowledge"):
            record(False, "Milvus collection equipment_knowledge",
                   "python scripts/build_knowledge_base.py data/knowledge_sample.md "
                   "--course-id CNC-1000 --no-context")
            return
        # Milvus 重启后 collection 不常驻内存，查询前需 load（同 retriever._ensure_loaded）
        try:
            client.load_collection("equipment_knowledge")
        except Exception:
            pass  # 已加载时部分版本会抛异常，忽略后仍尝试查询
        rows = client.query(
            collection_name="equipment_knowledge",
            filter='course_id == "CNC-1000"',
            output_fields=["id"],
            limit=1,
        )
        record(bool(rows), "示例知识域 CNC-1000（14 chunk）",
               "python scripts/build_knowledge_base.py data/knowledge_sample.md "
               "--course-id CNC-1000 --no-context")
    except Exception as e:
        record(False, "Milvus 连接", f"{e} → docker compose -f deploy/docker-compose.yml up -d")


def check_llm() -> None:
    try:
        from backend.config import get_settings
        key = get_settings().deepseek_api_key
        record(bool(key) and not key.startswith("sk-your"), "DeepSeek API Key 已配置",
               "复制 deploy/.env.example 为 .env.local 并填写")
    except Exception as e:
        record(False, "配置读取", str(e))


def check_docker() -> None:
    import subprocess
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            record(False, "Docker daemon", "先启动 Docker Desktop")
            return
        running = {ln.split("\t")[0]: ln.split("\t")[1]
                   for ln in out.stdout.strip().splitlines() if "\t" in ln}
        for name in ("postgres", "milvus"):
            hit = next((s for n, s in running.items() if name in n), None)
            record(hit is not None, f"Docker 容器 {name}",
                   f"docker compose -f deploy/docker-compose.yml up -d {name}")
    except (OSError, subprocess.TimeoutExpired):
        record(False, "Docker daemon", "先启动 Docker Desktop")


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("═" * 56)
    print(" 演示环境预检（preflight）")
    print("═" * 56)

    check_docker()

    from backend.config import get_settings
    s = get_settings()
    check_tcp(s.db_host, s.db_port, f"PostgreSQL 端口 {s.db_host}:{s.db_port}",
              "docker compose -f deploy/docker-compose.yml up -d postgres")
    check_tcp(s.milvus_host, s.milvus_port, f"Milvus 端口 {s.milvus_host}:{s.milvus_port}",
              "docker compose -f deploy/docker-compose.yml up -d")

    asyncio.run(check_db())
    check_milvus()
    check_llm()

    failed = sum(1 for i, _, _ in results if i == FAIL)
    for icon, item, hint in results:
        line = f" {icon} {item}"
        if hint and icon == FAIL:
            line += f"  → {hint}"
        print(line)

    print("═" * 56)
    if failed:
        print(f" 结果：{failed} 项未就绪。按上面提示修复后重跑。")
        sys.exit(1)
    print(" 结果：全部就绪。启动服务：uvicorn backend.main:app --port 8000")
    print("       演示页面：http://localhost:8000/  （账号 engineer/admin123）")
