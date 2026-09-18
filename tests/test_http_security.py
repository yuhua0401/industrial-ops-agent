"""
HTTP 层安全与可观测性测试（企业级加固批次）。

用 httpx ASGITransport 直连 FastAPI app（不触发 lifespan，不依赖
真实 PG/Milvus/LLM）；鉴权 Token 用配置密钥现场签发（get_current_user
只解码不查库）。

覆盖：
- 鉴权：chat/stream 无 token → 401；伪造 token → 401；/health 公开
- 校验：带合法 token + 非法请求体 → 422 VALIDATION_ERROR 统一体
- 全局异常：未捕获异常 → 500 INTERNAL_ERROR 统一体（不泄漏堆栈）
- Request-ID：响应头回传 X-Request-ID
- readyz：依赖探针 mock 后 200/503 两态
- 限流：纯 ASGI 单元测试，窗口内超限 → 429
"""
import sys
from pathlib import Path

import httpx
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from httpx import ASGITransport, AsyncClient  # noqa: E402
from jose import jwt  # noqa: E402


@pytest.fixture()
def app():
    from backend.main import app as fastapi_app
    return fastapi_app


@pytest.fixture()
def client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _make_token() -> str:
    from backend.config import get_settings
    s = get_settings()
    return jwt.encode(
        {"sub": "999", "role": "admin", "tenant_id": "tenant_default"},
        s.jwt_secret_key, algorithm=s.jwt_algorithm,
    )


@pytest.mark.asyncio
async def test_health_public(client):
    """/health 存活探针保持公开。"""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_chat_requires_token(client):
    """无 token 访问 chat → 401（此前该端点裸奔）。"""
    resp = await client.post(
        "/api/v1/chat/stream",
        json={"session_id": "t", "message": "你好"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_rejects_forged_token(client):
    """伪造签名 token → 401。"""
    from backend.config import get_settings
    s = get_settings()
    forged = jwt.encode(
        {"sub": "1", "role": "admin"}, "wrong-secret-key", algorithm=s.jwt_algorithm)
    resp = await client.post(
        "/api/v1/chat/stream",
        json={"session_id": "t", "message": "你好"},
        headers={"Authorization": f"Bearer {forged}"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_validation_error_shape(client):
    """带合法 token + 非法请求体 → 422 VALIDATION_ERROR 统一体。"""
    resp = await client.post(
        "/api/v1/chat/stream",
        json={"session_id": "t"},          # 缺 message
        headers={"Authorization": f"Bearer {_make_token()}"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert isinstance(body["detail"], list)


@pytest.mark.asyncio
async def test_request_id_header(client):
    """响应头回传 X-Request-ID。"""
    resp = await client.get("/health")
    assert resp.headers.get("x-request-id")


@pytest.mark.asyncio
async def test_readyz_ok_and_degraded(app, monkeypatch):
    """readyz：探针全过 → 200；任一失败 → 503 degraded。"""
    async def ok_db():
        return None

    async def ok_milvus():
        return None

    async def bad_milvus():
        raise RuntimeError("milvus down")

    monkeypatch.setattr("backend.main._probe_db", ok_db)
    monkeypatch.setattr("backend.main._probe_milvus", ok_milvus)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["checks"]["postgres"] == "ok"

    monkeypatch.setattr("backend.main._probe_milvus", bad_milvus)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    # 就绪探针只暴露异常类型，不泄漏内部错误细节
    assert "RuntimeError" in body["checks"]["milvus"]


@pytest.mark.asyncio
async def test_global_exception_handler(app):
    """未捕获异常 → 500 INTERNAL_ERROR 统一体（不泄漏堆栈）。

    直接调用注册在 app 上的 handler（动态加路由会被 "/" 静态挂载遮蔽）。
    """
    from backend.main import unhandled_exception_handler

    scope = {"type": "http", "method": "GET", "path": "/x", "headers": [],
             "query_string": b"", "request_id": "t-123"}

    async def receive():
        return {"type": "http.request", "body": b""}

    sent = {}

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
        elif message["type"] == "http.response.body":
            sent.setdefault("body", b"") + message.get("body", b"")
            sent["body"] = sent.get("body", b"") + message.get("body", b"")

    class _Req:
        url = type("U", (), {"path": "/x"})()

        def __init__(self, scope):
            self.scope = scope

    resp = await unhandled_exception_handler(_Req(scope), RuntimeError("secret internal detail"))
    await resp(scope, receive, send)

    assert sent["status"] == 500
    body = resp.body.decode()
    assert "INTERNAL_ERROR" in body
    assert "secret" not in body   # 内部细节不外泄


# ──────────────────────────────────────────────────────────────
# 限流中间件（纯 ASGI 单元测试）
# ──────────────────────────────────────────────────────────────

class _StubApp:
    """透传 app：记录被调用次数并返回 200。"""

    def __init__(self):
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})


def _make_scope(path: str = "/api/v1/chat/stream", ip: str = "1.2.3.4") -> dict:
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [(b"x-forwarded-for", ip.encode())],
        "client": (ip, 12345),
        "query_string": b"",
    }


async def _drive(mw, scope) -> int:
    """驱动一次 ASGI 调用，返回响应状态码。"""
    status = {"code": 0}

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]

    await mw(scope, receive, send)
    return status["code"]


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit():
    """窗口内超过上限 → 429；豁免路径不计数。"""
    from backend.core.middleware import RateLimitMiddleware

    stub = _StubApp()
    mw = RateLimitMiddleware(stub, limit_per_minute=3)
    scope = _make_scope()

    for _ in range(3):
        assert await _drive(mw, dict(scope)) == 200
    assert stub.calls == 3
    # 第 4 次被拒
    assert await _drive(mw, dict(scope)) == 429
    assert stub.calls == 3

    # /health 豁免：不计数，即使达到上限也放行
    health_scope = _make_scope(path="/health")
    assert await _drive(mw, dict(health_scope)) == 200
    assert await _drive(mw, dict(health_scope)) == 200


@pytest.mark.asyncio
async def test_rate_limit_per_ip_isolated():
    """不同 IP 计数隔离。"""
    from backend.core.middleware import RateLimitMiddleware

    mw = RateLimitMiddleware(_StubApp(), limit_per_minute=1)
    assert await _drive(mw, _make_scope(ip="1.1.1.1")) == 200
    assert await _drive(mw, _make_scope(ip="1.1.1.1")) == 429
    assert await _drive(mw, _make_scope(ip="2.2.2.2")) == 200
