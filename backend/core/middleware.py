"""
middleware - 生产安全与可观测中间件

- RequestLoggingMiddleware：每请求生成 X-Request-ID（透传优先）→ 响应头回写
  + 结构化访问日志（method/path/status/duration_ms/request_id）
- RateLimitMiddleware：内存滑动窗口 per-IP 限流（单实例部署够用；
  多副本部署应换网关或 Redis，见 docs/production-checklist.md）

注意顺序：Request-IP 获取在限流之前；两者都基于纯 ASGI 实现，
比 BaseHTTPMiddleware 少一层任务包裹，对流式响应（SSE）友好。
"""
import json
import time
import uuid
from collections import defaultdict, deque

from backend.core.logger import get_logger

logger = get_logger(__name__)

# 命中即豁免的前缀（存活探针高频拉取，不计数、不记访问日志）
_EXEMPT_PREFIXES = ("/health", "/readyz")


def get_client_ip(scope: dict) -> str:
    """从 ASGI scope 提取客户端 IP（优先 X-Forwarded-For 首段，适配反代部署）。"""
    headers = {
        k.decode("latin-1").lower(): v.decode("latin-1")
        for k, v in scope.get("headers", [])
    }
    forwarded = headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


class RequestLoggingMiddleware:
    """请求 ID 注入 + 访问日志。

    用法：app.add_middleware(RequestLoggingMiddleware)
    上游可通过 X-Request-ID 请求头传入追踪 ID（网关场景），否则生成 8 位短码。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        request_id = (headers.get("x-request-id") or uuid.uuid4().hex[:8])[:32]
        scope["request_id"] = request_id
        path = scope.get("path", "")
        status_holder = {"status": 0}

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message.get("status", 0)
                raw_headers = list(message.get("headers", []))
                raw_headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": raw_headers}
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            # http.response.start 可能未到达（客户端断开），兜底记 0
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            if not any(path.startswith(p) for p in _EXEMPT_PREFIXES):
                logger.info(
                    "http.request",
                    method=scope.get("method", ""),
                    path=path,
                    status=status_holder["status"] or "-",
                    duration_ms=duration_ms,
                    request_id=request_id,
                    ip=get_client_ip(scope),
                )


class RateLimitMiddleware:
    """内存滑动窗口限流（per-IP）。

    用法：app.add_middleware(RateLimitMiddleware, limit_per_minute=120)
    超限返回 429 JSON {"code": "RATE_LIMITED", ...}。
    单进程内存实现，重启清零；多副本部署请使用网关限流或 Redis。
    """

    def __init__(self, app, limit_per_minute: int = 120):
        self.app = app
        self.limit = max(int(limit_per_minute), 1)
        self._window_ms = 60_000
        self._hits: dict[str, deque] = defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            await self.app(scope, receive, send)
            return

        ip = get_client_ip(scope)
        now_ms = time.monotonic() * 1000
        window = self._hits[ip]
        while window and window[0] <= now_ms - self._window_ms:
            window.popleft()

        if len(window) >= self.limit:
            logger.warning("http.rate_limited", ip=ip, path=path, limit=self.limit)
            await self._reject(send, ip)
            return

        window.append(now_ms)
        await self.app(scope, receive, send)

    async def _reject(self, send, ip: str) -> None:
        body = json.dumps(
            {"code": "RATE_LIMITED", "message": "请求过于频繁，请稍后再试"},
            ensure_ascii=False,
        ).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"retry-after", b"60"),
            ],
        })
        await send({"type": "http.response.body", "body": body})
