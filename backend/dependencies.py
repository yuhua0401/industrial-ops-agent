"""
dependencies - 依赖注入

Author:hiema1
Date:2026/8/10
Version:0.0.1
"""
# FastAPI 依赖注入：① 数据库会话 get_db  ② 当前用户鉴权 get_current_user

from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import (  # 解析 Authorization: Bearer 头
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from jose import JWTError, jwt  # python-jose：JWT 的编解码库
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config import get_settings

settings = get_settings()

# ── PostgreSQL 异步连接池（与 3.2 相同）──────────────────────────
engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    echo=False,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：获取异步数据库会话，自动提交 / 回滚 / 关闭。"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── JWT 鉴权 ───────────────────────────────────────────────────
# auto_error=False：缺少 Authorization 头时返回 None，由下方统一抛 401
# （FastAPI 的 HTTPBearer 默认缺头返回 403，不符合 401 语义）
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    # Depends(bearer_scheme)：自动取出 Bearer Token；没带或格式错会得到 None
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    """FastAPI 依赖：验证 JWT Token，返回当前用户信息。
    返回 {"user_id": str, "role": str, "tenant_id": str}；Token 无效则抛 401。"""
    # 预先准备好「401 凭证无效」异常，多处复用
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:                  # 未携带 Token → 统一 401
        raise credentials_exception
    try:
        # 用密钥和算法解码 Token；若签名不对/过期，会抛 JWTError
        payload = jwt.decode(
            credentials.credentials,            # 实际的 token 字符串
            settings.jwt_secret_key,            # 验签密钥（和签发时同一个）
            algorithms=[settings.jwt_algorithm],
        )
        user_id: str = payload.get("sub")                                  # 标准字段 sub = 用户ID
        role: str = payload.get("role", "customer")                        # 角色（默认 customer，对齐 User 模型）
        tenant_id: str = payload.get("tenant_id", settings.default_tenant_id)  # 租户

        if not user_id:                         # Token 里没有用户ID，视为无效
            raise credentials_exception

    except JWTError as exc:                     # 解码失败（签名错/过期等）
        raise credentials_exception from exc

    return {"user_id": user_id, "role": role, "tenant_id": tenant_id}
