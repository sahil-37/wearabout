"""
JWT Authentication Middleware
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt

from app.config import settings
from app.schemas.auth import TokenData, UserInDB

logger = logging.getLogger(__name__)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)

# HTTP Bearer scheme (for API key style auth)
bearer_scheme = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token

    Args:
        data: Payload data to encode
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode a JWT token

    Args:
        token: JWT token string

    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            return None

        scopes = payload.get("scopes", [])
        return TokenData(username=username, scopes=scopes)

    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None


# Pre-computed bcrypt hashes for local dev (admin123, demo123)
_ADMIN_HASH = "$2b$12$4zIvV23TKWpsIkxf2ypSCORr.bJBXa2CU/yaNZPbZtA6424FpV9FC"
_DEMO_HASH  = "$2b$12$L85fRH94x92cfkJPRCIMO.rxRh7e3ekbRori12vxlbFjBzPN4To5y"

# Fake users database (replace with real database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "disabled": False,
        "hashed_password": _ADMIN_HASH,
    },
    "demo": {
        "username": "demo",
        "email": "demo@example.com",
        "full_name": "Demo User",
        "disabled": False,
        "hashed_password": _DEMO_HASH,
    }
}


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database"""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Authenticate a user with username and password

    Args:
        username: User's username
        password: User's password

    Returns:
        UserInDB if authenticated, None otherwise
    """
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[UserInDB]:
    """
    Get current user from JWT token (optional auth)

    Args:
        token: JWT token from Authorization header

    Returns:
        UserInDB if authenticated, None otherwise
    """
    if not token:
        return None

    token_data = verify_token(token)
    if not token_data:
        return None

    user = get_user(token_data.username)
    return user


async def require_auth(
    token: Optional[str] = Depends(oauth2_scheme)
) -> UserInDB:
    """
    Require authentication (raises 401 if not authenticated)

    Args:
        token: JWT token from Authorization header

    Returns:
        UserInDB

    Raises:
        HTTPException: If not authenticated
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    token_data = verify_token(token)
    if not token_data:
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    return user


async def get_current_user_from_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> Optional[UserInDB]:
    """
    Alternative: Get current user from Bearer token

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        UserInDB if authenticated, None otherwise
    """
    if not credentials:
        return None

    token_data = verify_token(credentials.credentials)
    if not token_data:
        return None

    return get_user(token_data.username)


# API Key authentication (simpler alternative)
API_KEYS = {
    "test-api-key-123": "demo",
    "admin-api-key-456": "admin",
}


async def get_api_key_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> Optional[UserInDB]:
    """
    Authenticate via API key

    Args:
        credentials: HTTP Bearer credentials containing API key

    Returns:
        UserInDB if valid API key, None otherwise
    """
    if not credentials:
        return None

    api_key = credentials.credentials
    if api_key in API_KEYS:
        username = API_KEYS[api_key]
        return get_user(username)

    return None
