"""
Authentication endpoints
"""

import logging
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.config import settings
from app.middleware.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    require_auth,
)
from app.schemas.auth import LoginRequest, Token, UserInDB, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """
    OAuth2 compatible token login, get an access token for future requests.

    Args:
        form_data: OAuth2 password request form

    Returns:
        JWT access token
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )

    return Token(access_token=access_token, token_type="bearer")


@router.post("/login", response_model=Token)
async def login(request: LoginRequest) -> Token:
    """
    JSON login endpoint (alternative to OAuth2 form).

    Args:
        request: Login request with username and password

    Returns:
        JWT access token
    """
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )

    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: UserInDB = Depends(require_auth)
) -> UserResponse:
    """
    Get current authenticated user info.

    Args:
        current_user: Current authenticated user

    Returns:
        User information
    """
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        disabled=current_user.disabled
    )


@router.get("/verify")
async def verify_token_endpoint(
    current_user: Optional[UserInDB] = Depends(get_current_user)
) -> dict:
    """
    Verify if the current token is valid.

    Args:
        current_user: Current user (if authenticated)

    Returns:
        Verification status
    """
    if current_user:
        return {
            "valid": True,
            "username": current_user.username,
            "message": "Token is valid"
        }
    return {
        "valid": False,
        "username": None,
        "message": "No valid token provided"
    }
