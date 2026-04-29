"""
Authentication Schemas
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Token(BaseModel):
    """JWT Token response"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None
    scopes: List[str] = []


class UserBase(BaseModel):
    """Base user schema"""
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[str] = None  # Use str instead of EmailStr to avoid dependency
    full_name: Optional[str] = None
    disabled: bool = False


class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8)


class UserInDB(UserBase):
    """User stored in database"""
    hashed_password: str


class UserResponse(UserBase):
    """User response schema (without password)"""
    pass


class LoginRequest(BaseModel):
    """Login request schema"""
    username: str
    password: str
