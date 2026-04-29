"""
Input validation utilities
"""

from typing import Optional
from app.config import settings


def validate_image_file(
    filename: str,
    file_size: int
) -> Optional[str]:
    """
    Validate image file

    Args:
        filename: Name of the file
        file_size: Size of the file in bytes

    Returns:
        Error message if invalid, None if valid
    """
    # Check extension
    if not filename:
        return "Filename is required"

    extension = filename.split('.')[-1].lower()
    if extension not in settings.ALLOWED_EXTENSIONS:
        return f"File extension must be one of: {', '.join(settings.ALLOWED_EXTENSIONS)}"

    # Check file size
    if file_size > settings.MAX_FILE_SIZE:
        return f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE / (1024 * 1024):.1f} MB"

    if file_size == 0:
        return "File is empty"

    return None


def validate_gender(gender: str) -> bool:
    """
    Validate gender parameter

    Args:
        gender: Gender value to validate

    Returns:
        True if valid, False otherwise
    """
    return gender in settings.GENDER_OPTIONS


def validate_category(category: str) -> bool:
    """
    Validate fashion category

    Args:
        category: Category value to validate

    Returns:
        True if valid, False otherwise
    """
    return category in settings.FASHION_CATEGORIES


def validate_integer(
    value: int,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None
) -> bool:
    """
    Validate integer value

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        True if valid, False otherwise
    """
    if min_val is not None and value < min_val:
        return False

    if max_val is not None and value > max_val:
        return False

    return True


def validate_float(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> bool:
    """
    Validate float value

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        True if valid, False otherwise
    """
    if min_val is not None and value < min_val:
        return False

    if max_val is not None and value > max_val:
        return False

    return True
