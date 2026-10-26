"""Registration checks and password hashing (Werkzeug, never plaintext)."""

import re

from werkzeug.security import check_password_hash, generate_password_hash

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    return generate_password_hash(password, method="pbkdf2:sha256")


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


def validate_registration(username: str, email: str, password: str, confirm: str) -> dict[str, str]:
    errors: dict[str, str] = {}
    if not USERNAME_RE.match(username):
        errors["username"] = "Username must be 3-30 characters: letters, numbers, dot, dash or underscore."
    if len(email) > 120 or not EMAIL_RE.match(email):
        errors["email"] = "Please enter a valid email address."
    if len(password) < 8:
        errors["password"] = "Password must be at least 8 characters."
    elif password != confirm:
        errors["confirm"] = "The two passwords do not match."
    return errors
