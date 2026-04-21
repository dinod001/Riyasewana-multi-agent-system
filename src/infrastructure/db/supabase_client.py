"""
Supabase client — REST API + SQLAlchemy helpers.

Provides:
- ``get_supabase_client()``  — Supabase REST/Auth/Realtime client
- ``get_supabase_engine()``  — Delegates to sql_client.get_sql_engine()
- ``get_supabase_session()`` — Delegates to sql_client.get_session()
- Utility functions: test_connection, pgvector check, schema init, RLS, etc.

The canonical SQLAlchemy engine/session lives in ``sql_client.py``.
This module re-exports those functions under Supabase-prefixed names
for convenience so existing imports continue to work.
"""

import os
from loguru import logger
from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from supabase import create_client, Client

from infrastructure.config import EMBEDDING_DIM

# Canonical engine/session — single source of truth
from infrastructure.db.sql_client import get_sql_engine, get_session
# ---------------------------------------------------------------------------
# Supabase REST client (Auth, Realtime, Storage)
# ---------------------------------------------------------------------------

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Get or create Supabase REST client (for Auth, Realtime, Storage).

    Returns:
        Supabase Client instance
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file"
        )

    _supabase_client = create_client(supabase_url, supabase_key)
    logger.info(f"✓ Supabase client created: {supabase_url}")

    return _supabase_client


# ---------------------------------------------------------------------------
# SQLAlchemy delegates — thin wrappers over sql_client
# ---------------------------------------------------------------------------


def get_supabase_engine():
    """
    Return the canonical SQLAlchemy engine (delegates to sql_client).

    Kept for backward-compatible imports.
    """
    return get_sql_engine()


def get_supabase_session() -> Session:
    """
    Return a new SQLAlchemy session (delegates to sql_client).

    Kept for backward-compatible imports.
    """
    return get_session()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def test_connection() -> bool:
    """Test Supabase PostgreSQL connection."""
    try:
        engine = get_sql_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

        logger.info("✅ Supabase connection test: SUCCESS")
        return True

    except Exception as e:
        logger.error(f"❌ Supabase connection test: FAILED - {e}")
        return False


def check_pgvector_installed() -> bool:
    """Check if pgvector extension is installed in Supabase."""
    try:
        engine = get_sql_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            installed = result.scalar() == "vector"

        if installed:
            logger.info("✅ pgvector extension: INSTALLED")
        else:
            logger.warning("⚠️  pgvector extension: NOT INSTALLED")
            logger.warning("   Run in Supabase SQL Editor: CREATE EXTENSION vector;")

        return installed

    except Exception as e:
        logger.error(f"❌ Failed to check pgvector: {e}")
        return False


def set_user_context(user_id: str):
    """
    Set user context for Row Level Security (RLS).

    Args:
        user_id: User identifier
    """
    engine = get_sql_engine()

    with engine.connect() as conn:
        conn.execute(text(f"SET app.user_id = '{user_id}'"))



# ---------------------------------------------------------------------------
# Health check on import
# ---------------------------------------------------------------------------

try:
    if test_connection():
        check_pgvector_installed()
except Exception as e:
    logger.warning(f"Supabase not fully configured: {e}")
    logger.info("Set SUPABASE_URL, SUPABASE_ANON_KEY, and SUPABASE_DB_URL in .env")
