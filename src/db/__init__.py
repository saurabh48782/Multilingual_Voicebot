"""Database access layer: persistence helpers over the asyncpg pool.

Schema (DDL) is owned by Alembic migrations (``alembic/versions/``), not by
runtime code - ``scripts/start_api.sh`` runs ``alembic upgrade head`` before
the server starts. This package holds only the query helpers.
"""
