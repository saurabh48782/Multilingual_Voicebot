import asyncio
import os
from logging.config import fileConfig
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No ORM models: migrations are hand-written, so autogenerate is unused.
target_metadata = None


def _resolve_dsn() -> str:
    """Reach Postgres via the same ``CHECKPOINT_DSN`` the app uses, rewriting
    the scheme to SQLAlchemy's asyncpg driver so no extra DB driver (psycopg2)
    is needed. ``.env`` is loaded best-effort for local ``uv run alembic``;
    under docker compose the variable is already exported."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv is optional
        pass

    dsn = os.environ.get("CHECKPOINT_DSN")
    if not dsn:
        raise RuntimeError(
            "CHECKPOINT_DSN is not set; Alembic needs it to reach Postgres. "
            "Export it or add it to .env before running migrations."
        )
    parts = urlsplit(dsn)
    if parts.scheme in ("postgresql", "postgres"):
        parts = parts._replace(scheme="postgresql+asyncpg")
    return urlunsplit(parts)


config.set_main_option("sqlalchemy.url", _resolve_dsn())


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
