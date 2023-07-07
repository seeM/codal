from importlib.resources import files

from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from alembic.command import upgrade
from alembic.config import Config

from .settings import settings

engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI, connect_args={"check_same_thread": False}
)

# TODO: Do we need autocommit and autoflush?
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
SessionLocal = sessionmaker(bind=engine)

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


class Base(DeclarativeBase):
    metadata = metadata


def migrate():
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = Config()
    script_location = str(files("codal") / "alembic")
    config.set_main_option("script_location", script_location)
    config.set_main_option("sqlalchemy.url", settings.SQLALCHEMY_DATABASE_URI)
    upgrade(config, "head")
