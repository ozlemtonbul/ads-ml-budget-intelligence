from typing import Dict

import pandas as pd
from sqlalchemy import create_engine

from config.settings import (
    POSTGRES_DB,
    POSTGRES_ENABLED,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_postgres_engine():
    """
    Create PostgreSQL SQLAlchemy engine.
    """

    if not POSTGRES_ENABLED:
        logger.info("PostgreSQL export is disabled.")
        return None

    required = {
        "POSTGRES_USER": POSTGRES_USER,
        "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
        "POSTGRES_HOST": POSTGRES_HOST,
        "POSTGRES_PORT": POSTGRES_PORT,
        "POSTGRES_DB": POSTGRES_DB,
    }

    missing = [
        key
        for key, value in required.items()
        if value in (None, "")
    ]

    if missing:
        raise ValueError(
            f"Missing PostgreSQL configuration: {', '.join(missing)}"
        )

    connection_url = (
        f"postgresql+psycopg2://"
        f"{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    engine = create_engine(
        connection_url,
        pool_pre_ping=True,
    )

    logger.info("PostgreSQL engine created successfully.")

    return engine


def write_outputs_to_postgres(
    outputs: Dict[str, pd.DataFrame],
    engine,
    if_exists: str = "replace",
) -> None:
    """
    Write output DataFrames into PostgreSQL.
    """

    if engine is None:
        logger.info("PostgreSQL engine not available. Export skipped.")
        return

    if not outputs:
        logger.warning("No output tables supplied.")
        return

    for table_name, df in outputs.items():

        if df is None or df.empty:
            logger.warning(
                "Skipping table '%s' because dataframe is empty.",
                table_name,
            )
            continue

        try:

            df.to_sql(
                table_name,
                engine,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=1000,
            )

            logger.info(
                "Table '%s' exported successfully (%d rows).",
                table_name,
                len(df),
            )

        except Exception as exc:

            logger.exception(
                "Failed to export table '%s': %s",
                table_name,
                exc,
            )

            raise

    logger.info("PostgreSQL export completed successfully.")