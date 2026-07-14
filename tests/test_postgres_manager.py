from unittest.mock import MagicMock, patch

import pandas as pd

from src.warehouse.postgres_manager import (
    get_postgres_engine,
    write_outputs_to_postgres,
)


@patch(
    "src.warehouse.postgres_manager.POSTGRES_ENABLED",
    False,
)
def test_get_postgres_engine_returns_none_when_disabled():
    engine = get_postgres_engine()

    assert engine is None


@patch(
    "src.warehouse.postgres_manager.create_engine",
)
@patch(
    "src.warehouse.postgres_manager.POSTGRES_DB",
    "ads_budget",
)
@patch(
    "src.warehouse.postgres_manager.POSTGRES_PORT",
    "5432",
)
@patch(
    "src.warehouse.postgres_manager.POSTGRES_HOST",
    "postgres",
)
@patch(
    "src.warehouse.postgres_manager.POSTGRES_PASSWORD",
    "postgres",
)
@patch(
    "src.warehouse.postgres_manager.POSTGRES_USER",
    "postgres",
)
@patch(
    "src.warehouse.postgres_manager.POSTGRES_ENABLED",
    True,
)
def test_get_postgres_engine_creates_expected_connection(
    mock_create_engine,
):
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine

    result = get_postgres_engine()

    assert result is mock_engine

    mock_create_engine.assert_called_once_with(
    "postgresql+psycopg2://postgres:postgres"
    "@postgres:5432/ads_budget",
    pool_pre_ping=True,
)

def test_write_outputs_returns_when_engine_is_none():
    outputs = {
        "sample_table": pd.DataFrame(
            {
                "CampaignId": [1],
            }
        )
    }

    result = write_outputs_to_postgres(
        outputs=outputs,
        engine=None,
    )

    assert result is None


@patch.object(
    pd.DataFrame,
    "to_sql",
)
def test_write_outputs_writes_non_empty_dataframe(
    mock_to_sql,
):
    engine = MagicMock()

    outputs = {
        "ads_daily_fact": pd.DataFrame(
            {
                "CampaignId": [1, 2],
                "Spend": [100.0, 200.0],
            }
        )
    }

    write_outputs_to_postgres(
        outputs=outputs,
        engine=engine,
        if_exists="replace",
    )

    mock_to_sql.assert_called_once_with(
        "ads_daily_fact",
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )


@patch.object(
    pd.DataFrame,
    "to_sql",
)
def test_write_outputs_skips_empty_dataframe(
    mock_to_sql,
):
    engine = MagicMock()

    outputs = {
        "empty_table": pd.DataFrame(),
    }

    write_outputs_to_postgres(
        outputs=outputs,
        engine=engine,
    )

    mock_to_sql.assert_not_called()


@patch.object(
    pd.DataFrame,
    "to_sql",
)
def test_write_outputs_skips_none_dataframe(
    mock_to_sql,
):
    engine = MagicMock()

    outputs = {
        "none_table": None,
    }

    write_outputs_to_postgres(
        outputs=outputs,
        engine=engine,
    )

    mock_to_sql.assert_not_called()


@patch.object(
    pd.DataFrame,
    "to_sql",
)
def test_write_outputs_processes_multiple_tables(
    mock_to_sql,
):
    engine = MagicMock()

    outputs = {
        "table_one": pd.DataFrame(
            {
                "Value": [1],
            }
        ),
        "table_two": pd.DataFrame(
            {
                "Value": [2],
            }
        ),
    }

    write_outputs_to_postgres(
        outputs=outputs,
        engine=engine,
        if_exists="append",
    )

    assert mock_to_sql.call_count == 2

    first_call = mock_to_sql.call_args_list[0]
    second_call = mock_to_sql.call_args_list[1]

    assert first_call.args[0] == "table_one"
    assert second_call.args[0] == "table_two"

    assert first_call.kwargs["if_exists"] == "append"
    assert second_call.kwargs["if_exists"] == "append"