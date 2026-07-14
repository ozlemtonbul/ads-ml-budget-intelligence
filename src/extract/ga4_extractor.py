from pathlib import Path

import pandas as pd
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
)
from google.oauth2 import service_account

from config.settings import (
    GA4_PROPERTY_ID,
    GA4_SERVICE_ACCOUNT_FILE,
    require_env,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class GA4Extractor:
    def __init__(self):
        property_id = require_env(GA4_PROPERTY_ID, "GA4_PROPERTY_ID")
        service_account_file = require_env(
            GA4_SERVICE_ACCOUNT_FILE,
            "GA4_SERVICE_ACCOUNT_FILE",
        )

        credentials_path = Path(service_account_file)

        if not credentials_path.exists():
            raise FileNotFoundError(
                f"GA4 service account file was not found: {credentials_path}"
            )

        self.credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path)
        )

        self.client = BetaAnalyticsDataClient(
            credentials=self.credentials
        )

        self.property_name = f"properties/{property_id}"

    def fetch_campaign_performance(
        self,
        date_from: str,
        date_to: str,
    ) -> pd.DataFrame:
        request = RunReportRequest(
            property=self.property_name,
            dimensions=[
                Dimension(name="date"),
                Dimension(name="sessionCampaignName"),
                Dimension(name="sessionSourceMedium"),
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="totalUsers"),
                Metric(name="engagedSessions"),
                Metric(name="ecommercePurchases"),
                Metric(name="purchaseRevenue"),
            ],
            date_ranges=[
                DateRange(
                    start_date=date_from,
                    end_date=date_to,
                )
            ],
        )

        try:
            response = self.client.run_report(request)
        except Exception as exc:
            logger.error("GA4 Data API error: %s", exc)
            raise

        rows = []

        for row in response.rows:
            rows.append(
                {
                    "Date": row.dimension_values[0].value,
                    "Campaign": row.dimension_values[1].value or "UNKNOWN",
                    "SourceMedium": row.dimension_values[2].value or "UNKNOWN",
                    "Sessions": float(row.metric_values[0].value or 0),
                    "TotalUsers": float(row.metric_values[1].value or 0),
                    "EngagedSessions": float(row.metric_values[2].value or 0),
                    "Purchases": float(row.metric_values[3].value or 0),
                    "PurchaseRevenue": float(row.metric_values[4].value or 0),
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            logger.warning(
                "GA4 returned no campaign data for %s to %s.",
                date_from,
                date_to,
            )
            return df

        df["Date"] = pd.to_datetime(
            df["Date"],
            format="%Y%m%d",
            errors="coerce",
        )

        df["GA4ConversionRate"] = (
            df["Purchases"]
            / df["Sessions"].replace(0, pd.NA)
        )

        df["GA4RevenuePerSession"] = (
            df["PurchaseRevenue"]
            / df["Sessions"].replace(0, pd.NA)
        )

        df["EngagementRate"] = (
            df["EngagedSessions"]
            / df["Sessions"].replace(0, pd.NA)
        )

        return df.fillna(0)