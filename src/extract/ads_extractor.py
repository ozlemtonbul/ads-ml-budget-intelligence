import pandas as pd
from google.ads.googleads.errors import GoogleAdsException

from config.google_ads_client import get_google_ads_client
from config.settings import GOOGLE_ADS_CUSTOMER_ID, require_env
from src.utils.logger import get_logger


logger = get_logger(__name__)


def extract_category(adgroup_name: str) -> str:
    adgroup_name = str(adgroup_name or "").strip()

    if "|" in adgroup_name:
        return adgroup_name.split("|")[0].strip()

    return (
        adgroup_name.split()[0].strip()
        if adgroup_name
        else "UNKNOWN"
    )


def extract_product(adgroup_name: str) -> str:
    adgroup_name = str(adgroup_name or "").strip()

    if "|" in adgroup_name:
        return adgroup_name.split("|", 1)[1].strip()

    return adgroup_name if adgroup_name else "UNKNOWN"


def fetch_ads_purchase_only(
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    customer_id = require_env(
        GOOGLE_ADS_CUSTOMER_ID,
        "GOOGLE_ADS_CUSTOMER_ID",
    )

    client = get_google_ads_client()
    service = client.get_service("GoogleAdsService")

    query = f"""
        SELECT
          segments.date,
          campaign.id,
          campaign.name,
          campaign.advertising_channel_type,
          ad_group.id,
          ad_group.name,
          metrics.impressions,
          metrics.clicks,
          metrics.conversions,
          metrics.conversions_value,
          metrics.cost_micros
        FROM ad_group
        WHERE segments.date BETWEEN '{date_from}' AND '{date_to}'
    """

    rows = []

    try:
        response = service.search(
            customer_id=customer_id,
            query=query,
        )

        for row in response:
            rows.append(
                {
                    "Date": str(row.segments.date),
                    "CampaignId": int(
                        row.campaign.id or 0
                    ),
                    "Campaign": (
                        row.campaign.name
                        or "UNKNOWN"
                    ),
                    "Channel": str(
                        row.campaign.advertising_channel_type
                    ),
                    "AdGroupId": int(
                        row.ad_group.id or 0
                    ),
                    "AdGroup": (
                        row.ad_group.name
                        or "UNKNOWN"
                    ),
                    "Impressions": int(
                        row.metrics.impressions or 0
                    ),
                    "Clicks": int(
                        row.metrics.clicks or 0
                    ),
                    "Conversions": float(
                        row.metrics.conversions or 0.0
                    ),
                    "ConversionValue": float(
                        row.metrics.conversions_value or 0.0
                    ),
                    "Spend": (
                        float(
                            row.metrics.cost_micros or 0.0
                        )
                        / 1_000_000.0
                    ),
                }
            )

    except GoogleAdsException as exc:
        logger.error(
            "Google Ads API error: %s",
            exc,
        )
        raise

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        logger.warning(
            "Google Ads API returned no data for %s to %s.",
            date_from,
            date_to,
        )
        return dataframe

    dataframe["Date"] = pd.to_datetime(
        dataframe["Date"],
        errors="coerce",
    )

    numeric_columns = [
        "CampaignId",
        "AdGroupId",
        "Impressions",
        "Clicks",
        "Conversions",
        "ConversionValue",
        "Spend",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).fillna(0)

    dataframe["Category"] = (
        dataframe["AdGroup"]
        .apply(extract_category)
    )

    dataframe["ProductGroup"] = (
        dataframe["AdGroup"]
        .apply(extract_product)
    )

    logger.info(
        "Google Ads data loaded successfully. Rows: %d",
        len(dataframe),
    )

    return dataframe


class AdsExtractor:
    def __init__(self):
        self.client = None

    def get_client(self):
        if self.client is None:
            self.client = get_google_ads_client()

        return self.client

    def fetch_purchase_data(
        self,
        date_from: str,
        date_to: str,
    ) -> pd.DataFrame:
        return fetch_ads_purchase_only(
            date_from,
            date_to,
        )