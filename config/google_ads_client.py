from google.ads.googleads.client import GoogleAdsClient

from config.settings import (
    GOOGLE_ADS_CLIENT_ID,
    GOOGLE_ADS_CLIENT_SECRET,
    GOOGLE_ADS_DEVELOPER_TOKEN,
    GOOGLE_ADS_LOGIN_CUSTOMER_ID,
    GOOGLE_ADS_REFRESH_TOKEN,
    require_env,
)


def get_google_ads_client() -> GoogleAdsClient:
    """
    Create and return a Google Ads API client.
    """

    config = {
        "developer_token": require_env(
            GOOGLE_ADS_DEVELOPER_TOKEN,
            "GOOGLE_ADS_DEVELOPER_TOKEN",
        ),
        "client_id": require_env(
            GOOGLE_ADS_CLIENT_ID,
            "GOOGLE_ADS_CLIENT_ID",
        ),
        "client_secret": require_env(
            GOOGLE_ADS_CLIENT_SECRET,
            "GOOGLE_ADS_CLIENT_SECRET",
        ),
        "refresh_token": require_env(
            GOOGLE_ADS_REFRESH_TOKEN,
            "GOOGLE_ADS_REFRESH_TOKEN",
        ),
        "login_customer_id": require_env(
            GOOGLE_ADS_LOGIN_CUSTOMER_ID,
            "GOOGLE_ADS_LOGIN_CUSTOMER_ID",
        ),
        "use_proto_plus": True,
    }

    try:
        client = GoogleAdsClient.load_from_dict(config)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Google Ads client."
        ) from exc

    return client


if __name__ == "__main__":
    client = get_google_ads_client()
    print("✅ Google Ads Client created successfully.")