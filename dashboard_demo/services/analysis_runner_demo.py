from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dashboard_demo.app_config import OUTPUT_DIR


@dataclass(frozen=True)
class AnalysisRunResult:
    success: bool
    message: str
    run_directory: Path | None = None
    log_tail: str = ""


def get_source_status() -> dict[str, bool]:
    """
    Public demo sürümünde gerçek API bağlantısı kullanılmaz.

    Dashboard tarafındaki mevcut durum kontrollerinin hata vermemesi için
    kaynaklar demo ortamında kullanılabilir olarak gösterilir.
    """
    return {
        "google_ads": True,
        "ga4": True,
    }


def run_analysis_for_period(
    start_date: date,
    end_date: date,
    timeout_seconds: int = 1800,
) -> AnalysisRunResult:
    """
    Public demo tarih seçimini işler.

    Bu fonksiyon:
    - main.py dosyasını çalıştırmaz.
    - Google Ads API çağrısı yapmaz.
    - GA4 API çağrısı yapmaz.
    - demo_data klasörüne gerçek veri yazmaz.
    - Mevcut anonim demo dosyalarının kullanılmasını sağlar.
    """
    del timeout_seconds

    normalized_start = min(start_date, end_date)
    normalized_end = max(start_date, end_date)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_path = OUTPUT_DIR / "ads_daily_fact.csv"

    if not daily_path.exists():
        return AnalysisRunResult(
            success=False,
            message=(
                "Anonim demo verisi bulunamadı. "
                "Önce scripts/build_public_demo.py dosyasını çalıştırın."
            ),
            run_directory=OUTPUT_DIR,
        )

    return AnalysisRunResult(
        success=True,
        message=(
            f"{normalized_start:%d.%m.%Y}–"
            f"{normalized_end:%d.%m.%Y} dönemi için "
            "anonim demo verileri kullanılıyor. "
            "Canlı Google Ads veya GA4 API çağrısı yapılmadı."
        ),
        run_directory=OUTPUT_DIR,
    )


