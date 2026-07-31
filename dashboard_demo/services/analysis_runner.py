from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config.settings import ga4_ready, google_ads_ready
from dashboard_demo.app_config import OUTPUT_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class AnalysisRunResult:
    success: bool
    message: str
    run_directory: Path | None = None
    log_tail: str = ""


def get_source_status() -> dict[str, bool]:
    """Return configuration readiness without exposing credentials."""
    return {
        "google_ads": google_ads_ready(),
        "ga4": ga4_ready(),
    }


def _period_matches(
    dataframe: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> bool:
    if dataframe.empty:
        return False

    if not {
        "AnalysisStartDate",
        "AnalysisEndDate",
    }.issubset(dataframe.columns):
        return False

    starts = pd.to_datetime(
        dataframe["AnalysisStartDate"],
        errors="coerce",
    ).dropna()
    ends = pd.to_datetime(
        dataframe["AnalysisEndDate"],
        errors="coerce",
    ).dropna()

    if starts.empty or ends.empty:
        return False

    return (
        starts.dt.date.eq(start_date).all()
        and ends.dt.date.eq(end_date).all()
    )


def _validate_staged_outputs(
    staging_directory: Path,
    start_date: date,
    end_date: date,
) -> tuple[bool, str]:
    daily_path = staging_directory / "ads_daily_fact.csv"

    if not daily_path.exists():
        return False, "Google Ads günlük veri çıktısı oluşturulamadı."

    try:
        daily = pd.read_csv(daily_path)
    except Exception as exc:
        return False, f"Günlük veri çıktısı okunamadı: {exc}"

    if daily.empty or "Date" not in daily.columns:
        return False, "Seçilen dönem için kullanılabilir günlük veri bulunamadı."

    daily_dates = pd.to_datetime(
        daily["Date"],
        errors="coerce",
    ).dropna()

    if daily_dates.empty:
        return False, "Günlük veri çıktısında geçerli tarih bulunamadı."

    if (
        daily_dates.min().date() < start_date
        or daily_dates.max().date() > end_date
    ):
        return False, "API çıktısındaki tarihler seçilen dönemle uyuşmuyor."

    dated_output_names = (
        "ads_budget_optimization_recommendations.csv",
        "ads_rule_based_fallback_recommendations.csv",
    )

    matching_recommendation_found = False

    for file_name in dated_output_names:
        file_path = staging_directory / file_name

        if not file_path.exists():
            continue

        try:
            frame = pd.read_csv(file_path)
        except Exception:
            continue

        if _period_matches(frame, start_date, end_date):
            matching_recommendation_found = True
            break

    if not matching_recommendation_found:
        return (
            False,
            "Seçilen döneme ait öneri veya fallback çıktısı oluşturulamadı.",
        )

    return True, ""


def _write_manifest(
    directory: Path,
    start_date: date,
    end_date: date,
) -> None:
    manifest = {
        "analysis_start_date": start_date.isoformat(),
        "analysis_end_date": end_date.isoformat(),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "source": "dashboard",
    }

    (directory / "analysis_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _promote_outputs(
    staging_directory: Path,
    start_date: date,
    end_date: date,
) -> Path:
    period_key = (
        f"{start_date.isoformat()}_{end_date.isoformat()}"
    )
    run_directory = OUTPUT_DIR / "runs" / period_key
    run_directory.mkdir(parents=True, exist_ok=True)

    for source in staging_directory.iterdir():
        if not source.is_file():
            continue

        shutil.copy2(
            source,
            run_directory / source.name,
        )
        shutil.copy2(
            source,
            OUTPUT_DIR / source.name,
        )

    _write_manifest(
        run_directory,
        start_date,
        end_date,
    )
    _write_manifest(
        OUTPUT_DIR,
        start_date,
        end_date,
    )

    return run_directory


def run_analysis_for_period(
    start_date: date,
    end_date: date,
    timeout_seconds: int = 1800,
) -> AnalysisRunResult:
    """
    Run the existing pipeline for the selected dashboard period.

    Outputs are generated in a staging directory first. Existing dashboard
    outputs are replaced only after the staged run passes date validation.
    """
    original_start = start_date
    original_end = end_date
    start_date = min(original_start, original_end)
    end_date = max(original_start, original_end)

    source_status = get_source_status()

    if not source_status["google_ads"]:
        return AnalysisRunResult(
            success=False,
            message=(
                "Google Ads API yapılandırması eksik. "
                "Önce Google Ads erişim bilgilerini kontrol edin."
            ),
        )

    if not source_status["ga4"]:
        return AnalysisRunResult(
            success=False,
            message=(
                "GA4 API yapılandırması eksik. "
                "Önce GA4 property ve service-account ayarlarını kontrol edin."
            ),
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = OUTPUT_DIR / ".staging"
    staging_directory = staging_root / uuid.uuid4().hex
    staging_directory.mkdir(parents=True, exist_ok=False)

    environment = os.environ.copy()
    environment.update(
        {
            "DATE_MODE": "custom",
            "DATE_FROM": start_date.isoformat(),
            "DATE_TO": end_date.isoformat(),
            "OUTPUT_DIR": str(staging_directory),
            "VICCO_OUTPUT_DIR": str(staging_directory),
            "PYTHONUNBUFFERED": "1",
        }
    )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )
        return AnalysisRunResult(
            success=False,
            message="Analiz zaman aşımına uğradı.",
        )
    except Exception as exc:
        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )
        return AnalysisRunResult(
            success=False,
            message=f"Pipeline başlatılamadı: {exc}",
        )

    combined_log = "\n".join(
        part
        for part in (
            completed.stdout,
            completed.stderr,
        )
        if part
    )
    log_tail = "\n".join(
        combined_log.splitlines()[-30:]
    )

    if completed.returncode != 0:
        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )
        return AnalysisRunResult(
            success=False,
            message=(
                "Pipeline hata ile tamamlandı. "
                "Ayrıntılar için çalışma kaydını kontrol edin."
            ),
            log_tail=log_tail,
        )

    valid, validation_message = _validate_staged_outputs(
        staging_directory,
        start_date,
        end_date,
    )

    if not valid:
        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )
        return AnalysisRunResult(
            success=False,
            message=validation_message,
            log_tail=log_tail,
        )

    run_directory = _promote_outputs(
        staging_directory,
        start_date,
        end_date,
    )
    shutil.rmtree(
        staging_directory,
        ignore_errors=True,
    )

    return AnalysisRunResult(
        success=True,
        message=(
            f"{start_date:%d.%m.%Y}–{end_date:%d.%m.%Y} "
            "dönemi başarıyla analiz edildi."
        ),
        run_directory=run_directory,
        log_tail=log_tail,
    )



