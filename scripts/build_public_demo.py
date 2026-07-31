from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "outputs"
DEMO_DIR = PROJECT_ROOT / "demo_data"


CAMPAIGN_PATTERNS = (
    r"\bcc-gg-[^\s,;]+",
    r"\bcz-gg-[^\s,;]+",
)

CAMPAIGN_COLUMN_KEYWORDS = {
    "campaign",
    "campaignname",
    "campaign_name",
    "campaign name",
    "kampanya",
    "kampanyaadı",
    "kampanya_adi",
    "kampanya adı",
}

CATEGORY_COLUMN_KEYWORDS = {
    "category",
    "categoryname",
    "category_name",
    "kategori",
    "kategoriadı",
    "kategori_adi",
    "kategori adı",
}

CHANNEL_COLUMN_KEYWORDS = {
    "channel",
    "channelname",
    "channel_name",
    "kanal",
    "kanaladı",
    "kanal_adi",
    "kanal adı",
}

PRODUCT_COLUMN_KEYWORDS = {
    "product",
    "productname",
    "product_name",
    "ürün",
    "urun",
    "ürünadı",
    "urunadi",
    "ürün_adi",
    "urun_adi",
    "ürün adı",
    "urun adı",
}

BRAND_COLUMN_KEYWORDS = {
    "brand",
    "brandname",
    "brand_name",
    "marka",
    "markaadı",
    "marka_adi",
    "marka adı",
}


campaign_map: dict[str, str] = {}
category_map: dict[str, str] = {}
channel_map: dict[str, str] = {}
product_map: dict[str, str] = {}
brand_map: dict[str, str] = {}


def normalize_column_name(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
    )


def get_demo_value(
    value: Any,
    mapping: dict[str, str],
    prefix: str,
) -> Any:
    if pd.isna(value):
        return value

    text = str(value).strip()

    if not text:
        return value

    if text not in mapping:
        mapping[text] = f"{prefix} {len(mapping) + 1:03d}"

    return mapping[text]


def contains_real_campaign_name(value: Any) -> bool:
    if pd.isna(value):
        return False

    text = str(value)

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in CAMPAIGN_PATTERNS
    )


def replace_campaign_text(value: Any) -> Any:
    if pd.isna(value):
        return value

    text = str(value)

    if not contains_real_campaign_name(text):
        return value

    if text not in campaign_map:
        campaign_map[text] = (
            f"Demo Campaign {len(campaign_map) + 1:03d}"
        )

    return campaign_map[text]


def anonymize_series(
    series: pd.Series,
    mapping: dict[str, str],
    prefix: str,
) -> pd.Series:
    return series.map(
        lambda value: get_demo_value(
            value,
            mapping,
            prefix,
        )
    )


def anonymize_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    for column in result.columns:
        normalized = normalize_column_name(column)

        if normalized in CAMPAIGN_COLUMN_KEYWORDS:
            result[column] = anonymize_series(
                result[column],
                campaign_map,
                "Demo Campaign",
            )
            continue

        if normalized in CATEGORY_COLUMN_KEYWORDS:
            result[column] = anonymize_series(
                result[column],
                category_map,
                "Demo Category",
            )
            continue

        if normalized in CHANNEL_COLUMN_KEYWORDS:
            result[column] = anonymize_series(
                result[column],
                channel_map,
                "Demo Channel",
            )
            continue

        if normalized in PRODUCT_COLUMN_KEYWORDS:
            result[column] = anonymize_series(
                result[column],
                product_map,
                "Demo Product",
            )
            continue

        if normalized in BRAND_COLUMN_KEYWORDS:
            result[column] = anonymize_series(
                result[column],
                brand_map,
                "Demo Brand",
            )
            continue

        if (
            pd.api.types.is_object_dtype(result[column])
            or pd.api.types.is_string_dtype(result[column])
        ):
            result[column] = result[column].map(
                replace_campaign_text
            )

    return result


def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp1254",
        "latin-1",
    )

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"CSV okunamadı: {path}. Son hata: {last_error}"
    )


def process_csv(path: Path) -> None:
    dataframe = read_csv_safely(path)
    anonymized = anonymize_dataframe(dataframe)

    anonymized.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def anonymize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: anonymize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            anonymize_json_value(item)
            for item in value
        ]

    if isinstance(value, str):
        return replace_campaign_text(value)

    return value


def process_json(path: Path) -> None:
    try:
        content = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return

    anonymized = anonymize_json_value(content)

    path.write_text(
        json.dumps(
            anonymized,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def process_text_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="cp1254")
        except Exception:
            return
    except Exception:
        return

    updated = text

    for pattern in CAMPAIGN_PATTERNS:
        matches = re.findall(
            pattern,
            updated,
            flags=re.IGNORECASE,
        )

        for match in matches:
            replacement = get_demo_value(
                match,
                campaign_map,
                "Demo Campaign",
            )
            updated = updated.replace(
                match,
                replacement,
            )

    if updated != text:
        path.write_text(
            updated,
            encoding="utf-8",
        )


def verify_demo_directory() -> list[Path]:
    unsafe_files: list[Path] = []

    for path in DEMO_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".csv",
            ".json",
            ".txt",
            ".log",
        }:
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        if re.search(
            r"\b(?:cc|cz)-gg-",
            text,
            flags=re.IGNORECASE,
        ):
            unsafe_files.append(path)

    return unsafe_files


def main() -> None:
    if not SOURCE_DIR.exists():
        raise SystemExit(
            f"Kaynak outputs klasörü bulunamadı: {SOURCE_DIR}"
        )

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)

    shutil.copytree(
        SOURCE_DIR,
        DEMO_DIR,
    )

    csv_files = sorted(
        DEMO_DIR.rglob("*.csv")
    )

    json_files = sorted(
        DEMO_DIR.rglob("*.json")
    )

    text_files = sorted(
        [
            *DEMO_DIR.rglob("*.txt"),
            *DEMO_DIR.rglob("*.log"),
        ]
    )

    for index, path in enumerate(
        csv_files,
        start=1,
    ):
        process_csv(path)
        print(
            f"CSV {index:03d}/{len(csv_files):03d}: "
            f"{path.relative_to(DEMO_DIR)}"
        )

    for index, path in enumerate(
        json_files,
        start=1,
    ):
        process_json(path)
        print(
            f"JSON {index:03d}/{len(json_files):03d}: "
            f"{path.relative_to(DEMO_DIR)}"
        )

    for index, path in enumerate(
        text_files,
        start=1,
    ):
        process_text_file(path)
        print(
            f"TEXT {index:03d}/{len(text_files):03d}: "
            f"{path.relative_to(DEMO_DIR)}"
        )

    unsafe_files = verify_demo_directory()

    print()
    print("=" * 60)
    print("PUBLIC DEMO BUILD COMPLETE")
    print(f"Kaynak klasör : {SOURCE_DIR}")
    print(f"Demo klasörü  : {DEMO_DIR}")
    print(f"CSV dosyası   : {len(csv_files)}")
    print(f"JSON dosyası  : {len(json_files)}")
    print(f"Metin dosyası : {len(text_files)}")
    print(f"Kampanya      : {len(campaign_map)}")
    print(f"Kategori      : {len(category_map)}")
    print(f"Kanal         : {len(channel_map)}")
    print(f"Ürün          : {len(product_map)}")
    print(f"Marka         : {len(brand_map)}")
    print("=" * 60)

    if unsafe_files:
        print()
        print("UYARI: Gerçek kampanya kalıbı bulunan dosyalar:")

        for path in unsafe_files:
            print(
                f"- {path.relative_to(DEMO_DIR)}"
            )

        raise SystemExit(
            "Demo oluşturuldu ancak doğrulama başarısız oldu."
        )

    print()
    print("Doğrulama başarılı.")
    print("Gerçek outputs klasörüne dokunulmadı.")
    print("Demo verileri tamamen demo_data içine üretildi.")


if __name__ == "__main__":
    main()