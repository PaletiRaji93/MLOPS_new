"""
populate_raw_data.py
====================
Populate data/raw/ from the bundled March 2025 dataset
(inputdata/march_2025_data.zip), so students can start at validation without
running live ingestion.

It reads the provided March 2025 data and writes it into data/raw in the same
layout the ingestion scripts produce:

  dam / rtm  ->  data/raw/<ds>/year=YYYY/month=MM/date=YYYY-MM-DD/<ds>.csv
  weather    ->  data/raw/weather/<City_State>.csv
  calendar   ->  data/raw/calendar/calendar.csv

dam and rtm are partitioned by their Datetime column into year/month/date
folders. Existing partitions for the same dates are overwritten.

Usage
-----
  python scripts/populate_raw_data.py
  python scripts/populate_raw_data.py --zip inputdata/march_2025_data.zip
"""

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEFAULT_ZIP = ROOT / "inputdata" / "march_2025_data.zip"

# datasets that get split into year=/month=/date= partitions, keyed by datetime column
PARTITIONED = {"dam": "Datetime", "rtm": "Datetime"}


def _write_partitions(name: str, df: pd.DataFrame, dt_col: str) -> int:
    """Split df by calendar date and write one CSV per date partition."""
    dt = pd.to_datetime(df[dt_col])
    days = 0
    for day, chunk in df.groupby(dt.dt.strftime("%Y-%m-%d")):
        d = pd.Timestamp(day)
        out_dir = RAW / name / f"year={d.year}" / f"month={d.month:02d}" / f"date={day}"
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk.to_csv(out_dir / f"{name}.csv", index=False)
        days += 1
    return days


def main() -> None:
    ap = argparse.ArgumentParser(description="Populate data/raw from the March 2025 bundle")
    ap.add_argument("--zip", default=str(DEFAULT_ZIP), help="path to march_2025_data.zip")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Bundle not found: {zip_path}")

    print(f"Reading {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        # dam / rtm -> year/month/date partitions
        for name, dt_col in PARTITIONED.items():
            member = f"{name}.csv"
            if member not in names:
                print(f"  {name}: not in bundle, skipped")
                continue
            df = pd.read_csv(io.BytesIO(z.read(member)))
            days = _write_partitions(name, df, dt_col)
            print(f"  {name}: {len(df)} rows -> {days} date partitions under data/raw/{name}/")

        # weather -> per-city files
        weather_files = [n for n in names if n.startswith("weather/") and n.endswith(".csv")]
        if weather_files:
            (RAW / "weather").mkdir(parents=True, exist_ok=True)
            for n in weather_files:
                (RAW / "weather" / Path(n).name).write_bytes(z.read(n))
            print(f"  weather: {len(weather_files)} city files -> data/raw/weather/")

        # calendar -> single reference file
        if "calendar.csv" in names:
            (RAW / "calendar").mkdir(parents=True, exist_ok=True)
            (RAW / "calendar" / "calendar.csv").write_bytes(z.read("calendar.csv"))
            print("  calendar: -> data/raw/calendar/calendar.csv")

    print("Done. data/raw/ is populated with the March 2025 data.")


if __name__ == "__main__":
    main()
