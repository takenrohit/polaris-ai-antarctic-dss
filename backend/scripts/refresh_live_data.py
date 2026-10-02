"""
Fetch the latest public data and rebuild the live metocean store.

    python backend/scripts/refresh_live_data.py                # rebuild now
    python backend/scripts/refresh_live_data.py --dry-run      # only report newest NSIDC day
    python backend/scripts/refresh_live_data.py --days 21 --max-back 14

Sources (keyless): NOAA@NSIDC G02135 v4.0 daily GeoTIFFs and the Open-Meteo forecast API.
Exit codes: 0 ok, 2 data unavailable / network failure (previous store left untouched).

Schedule it (the freshness fail-safe trips after 48 h without a successful refresh):
    cron:   17 */6 * * *  cd /path/to/repo && python backend/scripts/refresh_live_data.py
If the API server is already running, call POST /api/forecast/refresh afterwards (or instead)
so it reloads the new store without a restart.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.data.live_fetch import (  # noqa: E402
    DataUnavailable, LiveFetchError, build_live_store, find_latest_available, http_get)
from app.data.ingestion import LIVE_CACHE_DIR, LIVE_NC_PATH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=21, help="observed days in the rolling window (>=14)")
    ap.add_argument("--max-back", type=int, default=14, help="days to search back for the newest NSIDC file")
    ap.add_argument("--out", type=Path, default=LIVE_NC_PATH)
    ap.add_argument("--cache", type=Path, default=LIVE_CACHE_DIR)
    ap.add_argument("--dry-run", action="store_true", help="only check what NSIDC has published")
    args = ap.parse_args()

    try:
        if args.dry_run:
            today = datetime.now(timezone.utc).date()
            latest, _ = find_latest_available(http_get, today, args.max_back)
            print(f"Newest NSIDC G02135 daily file: {latest.isoformat()} ({(today - latest).days} day(s) old)")
            return 0
        summary = build_live_store(args.out, args.cache, n_obs_days=args.days, max_back=args.max_back)
    except (DataUnavailable, LiveFetchError) as exc:
        print(f"REFRESH FAILED (previous store untouched): {exc}", file=sys.stderr)
        return 2

    print("Live store rebuilt:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
