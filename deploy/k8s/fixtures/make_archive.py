"""Serve a synthetic NYISO archive inside the kind cluster (CI and laptop).

The API reads it through NYISO_ARCHIVE_BASE=http://gridcast-archive-fixture:8080, so the
Kubernetes test path needs no external network and never loads NYISO's servers. The
archive is written at container start, not at image build time, so its dates are relative
to the day the pod starts: daily files from the 1st of the month two months back through
yesterday, plus a zip for every complete month (the client reads closed months only from
zips), mirroring the real archive's layout. Then it is served on port 8080.
"""

from __future__ import annotations

import functools
import http.server
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.synthetic import SyntheticNYISO  # noqa: E402

PORT = 8080


def main() -> None:
    root = Path(os.environ.get("ARCHIVE_ROOT", "/archive"))
    today = datetime.now(ZoneInfo("America/New_York")).date()
    first = (today - timedelta(days=62)).replace(day=1)
    last = today - timedelta(days=1)
    t0 = time.perf_counter()
    files = SyntheticNYISO(first, last, seed=21).write_archive(root, monthly_zips=True)
    print(
        f"synthetic archive {first}..{last}: {len(files)} files in "
        f"{time.perf_counter() - t0:.1f} s under {root}",
        flush=True,
    )
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    print(f"serving on :{PORT}", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), handler).serve_forever()


if __name__ == "__main__":
    main()
