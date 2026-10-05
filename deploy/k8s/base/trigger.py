"""Trigger one gridcast job through the API; the API process does the work.

Run by the CronJob pods: `python /trigger/trigger.py ingest_score`. The trigger never
touches the database itself, so SQLite keeps exactly one writer (the API pod).

Exit status decides what Kubernetes does next:
- 0 on HTTP 200 (the job ran) and on 409 (the same job is already running: nothing to do);
- 1 on anything else (the job failed, or the API was unreachable), so the Job retries
  with exponential backoff up to its backoffLimit.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: trigger.py <job name>", file=sys.stderr)
        return 2
    name = argv[1]
    base = os.environ.get("API_BASE", "http://gridcast-api:8000").rstrip("/")
    timeout = float(os.environ.get("TRIGGER_TIMEOUT_SECONDS", "2300"))
    request = urllib.request.Request(
        f"{base}/admin/jobs/{name}/run",
        method="POST",
        data=b"",
        headers={"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "")},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            print(f"{name}: {response.status} {response.read().decode()[:4000]}")
            return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:4000]
        if exc.code == 409:
            print(f"{name}: already running, nothing started ({body})")
            return 0
        print(f"{name}: HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"{name}: API unreachable at {base}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
