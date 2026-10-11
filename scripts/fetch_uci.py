"""Explicitly download and validate UCI's public spreadsheet; never create a database."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import httpx

from backend.app.services.uci import _snapshot, source_path

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
       "default%20of%20credit%20card%20clients.xls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=source_path())
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() and not args.refresh:
        stat = destination.stat()
        rows, customers = _snapshot(str(destination), stat.st_mtime_ns, stat.st_size)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".xls", delete=False) as output:
                temporary = Path(output.name)
                size = 0
                with httpx.stream("GET", URL, follow_redirects=True, timeout=30) as response:
                    response.raise_for_status()
                    for block in response.iter_bytes():
                        size += len(block)
                        if size > 20_000_000:
                            raise ValueError("UCI download exceeds 20 MB")
                        output.write(block)
            stat = temporary.stat()
            rows, customers = _snapshot(str(temporary), stat.st_mtime_ns, stat.st_size)
            if customers != 30000:
                raise ValueError("Expected the full 30,000-account UCI dataset")
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"UCI: {customers:,} accounts; {len(rows):,} delinquent September snapshots. {destination}")
    print("Yeh, I. (2009), UCI ML Repository. CC BY 4.0. Historical data; no database created.")


if __name__ == "__main__":
    main()
