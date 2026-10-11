"""Explicit schema provisioning in an EXISTING recovery database; never creates a database."""
import os

from backend.app.database.db import connect
from backend.app.database.migrations import migrate


def main():
    url = os.environ.get("RECOVERAI_DB_URL")
    if not url:
        raise SystemExit("Set RECOVERAI_DB_URL to an existing recovery-state database first.")
    if url == os.environ.get("RECOVERAI_SOURCE_URL"):
        raise SystemExit("Use a dedicated recovery database/role, separate from the read-only source.")
    conn = connect(url=url)
    try:
        print("Applied migrations:", ", ".join(migrate(conn)) or "none")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
