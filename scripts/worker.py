"""Run durable recovery jobs: python -m scripts.worker [--once].

A database session lock permits one worker per state database. A crashed worker releases
its lock automatically; its successor resumes checkpoints without redispatching actions.
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
from contextlib import ExitStack, contextmanager

from backend.app.config import DB_PATH
from backend.app.database.db import connect
from backend.app.database.jobs import JobStore, now
from backend.app.database.migrations import ensure_schema
from backend.app.services import source
from backend.app.services.recovery import Cancelled, RecoveryBlocked, RecoveryService
from backend.app.tools.executor import ActionPending


@contextmanager
def worker_lock(conn):
    stack = ExitStack()
    handle = None
    if conn.is_pg:
        locked = conn.execute("SELECT pg_try_advisory_lock(73198422) AS locked").fetchone()["locked"]
        conn.commit()
        if not locked:
            raise RuntimeError("Another recovery worker already owns this database")
    else:
        import fcntl
        handle = stack.enter_context(open(str(DB_PATH) + ".worker.lock", "a"))  # noqa: SIM115 -- ExitStack owns cleanup
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise RuntimeError("Another recovery worker already owns this database") from None
    try:
        yield
    finally:
        stack.close()
        if conn.is_pg:
            conn.execute("SELECT pg_advisory_unlock(73198422)")
            conn.commit()


def process_next(conn) -> bool:
    row = conn.execute(
        "SELECT job_id, tenant_id FROM recovery_jobs WHERE status = 'queued' "
        "ORDER BY created_at, job_id LIMIT 1").fetchone()
    if not row:
        conn.commit()
        return False
    jobs = JobStore(row["tenant_id"], conn=conn)
    jobs.finish(row["job_id"], "running")
    from backend.app.observability.logging import set_trace_id
    job = jobs.get(row["job_id"])
    set_trace_id(job.get("request_id") or job["job_id"])
    try:
        RecoveryService(jobs).run(job)
    except Cancelled:
        conn.rollback()
    except RecoveryBlocked as exc:
        conn.rollback()
        jobs.finish(row["job_id"], "blocked", error=str(exc))
    except ActionPending:
        conn.rollback()
        jobs.finish(row["job_id"], "needs_reconciliation",
                    error="An action has an uncertain outcome. Do not retry until reconciled.")
    except Exception as exc:
        conn.rollback()
        # Do not persist database URLs, source rows or provider exception messages.
        logging.getLogger(__name__).error("job_failed", extra={"job_id": row["job_id"],
                                                               "error_type": type(exc).__name__})
        jobs.finish(row["job_id"], "failed", error="Recovery failed; inspect worker logs.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if source.configured() or os.environ.get("RECOVERAI_PROFILE") == "production":
        raise SystemExit("External payments are read-only; the simulation worker is disabled.")
    conn = connect()
    try:
        ensure_schema(conn)
        with worker_lock(conn):
            conn.execute("UPDATE recovery_jobs SET status = 'queued', updated_at = ? "
                         "WHERE status = 'running'", (now(),))
            conn.commit()
            while True:
                worked = process_next(conn)
                if args.once:
                    break
                if not worked:
                    threading.Event().wait(2)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
