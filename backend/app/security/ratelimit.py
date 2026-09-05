"""Per-caller rate limiting.

A token bucket, in process. That is honest about its scope: with several API processes
each keeps its own bucket, so the effective limit is per-process. Stating that is better
than implying a distributed guarantee this does not provide -- a shared limiter needs
Redis, and adding Redis to make one number more accurate is not a trade worth making at
this size.

The bucket is keyed on the *principal* where one exists and the client address otherwise,
because limiting by IP alone punishes everyone behind one NAT for one bad actor, and
limiting by principal alone leaves the unauthenticated endpoints unprotected.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Bucket:
    tokens: float
    updated_at: float


@dataclass
class RateLimiter:
    """`capacity` requests in a burst, refilled at `refill_per_second`."""
    capacity: float = 120.0
    refill_per_second: float = 2.0
    #: Bound on tracked keys, evicted least-recently-used. An unbounded dict keyed on
    #: attacker-controlled input is a memory-exhaustion bug.
    max_keys: int = 20_000
    _buckets: OrderedDict[str, Bucket] = field(default_factory=OrderedDict, repr=False)
    allowed: int = 0
    blocked: int = 0

    def check(self, key: str, cost: float = 1.0, now: float | None = None) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        t = now if now is not None else time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = Bucket(tokens=self.capacity, updated_at=t)
            self._buckets[key] = bucket
        else:
            self._buckets.move_to_end(key)
            elapsed = max(0.0, t - bucket.updated_at)
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
            bucket.updated_at = t

        while len(self._buckets) > self.max_keys:
            self._buckets.popitem(last=False)

        if bucket.tokens >= cost:
            bucket.tokens -= cost
            self.allowed += 1
            return True, 0.0
        self.blocked += 1
        deficit = cost - bucket.tokens
        if self.refill_per_second <= 0:
            # A bucket that never refills. Legitimate as a hard quota, but there is no
            # finite retry-after to report, and dividing by zero to find one turns a
            # rate-limit response into a 500 -- which is a denial of service caused by
            # the denial-of-service defence.
            return False, float("inf")
        return False, round(deficit / self.refill_per_second, 2)

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._buckets.clear()
        else:
            self._buckets.pop(key, None)

    def stats(self) -> dict:
        return {"tracked_keys": len(self._buckets), "allowed": self.allowed,
                "blocked": self.blocked, "capacity": self.capacity,
                "refill_per_second": self.refill_per_second,
                "scope": "per-process (not shared across replicas)"}
