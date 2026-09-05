"""HTTP routers, split by concern so `main.py` stays readable.

Every router takes its principal from `deps.require`, and every store it opens is scoped
to that principal's tenant. There is no route that reads a tenant from the request.
"""
