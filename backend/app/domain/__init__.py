"""Domain primitives: money, case lifecycle, events, data provenance.

Nothing in here imports FastAPI, a database driver, an LLM client or a payment
provider. That is deliberate -- these are the types the rest of the system agrees on,
so they must not drag an adapter along with them.
"""
