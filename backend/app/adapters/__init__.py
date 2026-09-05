"""Adapters to the outside world: payment rails, processor code vocabularies.

Everything here implements an interface the business layer defines. The direction of the
dependency is the point -- `backend.app.tools.executor` must not know that Stripe exists.
"""
