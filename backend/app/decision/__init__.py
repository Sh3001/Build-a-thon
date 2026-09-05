"""Choosing an action by expected incremental profit.

The planner proposes candidates; this package scores them. Its output is still only a
proposal -- everything it returns goes through the policy engine before it can run.
"""
