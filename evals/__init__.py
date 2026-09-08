"""CareDesk's evaluation dataset: hand-authored test cases, not app code.

Kept as a top-level package alongside src/, not inside src/caredesk/, so it
never ships as part of the installed application package -- eval content
and the code that serves requests are different concerns with different
lifecycles.
"""
