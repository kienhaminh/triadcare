"""Manuscript measurement pipeline: run frozen scenarios, then score saved artifacts.

The pipeline has three steps. `run` executes the frozen tasks of one scenario with one
model and saves artifacts. `score` compares those artifacts with the frozen reference.
`report` sums the scores into the manuscript tables. All denominators come from the
frozen release under eval/reference_units; nothing is hard-coded. `checks` is an optional
judge-free regression check of saved runs.
"""
