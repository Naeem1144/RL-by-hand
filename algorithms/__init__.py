"""Multi-armed bandit algorithms implemented from scratch."""

__all__ = ["run_bandit", "run_thompson", "run_ucb"]


def __getattr__(name: str):
    """Lazily expose the run functions so ``from algorithms import run_ucb``
    works without eagerly importing submodules (which would trip up
    ``python -m algorithms.<name>``)."""
    if name == "run_bandit":
        from .epsilon_greedy import run_bandit

        return run_bandit
    if name == "run_thompson":
        from .thompson import run_thompson

        return run_thompson
    if name == "run_ucb":
        from .ucb import run_ucb

        return run_ucb
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
