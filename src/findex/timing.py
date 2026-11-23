"""Small logging decorator preserving wrapped function metadata."""

import logging
from collections.abc import Callable
from functools import wraps
from time import perf_counter


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Log execution time at INFO level even when the call raises."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            logging.getLogger("findex.timing").info(
                "TIMED %s %.6f seconds",
                func.__qualname__,
                elapsed,
            )

    return wrapper
