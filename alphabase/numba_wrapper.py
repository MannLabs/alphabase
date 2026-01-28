"""Compatibility module for optional numba dependency.

This module provides fallback implementations when numba is not installed.
When numba is unavailable, decorated functions raise ImportError when called.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

try:
    import numba
    from numba import jit, njit, prange, vectorize

    HAS_NUMBA = True

    nb_ = numba

    numba_jit = jit
    numba_njit = njit
    numba_vectorize = vectorize
    numba_prange = prange

    NumbaTypedDict = numba.typed.Dict
    numba_types = numba.types

except ImportError:
    HAS_NUMBA = False

    _NUMBA_WARNING_SHOWN = False

    def _show_numba_warning() -> None:
        global _NUMBA_WARNING_SHOWN  # noqa: PLW0603
        if not _NUMBA_WARNING_SHOWN:
            warnings.warn(
                "numba is not installed. "
                "Install with `pip install alphabase[full]` for full functionality.",
                UserWarning,
                stacklevel=3,
            )
            _NUMBA_WARNING_SHOWN = True

    def _make_raising_decorator(decorator_name: str) -> Callable:
        """Create a decorator that raises ImportError when the decorated function is called."""
        import functools

        def decorator(
            *args: Any,  # noqa: ANN401
            **kwargs: Any,  # noqa: ANN401, ARG001
        ) -> Callable:
            _show_numba_warning()

            def make_wrapper(func: Callable) -> Callable:
                @functools.wraps(func)
                def wrapper(
                    *a: Any,  # noqa: ANN401, ARG001
                    **kw: Any,  # noqa: ANN401, ARG001
                ) -> Any:  # noqa: ANN401
                    raise ImportError(
                        f"numba is required to call '{func.__name__}'. "
                        "Install with `pip install alphabase[numba]`"
                    )

                return wrapper

            # @decorator without parentheses: args[0] is the function
            if args and callable(args[0]):
                return make_wrapper(args[0])

            # @decorator() or @decorator(option=value): return actual decorator
            return make_wrapper

        decorator.__name__ = decorator_name
        return decorator

    def _numba_vectorize_raising(
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401, ARG001
    ) -> Callable:
        """Fallback for numba.vectorize that raises when called."""
        import functools

        _show_numba_warning()

        def make_wrapper(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(
                *a: Any,  # noqa: ANN401, ARG001
                **kw: Any,  # noqa: ANN401, ARG001
            ) -> Any:  # noqa: ANN401
                raise ImportError(
                    f"numba is required to call '{func.__name__}'. "
                    "Install with `pip install alphabase[numba]`"
                )

            return wrapper

        # @numba_vectorize without parentheses: args[0] is the function
        if args and callable(args[0]):
            return make_wrapper(args[0])

        # @numba_vectorize([signatures], target=...) - return wrapper
        return make_wrapper

    class _TypedDictFallback(dict):
        """Fallback for numba.typed.Dict that uses a regular dict."""

        @staticmethod
        def empty(
            *args: Any,  # noqa: ANN401, ARG004
            **kwargs: Any,  # noqa: ANN401, ARG004
        ) -> dict:
            return {}

    class _RecursiveStubClass:
        """Fallback stub class. Returns self for any access."""

        def __getattr__(self, name: str) -> _RecursiveStubClass:
            return self

        def __getitem__(self, key: Any) -> _RecursiveStubClass:  # noqa: ANN401
            return self

        def __call__(
            self,
            *args,  # noqa: ARG002
            **kwargs,  # noqa: ARG002
        ) -> _RecursiveStubClass:
            return self

    nb_ = _RecursiveStubClass()

    numba_jit = _make_raising_decorator("jit")
    numba_njit = _make_raising_decorator("njit")
    numba_vectorize = _numba_vectorize_raising
    numba_prange = range

    NumbaTypedDict = _TypedDictFallback
    numba_types = _RecursiveStubClass()
