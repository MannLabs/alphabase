"""Compatibility module for optional numba dependency.

This module provides fallback implementations when numba is not installed.
When numba is unavailable, decorated functions will raise NotImplementedError.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

try:
    import numba
    from numba import jit, njit, prange, vectorize

    HAS_NUMBA = True

    NumbaTypedDict = numba.typed.Dict
    numba_types = numba.types

    numba_jit = jit
    numba_njit = njit
    numba_vectorize = vectorize
    numba_prange = prange
    nb_ = numba

except ImportError:
    HAS_NUMBA = False

    _NUMBA_WARNING_SHOWN = False

    def _show_numba_warning() -> None:
        global _NUMBA_WARNING_SHOWN  # noqa: PLW0603
        if not _NUMBA_WARNING_SHOWN:
            warnings.warn(
                "numba is not installed. "
                "Install with `pip install alphabase[numba]` for full functionality.",
                UserWarning,
                stacklevel=3,
            )
            _NUMBA_WARNING_SHOWN = True

    def _make_stub_decorator(decorator_name: str) -> Callable:
        """Create a stub decorator that raises NotImplementedError when called."""

        def decorator(
            *args: Any,  # noqa: ANN401
            **kwargs: Any,  # noqa: ANN401, ARG001
        ) -> Callable:
            _show_numba_warning()

            def make_wrapper(func: Callable) -> Callable:
                def wrapper(
                    *args_: Any,  # noqa: ANN401
                    **kwargs_: Any,  # noqa: ANN401
                ) -> None:
                    raise NotImplementedError(
                        f"Function '{func.__name__}' requires numba. "
                        "Install with `pip install alphabase[numba]`."
                    )

                wrapper.__name__ = func.__name__
                return wrapper

            # @decorator without parentheses: args[0] is the function
            if args and callable(args[0]):
                return make_wrapper(args[0])

            # @decorator() or @decorator(option=value): return the wrapper maker
            return make_wrapper

        decorator.__name__ = decorator_name
        return decorator

    numba_njit = _make_stub_decorator("njit")
    numba_jit = _make_stub_decorator("jit")
    numba_vectorize = _make_stub_decorator("vectorize")
    numba_prange = range

    class _TypedDictFallback(dict):
        """Fallback for numba.typed.Dict that uses a regular dict."""

        @staticmethod
        def empty(
            *args: Any,  # noqa: ANN401, ARG004
            **kwargs: Any,  # noqa: ANN401, ARG004
        ) -> dict:
            return {}

    NumbaTypedDict = _TypedDictFallback

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
    numba_types = _RecursiveStubClass()
