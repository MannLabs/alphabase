"""Compatibility module for optional numba dependency.

This module provides fallback implementations when numba is not installed.
When numba is unavailable, decorators become no-ops (identity functions).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

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

    def _make_identity_decorator(decorator_name: str) -> Callable:
        """Create a stub decorator that returns the function as-is."""

        def decorator(
            *args: Any,  # noqa: ANN401
            **kwargs: Any,  # noqa: ANN401, ARG001
        ) -> Callable:
            _show_numba_warning()

            # @decorator without parentheses: args[0] is the function
            if args and callable(args[0]):
                return args[0]

            # @decorator() or @decorator(option=value): return identity function
            def identity(func: Callable) -> Callable:
                return func

            return identity

        decorator.__name__ = decorator_name
        return decorator

    def _numba_vectorize(
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401, ARG001
    ) -> Callable:
        """Fallback for numba.vectorize using numpy.vectorize."""
        _show_numba_warning()

        def wrap_with_numpy_vectorize(func: Callable) -> Callable:
            return np.vectorize(func)

        # @numba_vectorize without parentheses: args[0] is the function
        if args and callable(args[0]):
            return np.vectorize(args[0])

        # @numba_vectorize([signatures], target=...) - ignore signatures, return wrapper
        return wrap_with_numpy_vectorize

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

    numba_jit = _make_identity_decorator("jit")
    numba_njit = _make_identity_decorator("njit")
    numba_vectorize = _numba_vectorize
    numba_prange = range

    NumbaTypedDict = _TypedDictFallback
    numba_types = _RecursiveStubClass()
