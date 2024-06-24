import warnings

from alphabase.constants.atom import *  # noqa: F403 TODO remove in the next release

warnings.warn(
    "The module `alphabase.constants.element` is deprecated, "
    "it will be removed in alphabase>=1.3.0. "
    "Please use `alphabase.constants.atom` instead",
    FutureWarning,
)
