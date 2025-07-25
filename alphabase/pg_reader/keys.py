"""Constants for accessing the columns of a PG dataframe."""

from alphabase.psm_reader.keys import ConstantsClass


class PGCols(metaclass=ConstantsClass):
    """Standardized keys for PG readers."""

    # Minimal columns
    PROTEINS = "proteins"
