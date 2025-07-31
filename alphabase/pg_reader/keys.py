"""Constants for accessing the columns of a PG dataframe."""

from alphabase.psm_reader.keys import ConstantsClass


class PGCols(metaclass=ConstantsClass):
    """Standardized keys for PG readers.

    Attributes
    ----------
    PROTEINS
        Uniprot names of proteins in the respective protein group.
        Individual entries are separated by a semicolon in the unified output.
    UNIPROT_IDS
        Uniprot IDs of all proteins in the respective protein group.
        Individual entries are separated by a semicolon in the unified output.

    """

    # Minimal columns
    PROTEINS = "proteins"
    UNIPROT_IDS = "uniprot_ids"
