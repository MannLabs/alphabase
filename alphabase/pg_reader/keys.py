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
    ENSEMBL_IDS
        ENSEMBL IDs of proteins in the respective protein group.
        Individual entries are separated by a semicolon in the unified output.
    GENES
        Gene names encoding the proteins in the respective protein group. Uses HGNC names for humans.
        Individual entries are separated by a semicolon in the unified output.
    PROTEIN_CANDIDATES
        Uniprot IDs of all proteins matching a specific precursor before protein inference.
        Individual entries are separated by a semicolon in the unified output.
    DESCRIPTION
        Long text description of one or more proteins in the respective protein group.
    SOURCE_DB
        Source databases for a specific feature, e.g. Uniprot, Swissprot, Ensembl.
        Individual entries are separated by a semicolon in the unified output.
    DECOY_INDICATOR
        Boolean that indicates that the respective feature is maked as a decoy by the respective
        search engine.
    N_SEQUENCES
        Number of distinct sequences identified from the data
    PEPTIDE_COUNT
        Number of peptides mapping to the specific protein group
    PROTEOTYPIC_PEPTIDE_COUNT
        Number of proteotypic, that is gene-specific, peptides mapping to the specific protein group

    """

    # Minimal columns
    PROTEINS = "proteins"
    UNIPROT_IDS = "uniprot_ids"
    ENSEMBL_IDS = "ensembl_ids"
    GENES = "genes"
    PROTEIN_CANDIDATES = "protein_candidates"
    DESCRIPTION = "description"
    SOURCE_DB = "source_db"
    DECOY_INDICATOR = "is_decoy"
    PEPTIDE_COUNT = "peptide_count"
    PROTEOTYPIC_PEPTIDE_COUNT = "proteotypic_peptide_count"
