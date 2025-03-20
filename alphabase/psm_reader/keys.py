"""Constants for accessing the columns of a PSM dataframe."""

from typing import Any, List, NoReturn


class ConstantsClass(type):
    """A metaclass for classes that should only contain string constants."""

    def __setattr__(cls, name: Any, value: Any) -> NoReturn:  # noqa: ANN401
        """Raise an error when trying to set an attribute."""
        raise TypeError("Constants class cannot be modified")

    def get_values(cls) -> List[str]:
        """Get all user-defined string values of the class."""
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]


class PsmDfCols(metaclass=ConstantsClass):
    """Constants for accessing the columns of a PSM dataframe."""

    # TODO: these are used only in th psm_reader package and the spectral_library.reader module so far
    MOD_SITES = "mod_sites"
    MODIFIED_SEQUENCE = "modified_sequence"
    SEQUENCE = "sequence"
    DECOY = "decoy"
    MODS = "mods"
    SCORE = "score"
    TO_REMOVE = "to_remove"
    AA_MASS_DIFFS = "aa_mass_diffs"
    AA_MASS_DIFF_SITES = "aa_mass_diff_sites"
    RT = "rt"
    RT_START = "rt_start"
    RT_STOP = "rt_stop"
    RT_NORM = "rt_norm"
    SPEC_IDX = "spec_idx"
    SCANNR = "scannr"
    FDR = "fdr"
    NAA = "nAA"
    CCS = "ccs"
    MOBILITY = "mobility"
    PEPTIDE_FDR = "peptide_fdr"
    PROTEIN_FDR = "protein_fdr"

    RAW_NAME = "raw_name"
    CHARGE = "charge"
    PROTEINS = "proteins"
    INTENSITY = "intensity"

    SCAN_NUM = "scan_num"
    PRECURSOR_MZ = "precursor_mz"
    DIANN_SPEC_INDEX = "diann_spec_idx"

    # part of the output, but not directly referenced in code
    UNIPROT_IDS = "uniprot_ids"
    GENES = "genes"
    QUERY_ID = "query_id"

    # extra FDR columns for DIANN
    FDR1_SEARCH1 = "fdr1_search1"
    FDR2_SEARCH1 = "fdr2_search1"
    FDR1_SEARCH2 = "fdr1_search2"
    FDR2_SEARCH2 = "fdr2_search2"


class LibPsmDfCols(metaclass=ConstantsClass):
    """Constants for accessing the columns of a Library PSM dataframe."""

    FRAG_START_IDX = "frag_start_idx"
    FRAG_STOP_IDX = "frag_stop_idx"

    # not referenced in reader classes
    FRAGMENT_INTENSITY = "fragment_intensity"
    FRAGMENT_MZ = "fragment_mz"
    FRAGMENT_TYPE = "fragment_type"
    FRAGMENT_CHARGE = "fragment_charge"
    FRAGMENT_SERIES = "fragment_series"
    FRAGMENT_LOSS_TYPE = "fragment_loss_type"
