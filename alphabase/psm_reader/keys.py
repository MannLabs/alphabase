class ConstantsClass(type):
    """A metaclass for classes that should only contain string constants."""

    def __setattr__(self, name, value):
        raise TypeError("Constants class cannot be modified")

    def get_values(cls):
        """Get all user-defined string values of the class."""
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]


class PsmDfCols(metaclass=ConstantsClass):
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

    SCAN_NUM = "scan_num"
    PRECURSOR_MZ = "precursor_mz"
    DIANN_SPEC_INDEX = "diann_spec_idx"

    FRAG_START_IDX = "frag_start_idx"
    FRAG_STOP_IDX = "frag_stop_idx"

    # part of the output, but not directly referenced
    _UNIPROT_IDS = "uniprot_ids"
    _GENES = "genes"
    _QUERY_ID = "query_id"
