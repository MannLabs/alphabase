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

    SCAN_NUM = "scan_num"
    PRECURSOR_MZ = "precursor_mz"
    DIANN_SPEC_INDEX = "diann_spec_idx"

    # part of the output, but not directly referenced
    _UNIPROT_IDS = "uniprot_ids"
    _GENES = "genes"
    _QUERY_ID = "query_id"

    # part of psm_reader_yaml, but not directly referenced
    _INTENSITY = "intensity"


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
