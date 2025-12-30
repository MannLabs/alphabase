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
    """Constants for accessing the columns of a PSM dataframe.

    Attributes
    ----------
    MOD_SITES, MODIFIED_SEQUENCE
        Peptide sequence annotated with modification locations and standardized modification labels.
    SEQUENCE
        Peptide amino-acid sequence without modification annotations (plain sequence of residues).
    DECOY
        Label indicating whether the matched sequence is a decoy (reverse/shuffled) entry used for FDR estimation.
    TMP_MODS
        Intermediate column containing raw modification strings from search engines (e.g., MSFragger's
        "Assigned Modifications" format: "5S(79.9663), N-term(304.2071)"). This is translated into
        the standardized MODS and MOD_SITES columns during processing and is not present in the final output.
    MODS
        List of modifications on the peptide in unimod standardization
    SCORE
        Primary identification score for the PSM produced by the search engine quantifying quality of the PSM in arbitrary units.
    TO_REMOVE
    AA_MASS_DIFFS
    AA_MASS_DIFF_SITES
    RT
        Retention time of the precursor (time when the precursor eluted during chromatography) in [UNIT].
    RT_START
        Start of search window in retention time of the precursor (time when the precursor eluted during chromatography) in [UNIT].
    RT_STOP
        End of search window in retention time of the precursor (time when the precursor eluted during chromatography) in [UNIT].
    RT_NORM
        Normalized retention time of the precursor (time when the precursor eluted during chromatography) in [UNIT].
    SPEC_IDX
        Spectrum index or internal spectrum identifier used by some file formats/engines to reference a spectrum; may be distinct from scan number.
    SCANNR
        MS/MS scan number (integer) indexing the spectrum within the raw file; use to locate the spectrum in vendor files or mzML.
    FDR
        False discovery rate (FDR) or q-value associated with the PSM.
    NAA
        Number of amino acids in sequence
    CCS
        Collision Cross Section (CCS) value for the ion in [UNIT].
    MOBILITY
        Ion mobility value associated with the precursor when available in [UNIT].
    PEPTIDE_FDR
        FDR or q-value estimated at the peptide-sequence level.
    PROTEIN_FDR
        FDR or q-value estimated at the protein or protein-group level.
    RAW_NAME
        Original filename or run identifier as reported by the acquisition system or search engine.
        Represents the raw data source that produced the spectrum.
    CHARGE
        Charge state of precursor.
    PROTEINS
        Protein names (human readable format) of the matching protein group.
    INTENSITY
        Intensity of the corresponding protein group.
    SCAN_NUM
        MS/MS scan number (integer) indexing the na within the raw file.
    PRECURSOR_MZ
        Measured precursor mass-to-charge ratio (m/z) for the MS/MS spectrum.
    DIANN_SPEC_INDEX
        Precursor.Lib.Index index of the precursor in the internal representation used by DIA-NN for the spectral library.
    UNIPROT_IDS
        Uniprot identifiers of the corresponding protein group.
    GENES
        Gene names (typically in HGNC nomenclature) of the corresponding protein group.
    QUERY_ID
    FDR1_SEARCH1
        Empirical FDR cutoff for first-pass Global precursor Q-Value.
    FDR2_SEARCH1
        Empirical FDR cutoff for first-pass Global protein group Q-Value.
    FDR1_SEARCH2
        Empirical FDR cutoff for second pass spectral library precursor Q-Value.
    FDR2_SEARCH2
    PRECURSOR_ID
        Unique identifier for each precursor, i.e. a physical ionized peptide species.
    PRECURSOR_INTENSITY
        Computed intensity value for a specific precursor, i.e. a physical ionized peptide species.
    GENE_INTENSITY
        Computed intensity value on a gene level.
    PEPTIDE_INTENSITY
        Computed intensity value on a peptide/sequence level.
        Empirical FDR cutoff for second pass library protein group Q-Value.

    """

    # TODO: these are used only in th psm_reader package and the spectral_library.reader module so far
    MOD_SITES = "mod_sites"
    MODIFIED_SEQUENCE = "modified_sequence"
    SEQUENCE = "sequence"
    DECOY = "decoy"
    TMP_MODS = "_tmp_mods"
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

    PRECURSOR_ID = "precursor_id"
    PRECURSOR_INTENSITY = "precursor_intensity"

    GENE_INTENSITY = "gene_intensity"

    PEPTIDE_INTENSITY = "peptide_intensity"


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


class MsFraggerTokens(metaclass=ConstantsClass):
    """String tokens used in MSFragger output formats."""

    MOD_START = "("
    MOD_STOP = ")"
    MOD_SEPARATOR = ","
    N_TERM = "N-term"
    C_TERM = "C-term"
    DECOY_PREFIX = "rev_"
