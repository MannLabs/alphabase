"""Reader for spectral libraries exported by PEAKS Studio."""

import copy
import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alphabase.constants.aa import aa_formula
from alphabase.peptide.fragment import LOSS_MAPPING, SERIES_MAPPING
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.modification_mapper import ModificationMapper
from alphabase.psm_reader.psm_reader import psm_reader_yaml
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.utils import _get_delimiter

logger = logging.getLogger(__name__)

# The set of single-letter codes AlphaBase itself recognizes (standard 20 AAs
# plus extended codes it defines on purpose, e.g. U=selenocysteine,
# O=pyrrolysine, X/B/J/Z=ambiguous/unknown placeholders with a disabling mass
# - see alphabase/constants/const_files/amino_acid.tsv). Anything *outside*
# this set (lowercase, digits, punctuation) cannot come from a real PEAKS
# export and is rejected rather than silently passed through.
_VALID_SEQUENCE_CHARS = frozenset(aa_formula.index)
_REQUIRED_COLUMNS = (PsmDfCols.SEQUENCE, PsmDfCols.CHARGE, PsmDfCols.PRECURSOR_MZ)

# Column mapping and modification mapping live in psm_reader.yaml under the
# "peaks" key, like every other reader (see `alphabase/constants/const_files/
# psm_reader.yaml`), not as module-level constants here. The modification
# mapping there currently covers the *only* 3 (name, mass) combinations
# observed across the MPIB example library (52,672 rows, 8,094 modification
# occurrences) - cross-checked against modification.tsv:
#   Carboxymethyl@C          58.005479  (NOT Carbamidomethyl@C, 57.021464 - different mod!)
#   Oxidation@M               15.994915
#   Acetyl@Protein_N-term     42.010565
# Very likely incomplete for PEAKS libraries in general - extend the yaml
# entry as new modifications are encountered (an unmapped modification warns
# and drops the affected precursor rather than crashing - see
# `_harmonize_modifications`).

# Matches one modification token in PEAKS' "Modifications" column, e.g.
#   "10-Carboxymethyl-(58.01)" -> position="10", name="Carboxymethyl", mass="58.01"
#   "0-Acetylation (Protein N-term)-(42.01)" -> position="0", name="Acetylation (Protein N-term)"
# Multiple modifications are concatenated with ';' in the source column, but
# this pattern is applied with `finditer` (not `split`), so it finds every
# token regardless of what separator (if any) sits between them - verified
# against all 8,094 modification occurrences in the example file, 0 mismatches.
_PEAKS_MOD_TOKEN_RE = re.compile(
    r"(?P<position>\d+)-(?P<name>.+?)-\((?P<mass>[\d.]+)\)"
)

# Matches one fragment token in PEAKS' "Peaks List" column, e.g.
#   "214.11861:0.5882:b6[2+]" -> mz, intensity, ion_type='b', ion_number=6, charge=2
#   "218.11353:0.5980:y2"                        -> ... charge defaults to 1
#   "775.32904:0.0500:y7-H2O" / "716.28815:0.0500:b8-NH3" -> neutral losses
# Verified against all 897,272 fragment tokens in the example file: only 'b'
# and 'y' ion types occur, and only H2O/NH3 neutral losses - 0 unparseable tokens.
_PEAKS_FRAGMENT_TOKEN_RE = re.compile(
    r"^(?P<mz>[\d.]+):(?P<intensity>[\d.]+):"
    r"(?P<ion_type>[by])(?P<ion_number>\d+)"
    r"(?P<neutral_loss>-(?:H2O|NH3))?"
    r"(?:\[(?P<frag_charge>\d+)\+\])?$"
)

_PROTEIN_N_TERM_SITE = "0"
_C_TERM_SITE = "-1"


class PEAKSLibraryReader:
    """Reader for spectral libraries exported by PEAKS Studio (DB search / DIA-DB export).

    Converts a PEAKS library TSV (one row per precursor, with all fragment
    ions packed into a single "Peaks List" string column) into an AlphaBase
    :class:`~alphabase.spectral_library.flat.SpecLibFlat`.

    Design notes / precedent in this codebase
    -------------------------------------------
    - `alphabase.spectral_library.reader.LibraryReaderBase` is the existing
      spectral library reader, but it expects "long format" input (one row
      per *fragment*, like Spectronaut exports). PEAKS' format is the
      opposite shape (one row per *precursor*), so this class does not
      subclass it - it builds `precursor_df`/`fragment_df` directly instead.
    - `alphabase.psm_reader.sage_reader.SageReaderBase` is the closest
      precedent for the modification-harmonization step: Sage also reports
      modifications as (position, observed mass) rather than an embedded
      modified-sequence string.
    - PEAKS fragments already carry *observed* m/z and intensity (this is a
      results export, not an in-silico library), so unlike
      `LibraryReaderBase` this reader never calls `calc_fragment_mz_df()` -
      there is nothing to calculate.
    - Column mapping and modification mapping are still sourced from
      `psm_reader.yaml` / `ModificationMapper`, exactly like every reader in
      `alphabase.psm_reader` does, even though this class doesn't subclass
      `PSMReaderBase` (constructed directly in `__init__` instead of
      inherited, since the base class's `import_file` template method
      assumes the wrong input shape - see above).

    Example:
    -------
    >>> reader = PEAKSLibraryReader()
    >>> speclib = reader.import_file("lib.tsv")
    >>> speclib.precursor_df.head()

    """

    #: Key into `psm_reader.yaml`, same convention as `PSMReaderBase._reader_type`.
    _reader_type = "peaks"
    #: Also add every AlphaBase modification's generic UniMod-ID alias to the
    #: mapping (harmless no-op for PEAKS' flat naming style - see
    #: `alphabase.psm_reader.modification_mapper.ModificationMapper._extend_mod_brackets`)
    #: matching `LibraryReaderBase`'s choice for the same reason.
    _add_unimod_to_mod_mapping = True

    def __init__(
        self,
        modification_mapping: Optional[Dict[str, List[str]]] = None,
        rt_unit: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize the reader.

        Parameters
        ----------
        modification_mapping : dict, optional
            Additional/override PEAKS-name -> AlphaBase-name mappings, merged
            on top of the `"peaks"` entry in `psm_reader.yaml`'s
            `modification_mappings` section. See
            `alphabase.psm_reader.modification_mapper.ModificationMapper`.

        rt_unit : str, optional
            Unit of the retention time column. Defaults to
            `psm_reader_yaml["peaks"]["rt_unit"]` ("second" - the only
            convention observed in PEAKS exports so far).

        column_mapping : dict, optional
            PEAKS column name -> AlphaBase column name.
            Defaults to `psm_reader_yaml["peaks"]["column_mapping"]`.

        """
        self._rt_unit = (
            rt_unit
            if rt_unit is not None
            else psm_reader_yaml[self._reader_type]["rt_unit"]
        )
        if self._rt_unit != "second":
            # TODO: implement "minute"/"irt" if/when a PEAKS export using them shows up.
            raise NotImplementedError(
                f"rt_unit={self._rt_unit!r} is not supported yet, only 'second' has been observed in PEAKS exports."
            )

        self._modification_mapper = ModificationMapper(
            modification_mapping,
            reader_yaml=copy.deepcopy(psm_reader_yaml),
            mapping_type=psm_reader_yaml[self._reader_type][
                "modification_mapping_type"
            ],
            add_unimod_to_mod_mapping=self._add_unimod_to_mod_mapping,
        )

        self._column_mapping = (
            column_mapping or psm_reader_yaml[self._reader_type]["column_mapping"]
        )

    def import_file(self, filename: str) -> SpecLibFlat:
        """Read a PEAKS library TSV and return it as a SpecLibFlat.

        This is the reader's main entry point (mirrors the `import_file`
        convention used by every other reader in `alphabase.psm_reader`).

        Parameters
        ----------
        filename : str
            Path to the PEAKS library export, e.g. "lib.tsv".

        Returns
        -------
        SpecLibFlat
            Library with `precursor_df` (one row per precursor) and
            `fragment_df` (one row per fragment ion), linked via the
            `flat_frag_start_idx` / `flat_frag_stop_idx` columns.

        """
        raw_df = self._load_file(filename)

        precursor_df, raw_df = self._parse_precursors(raw_df)
        precursor_df, raw_df = self._harmonize_modifications(precursor_df, raw_df)
        precursor_df, fragment_df = self._parse_fragments(precursor_df, raw_df)

        return self._build_speclib_flat(precursor_df, fragment_df)

    def _load_file(self, filename: str) -> pd.DataFrame:
        """Load the raw PEAKS TSV into a DataFrame."""
        csv_sep = _get_delimiter(filename)
        # keep_default_na=False: an *empty* "Modifications" cell (unmodified
        # peptide) must stay the literal string "", not become NaN, so that
        # _harmonize_modifications can tell "no mods" apart from "bad row".
        return pd.read_csv(filename, sep=csv_sep, keep_default_na=False)

    def _parse_precursors(
        self, raw_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build the precursor-level columns: sequence, charge, precursor_mz, rt, nAA.

        Also validates the input:
        - `sequence`, `charge`, `precursor_mz` are required for every row - a
          blank value in any of them means the row can't be identified at
          all, which is treated as corrupt input and raises immediately
          (informatively, naming the offending rows) rather than surfacing a
          cryptic dtype-cast error later on.
        - `sequence` characters must be in AlphaBase's recognized AA alphabet
          (`_VALID_SEQUENCE_CHARS`). Unlike the required-field check, this is
          treated as a soft per-row problem: the row is dropped with a
          warning rather than aborting the whole file, since one malformed
          sequence shouldn't block the rest of a large library.
        """
        precursor_df = pd.DataFrame(index=raw_df.index)

        for alphabase_col, peaks_col in self._column_mapping.items():
            precursor_df[alphabase_col] = raw_df[peaks_col]

        self._validate_required_columns(precursor_df)

        precursor_df[PsmDfCols.CHARGE] = precursor_df[PsmDfCols.CHARGE].astype(np.int8)
        precursor_df[PsmDfCols.PRECURSOR_MZ] = precursor_df[
            PsmDfCols.PRECURSOR_MZ
        ].astype(np.float64)

        keep_mask = precursor_df[PsmDfCols.SEQUENCE].apply(self._is_valid_sequence)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            bad_sequences = sorted(
                set(precursor_df.loc[~keep_mask, PsmDfCols.SEQUENCE])
            )
            logger.warning(
                f"Dropped {n_dropped} precursor(s) with non-standard sequence characters: {bad_sequences}"
            )
        precursor_df = precursor_df.loc[keep_mask].reset_index(drop=True)
        raw_df = raw_df.loc[keep_mask].reset_index(drop=True)

        precursor_df[PsmDfCols.NAA] = precursor_df[PsmDfCols.SEQUENCE].str.len()

        # AlphaBase's internal `rt` convention is minutes (mirrors
        # PSMReaderBase._normalize_rt's handling of rt_unit == "second"). RT
        # is not a required field (unlike sequence/charge/precursor_mz): a
        # blank/unparseable value becomes NaN rather than raising, since RT
        # isn't needed to identify a precursor.
        precursor_df[PsmDfCols.RT] = (
            pd.to_numeric(precursor_df[PsmDfCols.RT], errors="coerce") / 60
        )
        max_rt = precursor_df[PsmDfCols.RT].max()
        precursor_df[PsmDfCols.RT_NORM] = (
            (precursor_df[PsmDfCols.RT] / max_rt).clip(0, 1)
            if pd.notna(max_rt) and max_rt > 0
            else 0.0
        )

        return precursor_df, raw_df

    @staticmethod
    def _validate_required_columns(precursor_df: pd.DataFrame) -> None:
        """Raise an informative error if any required field is blank."""
        for col in _REQUIRED_COLUMNS:
            is_blank = precursor_df[col].astype(str).str.strip() == ""
            if is_blank.any():
                bad_rows = precursor_df.index[is_blank].tolist()
                raise ValueError(
                    f"Missing required value(s) in column {col!r} at row(s) {bad_rows}. "
                    "Every precursor must have a sequence, charge and precursor m/z."
                )

    @staticmethod
    def _is_valid_sequence(sequence: str) -> bool:
        return len(sequence) > 0 and set(sequence) <= _VALID_SEQUENCE_CHARS

    def _harmonize_modifications(
        self, precursor_df: pd.DataFrame, raw_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Translate PEAKS "Modifications" strings into AlphaBase `mods`/`mod_sites`.

        Input examples (from dia_db_library.tsv, column "Modifications"):
            ""                                                    -> no modifications
            "10-Carboxymethyl-(58.01)"                            -> one modification
            "0-Acetylation (Protein N-term)-(42.01);4-Carboxymethyl-(58.01)"  -> multiple

        Target AlphaBase convention (see `alphabase.psm_reader.keys.PsmDfCols`):
            mods       = ';'-joined unimod-style names, e.g. "Acetyl@Protein_N-term;Carboxymethyl@C"
            mod_sites  = ';'-joined positions, same order as `mods`; "0" = N-term, "-1" = C-term

        Position convention (verified against all 8,094 modification
        occurrences in the example file, 0 mismatches): PEAKS reports
        positions as a 0-based index into the sequence string for side-chain
        modifications (e.g. "10-Carboxymethyl" on "AAAGEFADDPCSSVK" lands on
        the 'C' at 0-based index 10). AlphaBase side-chain `mod_sites` are
        1-based, so side-chain sites are PEAKS position + 1. N-terminal
        modifications are always reported at PEAKS position 0 and are mapped
        to AlphaBase's fixed N-term site "0" directly (not position + 1).
        """
        mods_list = []
        mod_sites_list = []
        keep_mask = []

        for mods_cell in raw_df["Modifications"]:
            if mods_cell == "":
                mods_list.append("")
                mod_sites_list.append("")
                keep_mask.append(True)
                continue

            names = []
            sites = []
            unknown = []
            for match in _PEAKS_MOD_TOKEN_RE.finditer(mods_cell):
                peaks_name = match.group("name")
                alphabase_name = self._modification_mapper.rev_mod_mapping.get(
                    peaks_name
                )
                if alphabase_name is None:
                    unknown.append(peaks_name)
                    continue

                peaks_position = int(match.group("position"))
                site = self._mod_site(alphabase_name, peaks_position)

                names.append(alphabase_name)
                sites.append(site)

            if unknown:
                logger.warning(
                    f"Unknown PEAKS modification(s) {set(unknown)} in {mods_cell!r}. "
                    "Dropping this precursor. Add a mapping via `modification_mapping` to keep it."
                )
                mods_list.append(None)
                mod_sites_list.append(None)
                keep_mask.append(False)
                continue

            mods_list.append(";".join(names))
            mod_sites_list.append(";".join(sites))
            keep_mask.append(True)

        precursor_df = precursor_df.copy()
        precursor_df[PsmDfCols.MODS] = mods_list
        precursor_df[PsmDfCols.MOD_SITES] = mod_sites_list

        keep_mask = pd.Series(keep_mask, index=precursor_df.index)
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            warnings.warn(
                f"Dropped {n_dropped} precursor(s) with unmapped modifications.",
                stacklevel=2,
            )

        return (
            precursor_df.loc[keep_mask].reset_index(drop=True),
            raw_df.loc[keep_mask].reset_index(drop=True),
        )

    @staticmethod
    def _mod_site(alphabase_mod_name: str, peaks_position: int) -> str:
        """Convert a PEAKS 0-based position to an AlphaBase `mod_sites` token."""
        mod_site_token = alphabase_mod_name.split("@", 1)[1]
        if "N-term" in mod_site_token:
            return _PROTEIN_N_TERM_SITE
        if "C-term" in mod_site_token:
            return _C_TERM_SITE
        return str(peaks_position + 1)

    def _parse_fragments(
        self, precursor_df: pd.DataFrame, raw_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Explode the packed "Peaks List" column into a flat fragment_df.

        Also attaches `flat_frag_start_idx` / `flat_frag_stop_idx` to
        `precursor_df` so each precursor can find its own slice of
        `fragment_df` (the contract `SpecLibFlat` relies on).

        `position` follows `alphabase.peptide.fragment.flatten_fragments`'
        convention: 0-based, left-to-right (N-term to C-term) backbone
        cleavage position - same formula used in
        `alphabase/spectral_library/reader.py::LibraryReaderBase._get_fragment_intensity`.
        `type` and `loss_type` are stored as the integer codes AlphaBase
        uses internally (`SERIES_MAPPING`/`LOSS_MAPPING`), not as strings.

        A fragment token that doesn't match `_PEAKS_FRAGMENT_TOKEN_RE` at all
        (garbled ion label) is logged and skipped - the rest of that
        precursor's fragments, and the precursor itself, are kept. A token
        that *does* parse but reports an ion number that's chemically
        impossible for the peptide's length (e.g. "y10" on a 5-residue
        peptide, which would compute a negative backbone position) raises
        instead: unlike a garbled label, this can only mean the position math
        above disagrees with the input in a way that silently produces a
        wrong-but-plausible-looking number (`position` is unsigned, so a
        negative value would otherwise wrap around to a huge one) - worth
        failing loudly rather than writing bad data.
        """
        mz_col = []
        intensity_col = []
        type_col = []
        number_col = []
        position_col = []
        charge_col = []
        loss_type_col = []

        frag_start_idx = np.empty(len(precursor_df), dtype=np.int64)
        frag_stop_idx = np.empty(len(precursor_df), dtype=np.int64)
        running_idx = 0

        for row_i, (peaks_list, n_aa) in enumerate(
            zip(raw_df["Peaks List"], precursor_df[PsmDfCols.NAA])
        ):
            frag_start_idx[row_i] = running_idx

            # An empty "Peaks List" (Peaks Count == 0) means zero fragments,
            # not one token: "".split(";") would otherwise yield [""], which
            # doesn't match the fragment regex and would be treated as a
            # (spuriously) unparseable token.
            tokens = peaks_list.split(";") if peaks_list else []

            for token in tokens:
                match = _PEAKS_FRAGMENT_TOKEN_RE.match(token)
                if match is None:
                    logger.warning(
                        f"Skipping unparseable PEAKS fragment token {token!r} (row {row_i})."
                    )
                    continue

                ion_type = match.group("ion_type")
                ion_number = int(match.group("ion_number"))
                neutral_loss = match.group("neutral_loss") or ""
                frag_charge = int(match.group("frag_charge") or 1)

                position = (
                    ion_number - 1
                    if ion_type == "b"
                    else n_aa - ion_number - 1  # ion_type == "y"
                )
                if not (0 <= position <= n_aa - 2):
                    raise ValueError(
                        f"Fragment {token!r} (row {row_i}) is not chemically possible "
                        f"for a {n_aa}-residue peptide: ion number {ion_number} implies "
                        f"backbone position {position}, expected 0..{n_aa - 2}."
                    )

                mz_col.append(float(match.group("mz")))
                intensity_col.append(float(match.group("intensity")))
                type_col.append(SERIES_MAPPING[ion_type])
                number_col.append(ion_number)
                position_col.append(position)
                charge_col.append(frag_charge)
                loss_type_col.append(LOSS_MAPPING[neutral_loss.lstrip("-") or ""])

                running_idx += 1

            frag_stop_idx[row_i] = running_idx

        precursor_df = precursor_df.copy()
        precursor_df["flat_frag_start_idx"] = frag_start_idx
        precursor_df["flat_frag_stop_idx"] = frag_stop_idx

        fragment_df = pd.DataFrame(
            {
                "mz": mz_col,
                "intensity": intensity_col,
                "type": np.array(type_col, dtype=np.uint8),
                "number": np.array(number_col, dtype=np.uint32),
                "position": np.array(position_col, dtype=np.uint32),
                "charge": np.array(charge_col, dtype=np.uint8),
                "loss_type": np.array(loss_type_col, dtype=np.int16),
            }
        )

        return precursor_df, fragment_df

    def _build_speclib_flat(
        self, precursor_df: pd.DataFrame, fragment_df: pd.DataFrame
    ) -> SpecLibFlat:
        """Assemble the final SpecLibFlat object from the parsed DataFrames.

        precursor_mz is kept as PEAKS' observed "m/z" value as-is (unlike
        readers that recalculate it from sequence+mods+charge) since PEAKS
        already reports the measured value directly.
        """
        speclib = SpecLibFlat()
        # `precursor_df` setter runs `refine_precursor_df` (fills nAA if
        # missing, casts charge/mod_sites dtypes) - see SpecLibBase.
        speclib.precursor_df = precursor_df
        # `fragment_df` has no public setter (see SpecLibFlat.fragment_df),
        # so it's assigned directly, same as `SpecLibFlat.parse_base_library` does.
        speclib._fragment_df = fragment_df  # noqa: SLF001
        return speclib
