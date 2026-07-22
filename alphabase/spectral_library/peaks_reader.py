"""Reader for spectral libraries exported by PEAKS Studio."""

import re
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

from alphabase.constants.modification import MOD_MASS
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import PSMReaderBase
from alphabase.spectral_library.flat import SpecLibFlat

# Column mapping and modification mapping live in psm_reader.yaml under the
# "peaks" key, like every other reader (see `alphabase/constants/const_files/
# psm_reader.yaml`). The modification mapping there currently covers the
# *only* 3 (name, mass) combinations observed across the MPIB example library
# (52,672 rows, 8,094 modification occurrences) - cross-checked against
# modification.tsv:
#   Carboxymethyl@C          58.005479  (NOT Carbamidomethyl@C, 57.021464 - different mod!)
#   Oxidation@M               15.994915
#   Acetyl@Protein_N-term     42.010565
# Very likely incomplete for PEAKS libraries in general - extend the yaml
# entry as new modifications are encountered (an unmapped modification warns
# and drops the affected precursor rather than crashing - see
# `_load_modifications`).
#
# Some PEAKS modification names are inherently residue-ambiguous: PEAKS names
# a modification once per *shared delta mass*, but AlphaBase/UniMod names are
# always residue-specific, e.g. "Deamidation (NQ)" (0.98 Da) covers both
# "Deamidated@N" and "Deamidated@Q" - see modification.tsv lines 40-42, both
# 0.984016 Da (Deamidated@R too, but PEAKS' own "(NQ)" name never applies to
# R). The yaml string mapping above can't express this (one PEAKS name -> one
# AlphaBase name), so these are resolved separately in `_load_modifications`
# by checking which of the candidate residues is actually present at the
# reported sequence position.
_RESIDUE_AMBIGUOUS_PEAKS_MODS: Dict[str, Dict[str, str]] = {
    "Deamidation (NQ)": {"N": "Deamidated@N", "Q": "Deamidated@Q"},
}

# PEAKS rounds the mass it prints in the "Modifications" column to 2 decimal
# places (e.g. "58.01" for a true Carboxymethyl@C delta of 58.005479 Da), so
# an exact match is never expected. This tolerance only needs to be tight
# enough to catch a *wrong* mapping - e.g. Carboxymethyl (58.01) vs
# Carbamidomethyl (57.02), ~1 Da apart, see above - not to validate
# instrument accuracy.
_MASS_SANITY_TOLERANCE_DA = 0.01

# Matches one modification token in PEAKS' "Modifications" column, e.g.
#   "10-Carboxymethyl-(58.01)" -> position="10", name="Carboxymethyl", mass="58.01"
#   "0-Acetylation (Protein N-term)-(42.01)" -> position="0", name="Acetylation (Protein N-term)"
# Multiple modifications are concatenated with ';' in the source column -
# verified against all 8,094 modification occurrences in the example file,
# 0 mismatches.
_PEAKS_MOD_TOKEN_RE = re.compile(
    r"(?P<position>\d+)-(?P<name>.+?)-\((?P<mass>[\d.]+)\)"
)

_PROTEIN_N_TERM_SITE = "0"
_C_TERM_SITE = "-1"


class PEAKSLibraryReader(PSMReaderBase, SpecLibFlat):
    """Reader for spectral libraries exported by PEAKS Studio (DB search / DIA-DB export).

    Subclasses `PSMReaderBase` like every other reader in `alphabase.psm_reader`
    (rather than parsing the file standalone) so that this reader follows the
    same `import_file` -> `_pre_process` -> `_translate_columns` ->
    `_load_modifications` -> `_post_process` pipeline as the rest of the
    codebase - see `alphabase.psm_reader.sage_reader.SageReaderBase` for the
    closest precedent (modifications reported as position + observed mass,
    not an embedded modified-sequence string).

    Slice 2 of this reader's implementation: real modification parsing.
    `_load_modifications` now translates PEAKS' "pos-Name-(mass)" tokens into
    AlphaBase's `mods`/`mod_sites` (Unimod-name mapping via `psm_reader.yaml`,
    residue-ambiguous name resolution, a mass sanity check, and the PEAKS
    0-based -> AlphaBase 1-based position offset). Sequence validation and
    fragments are still not implemented (later slices), and `_post_process`
    is still un-overridden, so the SpecLibFlat surface
    (`precursor_df`/`fragment_df`) isn't wired up yet either - see
    `import_file()`'s return value / `self.psm_df` for this slice's result.

    Example:
    -------
    >>> reader = PEAKSLibraryReader()
    >>> reader.import_file("lib.tsv")
    >>> reader.psm_df.head()

    """

    _reader_type = "peaks"
    _add_unimod_to_mod_mapping = True

    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        modification_mapping: Optional[Dict[str, List[str]]] = None,
        rt_unit: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the reader.

        Parameters
        ----------
        column_mapping : dict, optional
            PEAKS column name -> AlphaBase column name.
            Defaults to `psm_reader_yaml["peaks"]["column_mapping"]`.

        modification_mapping : dict, optional
            Additional/override PEAKS-name -> AlphaBase-name mappings, merged
            on top of the `"peaks"` entry in `psm_reader.yaml`'s
            `modification_mappings` section. See
            `alphabase.psm_reader.modification_mapper.ModificationMapper`.

        rt_unit : str, optional
            Unit of the retention time column, one of "minute", "second" or
            "irt". Defaults to `psm_reader_yaml["peaks"]["rt_unit"]`
            ("second" - the only convention observed across the available
            PEAKS exports so far).

        **kwargs : dict
            Passed through to `PSMReaderBase.__init__` (e.g. `fdr`).

        """
        # PEAKS fragments carry *observed* mz/intensity, so the dense
        # `charged_frag_types` machinery `SpecLibFlat` otherwise offers is
        # never used by this reader - no need to expose it as a parameter.
        SpecLibFlat.__init__(self)
        # `SpecLibFlat.__init__` doesn't set `_fragment_df` (only
        # `parse_base_library`/HDF loading do), so `fragment_df` would raise
        # AttributeError before the first `import_file()` call without this.
        self._fragment_df = pd.DataFrame()

        PSMReaderBase.__init__(
            self,
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            rt_unit=rt_unit,
            **kwargs,
        )

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:
        """Parse PEAKS "Modifications" strings into AlphaBase `mods`/`mod_sites`.

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
        1-based, so side-chain sites are PEAKS position + 1 (see
        `_convert_mod_site`). N-terminal modifications are always reported at
        PEAKS position 0 and are mapped to AlphaBase's fixed N-term site "0"
        directly (not position + 1).

        Soft-drops (`mods`/`mod_sites` set to `None` here, actually removed
        later once `_post_process` is implemented) precursors with an
        unmapped/mass-mismatched modification. Sequence-character validation
        is added in a later slice.
        """
        mods_list = []
        mod_sites_list = []
        # Collected across the whole file and logged once at the end (not
        # per-row): a single unmapped/mismatched modification is often
        # shared by many precursors, and warning on every one of them would
        # be far noisier than useful.
        all_unknown_reasons = set()
        n_unmapped_mod = 0

        for mods_cell, sequence in zip(
            origin_df["Modifications"], self._psm_df[PsmDfCols.SEQUENCE]
        ):
            if mods_cell == "":
                mods_list.append("")
                mod_sites_list.append("")
                continue

            names, sites, unknown = self._parse_mod_cell(mods_cell, sequence)
            if unknown:
                all_unknown_reasons.update(unknown)
                n_unmapped_mod += 1
                mods_list.append(None)
                mod_sites_list.append(None)
                continue

            mods_list.append(";".join(names))
            mod_sites_list.append(";".join(sites))

        if n_unmapped_mod:
            warnings.warn(
                f"Unknown or mass-mismatched PEAKS modification(s): "
                f"{all_unknown_reasons}. Add a mapping via `modification_mapping` "
                "to keep affected precursors.\n"
                f"Dropped {n_unmapped_mod} precursor(s) with unmapped modifications.",
                stacklevel=2,
            )

        self._psm_df[PsmDfCols.MODS] = mods_list
        self._psm_df[PsmDfCols.MOD_SITES] = mod_sites_list

    def _parse_mod_cell(
        self, mods_cell: str, sequence: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """Parse one non-empty PEAKS "Modifications" cell.

        Returns (names, sites, unknown) - `unknown` holds the raw PEAKS
        names/reasons for any token that couldn't be mapped, which the
        caller treats as reason to drop the whole precursor.
        """
        names = []
        sites = []
        unknown = []
        for match in _PEAKS_MOD_TOKEN_RE.finditer(mods_cell):
            peaks_name = match.group("name")
            peaks_position = int(match.group("position"))
            peaks_mass = float(match.group("mass"))

            alphabase_name = self._modification_mapper.rev_mod_mapping.get(
                peaks_name
            ) or self._resolve_residue_ambiguous_mod(
                peaks_name, sequence, peaks_position
            )
            if alphabase_name is None:
                unknown.append(peaks_name)
                continue

            expected_mass = MOD_MASS.get(alphabase_name)
            if expected_mass is not None and (
                abs(expected_mass - peaks_mass) > _MASS_SANITY_TOLERANCE_DA
            ):
                unknown.append(
                    f"{peaks_name} (would map to {alphabase_name}, but reported "
                    f"mass {peaks_mass} is not within "
                    f"{_MASS_SANITY_TOLERANCE_DA} Da of the expected "
                    f"{expected_mass:.4f})"
                )
                continue

            names.append(alphabase_name)
            sites.append(self._convert_mod_site(alphabase_name, peaks_position))

        return names, sites, unknown

    @staticmethod
    def _resolve_residue_ambiguous_mod(
        peaks_name: str, sequence: str, peaks_position: int
    ) -> Optional[str]:
        """Resolve a residue-ambiguous PEAKS mod name (e.g. "Deamidation (NQ)") via the sequence.

        Returns None if `peaks_name` isn't a known residue-ambiguous name, or
        if the residue actually at `peaks_position` isn't one of its
        candidates (e.g. a corrupt/mismatched position) - both are treated as
        "unknown" by the caller, same as a plain unmapped name.
        """
        residue_options = _RESIDUE_AMBIGUOUS_PEAKS_MODS.get(peaks_name)
        if residue_options is None or not (0 <= peaks_position < len(sequence)):
            return None
        return residue_options.get(sequence[peaks_position])

    @staticmethod
    def _convert_mod_site(alphabase_mod_name: str, peaks_position: int) -> str:
        """Convert a PEAKS 0-based position to an AlphaBase `mod_sites` token."""
        mod_site_token = alphabase_mod_name.split("@", 1)[1]
        if "N-term" in mod_site_token:
            return _PROTEIN_N_TERM_SITE
        if "C-term" in mod_site_token:
            return _C_TERM_SITE
        return str(peaks_position + 1)

    def _translate_modifications(self) -> None:
        """No-op: modification translation is handled in `_load_modifications`."""
