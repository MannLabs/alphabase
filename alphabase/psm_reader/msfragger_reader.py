"""MSFragger reader."""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyteomics import pepxml

from alphabase.constants.aa import AA_ASCII_MASS
from alphabase.constants.atom import MASS_H, MASS_O
from alphabase.constants.modification import MOD_MASS, ModificationKeys
from alphabase.psm_reader.keys import MsFraggerTokens, PsmDfCols
from alphabase.psm_reader.psm_reader import (
    PSMReaderBase,
    psm_reader_provider,
    psm_reader_yaml,
)


def _is_all_fragger_decoy(proteins: List[str]) -> bool:
    """Check if all proteins are MSFragger decoy entries.

    Parameters
    ----------
    proteins : List[str]
        List of protein identifiers

    Returns
    -------
    bool
        True if all proteins start with 'rev_' (case-insensitive)

    """
    return all(
        prot.lower().startswith(MsFraggerTokens.DECOY_PREFIX) for prot in proteins
    )


def _extract_position(entry: str) -> Tuple[int, str]:
    """Extract leading position digits from modification entry.

    Parameters
    ----------
    entry : str
        Modification entry like '5S(79.9663)'

    Returns
    -------
    tuple
        (position, remainder) e.g. (5, 'S(79.9663)')

    Raises
    ------
    ValueError
        If entry has no leading position digits

    """
    position = ""
    for char in entry:
        if char.isdigit():
            position += char
        else:
            break

    if not position:
        raise ValueError(
            f"Invalid modification entry '{entry}': expected format "
            f"'<position><AA>(<mass>)' (e.g., '5S(79.9663)'), "
            f"'N-term(<mass>)', or 'C-term(<mass>)'."
        )

    return int(position), entry[len(position) :]


def _extract_mass_shift(entry: str) -> float:
    """Extract mass shift from entry like 'N-term(304.2071)' or 'S(79.9663)'."""
    return float(
        entry.split(MsFraggerTokens.MOD_START)[1].rstrip(MsFraggerTokens.MOD_STOP)
    )


def _parse_lookup_key(lookup_key: str, entry: str) -> Tuple[str, float]:
    """Parse lookup key into amino acid and mass shift.

    Parameters
    ----------
    lookup_key : str
        Lookup key like 'S(79.9663)'
    entry : str
        Original entry for error messages

    Returns
    -------
    tuple
        (amino_acid, mass_shift)

    """
    if MsFraggerTokens.MOD_START not in lookup_key:
        raise ValueError(
            f"Invalid modification entry '{entry}': "
            f"could not parse amino acid and mass."
        )
    amino_acid = lookup_key.split(MsFraggerTokens.MOD_START)[0]
    mass_shift = _extract_mass_shift(lookup_key)
    return amino_acid, mass_shift


class MSFraggerModificationTranslator:
    """Translate MSFragger PSM.TSV modifications to alphabase format."""

    def __init__(
        self,
        mass_mapped_mods: List[str],
        mod_mass_tol: float,
        rev_mod_mapping: Dict[str, str],
    ):
        """Initialize MSFragger modification translator.

        Parameters
        ----------
        mass_mapped_mods : List[str]
            List of modification names to match against (e.g., ['Phospho@S', 'Oxidation@M'])
        mod_mass_tol : float
            Mass tolerance for matching modifications in Daltons.
        rev_mod_mapping : Dict[str, str]
            Reverse mapping from MSFragger format to alphabase format.
            Keys use MSFragger's native format: 'AA(mass)' or 'N-term(mass)'.
            Values use alphabase format: 'Mod@AA'.

        """
        self._mass_mapped_mods = mass_mapped_mods
        self._mod_mass_tol = mod_mass_tol
        self._rev_mod_mapping = rev_mod_mapping

    def translate(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """Translate modifications from MSFragger assigned modifications.

        Parameters
        ----------
        psm_df : pd.DataFrame
            DataFrame with PsmDfCols.TMP_MODS column containing raw assigned modifications strings

        Returns
        -------
        pd.DataFrame
            The input DataFrame with 'mods' and 'mod_sites' columns added

        """
        mods_list = []
        sites_list = []

        for _, row in psm_df.iterrows():
            assigned_mods = row.get(PsmDfCols.TMP_MODS, "")
            mods, sites = self._parse_assigned_modifications(assigned_mods)
            mods_list.append(mods)
            sites_list.append(sites)

        psm_df[PsmDfCols.MODS] = mods_list
        psm_df[PsmDfCols.MOD_SITES] = sites_list

        return psm_df

    def _parse_assigned_modifications(self, assigned_mods: str) -> Tuple[str, str]:
        """Parse MSFragger Assigned Modifications string.

        Parameters
        ----------
        assigned_mods : str
            MSFragger format: "5S(79.9663), 7S(79.9663), N-term(304.2071)"

        Returns
        -------
        tuple
            (mods_str, sites_str) where mods and sites are semicolon-separated.
            Example: ("Phospho@S;Phospho@S;TMT@Any_N-term", "5;7;0")

        """
        if not assigned_mods:
            return "", ""

        mod_entries = [
            m.strip() for m in assigned_mods.split(MsFraggerTokens.MOD_SEPARATOR)
        ]
        mods = []
        sites = []

        for entry in mod_entries:
            if not entry:
                continue
            mod_name, site = self._parse_single_modification(entry)
            mods.append(mod_name)
            sites.append(site)

        return ModificationKeys.SEPARATOR.join(mods), ModificationKeys.SEPARATOR.join(
            sites
        )

    def _parse_single_modification(self, entry: str) -> Tuple[str, str]:
        """Parse a single modification entry."""
        if entry.startswith(MsFraggerTokens.N_TERM):
            return self._resolve_terminal_mod(entry, "Any_N-term"), "0"
        if entry.startswith(MsFraggerTokens.C_TERM):
            return self._resolve_terminal_mod(entry, "Any_C-term"), "-1"

        position, lookup_key = _extract_position(entry)
        mod_name = self._resolve_positional_mod(lookup_key, entry, position)
        return mod_name, str(position)

    def _resolve_terminal_mod(self, entry: str, aa_or_term: str) -> str:
        """Resolve terminal modification name, checking rev_mod_mapping first."""
        if entry in self._rev_mod_mapping:
            return self._rev_mod_mapping[entry]
        return self._match_mod_by_mass(_extract_mass_shift(entry), aa_or_term)

    def _resolve_positional_mod(
        self, lookup_key: str, entry: str, position: int
    ) -> str:
        """Resolve positional modification name from lookup key like 'S(79.9663)'."""
        if lookup_key in self._rev_mod_mapping:
            return self._rev_mod_mapping[lookup_key]
        amino_acid, mass_shift = _parse_lookup_key(lookup_key, entry)
        return self._match_mod_by_mass(mass_shift, amino_acid, position)

    def _match_mod_by_mass(
        self, mass_shift: float, aa_or_term: str, position: Optional[int] = None
    ) -> str:
        """Match mass shift to modification name by finding the closest mass match.

        Parameters
        ----------
        mass_shift : float
            Mass shift in Daltons (e.g., 79.9663 for phosphorylation)
        aa_or_term : str
            Amino acid single letter code or terminal (Any_N-term, Any_C-term)
        position : Optional[int]
            Position in peptide (1-indexed). Used to match AA^Any_N-term mods at position 1.

        Returns
        -------
        str
            Modification name in alphabase format (e.g., 'Phospho@S')

        Raises
        ------
        ValueError
            If no matching modification is found

        """
        is_terminal = aa_or_term in [
            ModificationKeys.ANY_N_TERM,
            ModificationKeys.ANY_C_TERM,
        ]
        best_match = None
        best_mass_diff = float("inf")

        for mod_name in self._mass_mapped_mods:
            if mod_name not in MOD_MASS:
                continue

            mass_diff = abs(mass_shift - MOD_MASS[mod_name])
            if mass_diff >= self._mod_mass_tol or mass_diff >= best_mass_diff:
                continue

            mod_site = mod_name.split(ModificationKeys.SITE_SEPARATOR)[1]
            is_exact_match = mod_site == aa_or_term
            is_term_match = is_terminal and mod_site.endswith(aa_or_term)
            # Match AA^Any_N-term mods (e.g., Ammonia-loss@C^Any_N-term) at position 1
            is_nterm_aa_match = (
                position == 1
                and mod_site.startswith(aa_or_term)
                and mod_site.endswith(ModificationKeys.ANY_N_TERM)
            )
            if is_exact_match or is_term_match or is_nterm_aa_match:
                best_match = mod_name
                best_mass_diff = mass_diff

        if best_match is not None:
            return best_match

        raise ValueError(
            f"Unknown modification: mass_shift={mass_shift:.4f} at {aa_or_term}. "
            f"Add the modification to 'mass_mapped_mods' in psm_reader.yaml or extend "
            f"the reader's _mass_mapped_mods list before importing."
        )


def _get_mods_from_masses(  # noqa: PLR0912, C901 too many branches, too complex TODO: refactor
    sequence: str,
    msf_aa_mods: List[str],
    mass_mapped_mods: List[str],
    mod_mass_tol: float,
) -> Tuple[str, str, str, str]:
    mods = []
    mod_sites = []
    aa_mass_diffs = []
    aa_mass_diff_sites = []
    for mod in msf_aa_mods:
        _mass_str, site_str = mod.split(ModificationKeys.SITE_SEPARATOR)
        mod_mass = float(_mass_str)
        site = int(site_str)
        cterm_position = len(sequence) + 1
        if site > 0:
            if site < cterm_position:
                mod_mass = mod_mass - AA_ASCII_MASS[ord(sequence[site - 1])]
            else:
                mod_mass -= 2 * MASS_H + MASS_O
        else:
            mod_mass -= MASS_H

        mod_translated = False
        for mod_name in mass_mapped_mods:
            if abs(mod_mass - MOD_MASS[mod_name]) < mod_mass_tol:
                if site == 0:
                    _mod = (
                        mod_name.split(ModificationKeys.SITE_SEPARATOR)[0]
                        + ModificationKeys.SITE_SEPARATOR
                        + ModificationKeys.ANY_N_TERM
                    )
                elif site == 1:
                    if mod_name.endswith("^Any_N-term"):
                        _mod = mod_name
                        site_str = "0"
                    else:
                        _mod = (
                            mod_name.split(ModificationKeys.SITE_SEPARATOR)[0]
                            + ModificationKeys.SITE_SEPARATOR
                            + sequence[0]
                        )
                elif site == cterm_position:
                    if mod_name.endswith("C-term"):
                        _mod = mod_name
                    else:
                        _mod = (
                            mod_name.split(ModificationKeys.SITE_SEPARATOR)[0]
                            + ModificationKeys.SITE_SEPARATOR
                            + ModificationKeys.ANY_C_TERM
                        )  # what if only Protein C-term is listed?
                    site_str = "-1"
                else:
                    _mod = (
                        mod_name.split(ModificationKeys.SITE_SEPARATOR)[0]
                        + ModificationKeys.SITE_SEPARATOR
                        + sequence[site - 1]
                    )
                if _mod in MOD_MASS:
                    mods.append(_mod)
                    mod_sites.append(site_str)
                    mod_translated = True
                    break
        if not mod_translated:
            aa_mass_diffs.append(f"{mod_mass:.5f}")
            aa_mass_diff_sites.append(site_str)
    return (
        ModificationKeys.SEPARATOR.join(mods),
        ModificationKeys.SEPARATOR.join(mod_sites),
        ModificationKeys.SEPARATOR.join(aa_mass_diffs),
        ModificationKeys.SEPARATOR.join(aa_mass_diff_sites),
    )


class MSFraggerPsmTsvReader(PSMReaderBase):
    """Reader for MSFragger's psm.tsv file."""

    _reader_type = "msfragger_psm_tsv"

    def __init__(
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        rt_unit: Optional[str] = None,
        **kwargs,
    ):
        """Initialize MSFragger PSM TSV reader.

        Parameters
        ----------
        column_mapping : Optional[dict]
            Custom column name mapping.
        modification_mapping : Optional[dict]
            Custom modification mapping from alphabase format to MSFragger format.
            Keys use alphabase format: 'Mod@AA'.
            Values use MSFragger's native format: 'AA(mass)' or 'N-term(mass)' or 'C-term(mass)'.
            Example: {'Phospho@S': 'S(79.9663)', 'TMTpro@Any_N-term': 'N-term(304.2071)'}
        fdr : float
            False discovery rate threshold. Default: 0.01
        keep_decoy : bool
            Whether to keep decoy hits. Default: False
        rt_unit : Optional[str]
            Retention time unit.
        **kwargs
            Additional arguments passed to PSMReaderBase.

        """
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            rt_unit=rt_unit,
            **kwargs,
        )
        self._mass_mapped_mods = psm_reader_yaml.get(self._reader_type, {}).get(
            "mass_mapped_mods", []
        )
        self._mod_mass_tol = psm_reader_yaml.get(self._reader_type, {}).get(
            "mod_mass_tol", 0.1
        )

    def _translate_modifications(self) -> None:
        """No-op: modification translation is handled in _load_modifications."""

    def _load_file(self, filename: str) -> pd.DataFrame:
        """Load MSFragger PSM TSV file."""
        return pd.read_csv(filename, sep="\t", keep_default_na=False)

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """MSFragger PSM TSV preprocessing."""
        df.fillna("", inplace=True)
        df[[PsmDfCols.RAW_NAME, PsmDfCols.SCAN_NUM]] = (
            df["Spectrum"].str.split(".").apply(lambda x: pd.Series([x[0], int(x[1])]))
        )
        return df

    def _translate_decoy(self) -> None:
        self._psm_df[PsmDfCols.DECOY] = (
            self._psm_df[PsmDfCols.DECOY] == "true"
        ).astype(np.int8)

    def _translate_score(self) -> None:
        """MSFragger Hyperscore is already in correct format (larger = better)."""

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:  # noqa: ARG002
        """Parse modifications from PsmDfCols.TMP_MODS column (mapped from 'Assigned Modifications')."""
        modification_translator = MSFraggerModificationTranslator(
            mass_mapped_mods=self._mass_mapped_mods,
            mod_mass_tol=self._mod_mass_tol,
            rev_mod_mapping=self._modification_mapper.rev_mod_mapping or {},
        )
        self._psm_df = modification_translator.translate(self._psm_df)


class MSFraggerPepXMLReader(PSMReaderBase):
    """Reader for MSFragger's pep.xml file."""

    _reader_type = "msfragger_pepxml"

    def __init__(  # noqa: PLR0913, D417 # too many arguments in function definition, missing argument descriptions
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        # mod_seq_columns: Optional[List[str]] = None,# TODO: not needed here?
        fdr: float = 0.001,  # refers to E-value in the PepXML
        keep_decoy: bool = False,
        rt_unit: Optional[str] = None,
        # MSFragger reader-specific:
        keep_unknown_aa_mass_diffs: bool = False,
        **kwargs,
    ):
        """Initialize the MSFraggerreader.

        See documentation of `PSMReaderBase` for more information.

        MSFragger is not fully supported as we can only access the pepxml file.

        Parameters
        ----------
            keep_unknown_aa_mass_diffs:
                whether to keep PSMs with unknown amino acid mass differences, default: False


        See documentation of `PSMReaderBase` for the rest of parameters.

        """
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            rt_unit=rt_unit,
            **kwargs,
        )
        self._keep_unknown_aa_mass_diffs = keep_unknown_aa_mass_diffs
        # TODO: should those be set via API, too?
        self._mass_mapped_mods = psm_reader_yaml["msfragger_pepxml"]["mass_mapped_mods"]
        self._mod_mass_tol = psm_reader_yaml["msfragger_pepxml"]["mod_mass_tol"]

    def _translate_modifications(self) -> None:
        """No-op: modification translation is handled in _load_modifications."""

    def _load_file(self, filename: str) -> pd.DataFrame:
        """Load a MsFragger output file to a DataFrame."""
        return pepxml.DataFrame(filename)

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """MsFragger-specific preprocessing of output data."""
        df.fillna("", inplace=True)
        if "ion_mobility" in df.columns:
            df["ion_mobility"] = df["ion_mobility"].astype(float)
        df[PsmDfCols.RAW_NAME] = df["spectrum"].str.split(".").apply(lambda x: x[0])
        df["to_remove"] = 0  # TODO: revisit
        self.column_mapping[PsmDfCols.TO_REMOVE] = "to_remove"
        return df

    def _translate_decoy(self) -> None:
        self._psm_df[PsmDfCols.DECOY] = (
            self._psm_df[PsmDfCols.PROTEINS]
            .apply(_is_all_fragger_decoy)
            .astype(np.int8)
        )

        self._psm_df[PsmDfCols.PROTEINS] = self._psm_df[PsmDfCols.PROTEINS].apply(
            lambda x: ModificationKeys.SEPARATOR.join(x)
        )
        if not self._keep_decoy:
            self._psm_df[PsmDfCols.TO_REMOVE] += self._psm_df[PsmDfCols.DECOY] > 0

    def _translate_score(self) -> None:
        """Translate MSFragger evalue to AlphaBase score: the larger the better."""
        self._psm_df[PsmDfCols.SCORE] = -np.log(self._psm_df[PsmDfCols.SCORE] + 1e-100)

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:
        (
            self._psm_df[PsmDfCols.MODS],
            self._psm_df[PsmDfCols.MOD_SITES],
            self._psm_df[PsmDfCols.AA_MASS_DIFFS],
            self._psm_df[PsmDfCols.AA_MASS_DIFF_SITES],
        ) = zip(
            *origin_df[["peptide", "modifications"]].apply(
                lambda x: _get_mods_from_masses(
                    *x,
                    mass_mapped_mods=self._mass_mapped_mods,
                    mod_mass_tol=self._mod_mass_tol,
                ),
                axis=1,
            )
        )

        if not self._keep_unknown_aa_mass_diffs:
            self._psm_df[PsmDfCols.TO_REMOVE] += (
                self._psm_df[PsmDfCols.AA_MASS_DIFFS] != ""
            )
            self._psm_df.drop(
                columns=[PsmDfCols.AA_MASS_DIFFS, PsmDfCols.AA_MASS_DIFF_SITES],
                inplace=True,
            )

    def _post_process(self, origin_df: pd.DataFrame) -> None:
        self._psm_df = (
            self._psm_df.query(f"{PsmDfCols.TO_REMOVE}==0")
            .drop(columns=PsmDfCols.TO_REMOVE)
            .reset_index(drop=True)
        )
        super()._post_process(origin_df)


class MSFraggerPepXML(MSFraggerPepXMLReader):
    """Deprecated."""

    def __init__(self, *args, **kwargs):
        """Deprecated."""
        warnings.warn(
            "MSFraggerPepXML is deprecated and will ne removed in alphabase>1.5.0.",
            "Please use the equivalent MSFraggerPepXMLReader instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)


def register_readers() -> None:
    """Register MSFragger readers."""
    psm_reader_provider.register_reader("msfragger_psm_tsv", MSFraggerPsmTsvReader)
    psm_reader_provider.register_reader("msfragger", MSFraggerPsmTsvReader)
    psm_reader_provider.register_reader("msfragger_pepxml", MSFraggerPepXMLReader)
