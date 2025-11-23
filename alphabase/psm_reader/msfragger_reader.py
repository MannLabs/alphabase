"""MSFragger reader."""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pyteomics import pepxml

from alphabase.constants.aa import AA_ASCII_MASS
from alphabase.constants.atom import MASS_H, MASS_O
from alphabase.constants.modification import MOD_MASS
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import (
    PSMReaderBase,
    psm_reader_provider,
    psm_reader_yaml,
)


def _is_fragger_decoy(proteins: List[str]) -> bool:
    return all(prot.lower().startswith("rev_") for prot in proteins)


class MSFraggerModificationTranslation:
    """Translate MSFragger PSM.TSV modifications to alphabase format."""

    def __init__(
        self,
        mass_mapped_mods: List[str],
        mod_mass_tol: float = 0.1,
    ):
        """Initialize MSFragger modification translator.

        Parameters
        ----------
        mass_mapped_mods : List[str]
            List of modification names to match against (e.g., ['Phospho@S', 'Oxidation@M'])
        mod_mass_tol : float
            Mass tolerance for matching modifications in Daltons. Default: 0.1

        """
        self.mass_mapped_mods = mass_mapped_mods
        self.mod_mass_tol = mod_mass_tol

    def __call__(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """Translate modifications from MSFragger Assigned Modifications column.

        Parameters
        ----------
        psm_df : pd.DataFrame
            DataFrame with 'Assigned Modifications' column

        Returns
        -------
        pd.DataFrame
            The input DataFrame with 'mods' and 'mod_sites' columns added

        """
        psm_df = psm_df.copy()
        mods_list = []
        sites_list = []

        for _, row in psm_df.iterrows():
            assigned_mods = row.get("Assigned Modifications", "")

            if not assigned_mods:
                mods_list.append("")
                sites_list.append("")
                continue

            mods, sites = self._parse_assigned_modifications(assigned_mods)
            mods_list.append(mods)
            sites_list.append(sites)

        psm_df[PsmDfCols.MODS] = mods_list
        psm_df[PsmDfCols.MOD_SITES] = sites_list

        return psm_df

    def _parse_assigned_modifications(self, assigned_mods: str) -> Tuple[str, str]:
        """Parse MSFragger Assigned Modifications string.

        Directly maps mass shifts to modification names without conversion.

        Parameters
        ----------
        assigned_mods : str
            MSFragger format: "5S(79.9663), 7S(79.9663), N-term(304.2071)"

        Returns
        -------
        tuple
            (mods_str, sites_str) where mods and sites are semicolon-separated

        """
        if not assigned_mods:
            return "", ""

        mod_entries = [m.strip() for m in assigned_mods.split(",")]
        mods = []
        sites = []

        for entry in mod_entries:
            if not entry:
                continue

            if entry.startswith("N-term"):
                mass_shift = float(entry.split("(")[1].rstrip(")"))
                mod_name = self._match_mod_by_mass(mass_shift, "Any_N-term")
                mods.append(mod_name)
                sites.append("0")
            elif entry.startswith("C-term"):
                mass_shift = float(entry.split("(")[1].rstrip(")"))
                mod_name = self._match_mod_by_mass(mass_shift, "Any_C-term")
                mods.append(mod_name)
                sites.append("-1")
            else:
                parts = entry.split("(")
                if len(parts) != 2:  # noqa: PLR2004
                    continue

                pos_aa_str, mass_str = parts
                mass_shift = float(mass_str.rstrip(")"))

                pos_str = ""
                aa = ""
                for char in pos_aa_str:
                    if char.isdigit():
                        pos_str += char
                    else:
                        aa += char

                if pos_str and aa:
                    pos = int(pos_str)
                    mod_name = self._match_mod_by_mass(mass_shift, aa)
                    mods.append(mod_name)
                    sites.append(str(pos))

        return ";".join(mods), ";".join(sites)

    def _match_mod_by_mass(self, mass_shift: float, aa_or_term: str) -> str:
        """Match mass shift to modification name.

        Parameters
        ----------
        mass_shift : float
            Mass shift in Daltons (e.g., 79.9663 for phosphorylation)
        aa_or_term : str
            Amino acid single letter code or terminal (Any_N-term, Any_C-term)

        Returns
        -------
        str
            Modification name in alphabase format (e.g., 'Phospho@S')

        Raises
        ------
        ValueError
            If no matching modification is found

        """
        for mod_name in self.mass_mapped_mods:
            if mod_name not in MOD_MASS:
                continue

            mod_mass = MOD_MASS[mod_name]

            if abs(mass_shift - mod_mass) < self.mod_mass_tol:
                mod_base, mod_site = mod_name.split("@")

                if aa_or_term in ["Any_N-term", "Any_C-term"]:
                    if mod_site == aa_or_term or mod_site.endswith(aa_or_term):
                        return mod_name
                else:
                    if mod_site == aa_or_term:
                        return mod_name
                    if mod_site in ["X", "Any"]:
                        return f"{mod_base}@{aa_or_term}"

        raise ValueError(
            f"Unknown modification: mass_shift={mass_shift:.4f} at {aa_or_term}"
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
        _mass_str, site_str = mod.split("@")
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
                    _mod = mod_name.split("@")[0] + "@Any_N-term"
                elif site == 1:
                    if mod_name.endswith("^Any_N-term"):
                        _mod = mod_name
                        site_str = "0"
                    else:
                        _mod = mod_name.split("@")[0] + "@" + sequence[0]
                elif site == cterm_position:
                    if mod_name.endswith("C-term"):
                        _mod = mod_name
                    else:
                        _mod = (
                            mod_name.split("@")[0] + "@Any_C-term"
                        )  # what if only Protein C-term is listed?
                    site_str = "-1"
                else:
                    _mod = mod_name.split("@")[0] + "@" + sequence[site - 1]
                if _mod in MOD_MASS:
                    mods.append(_mod)
                    mod_sites.append(site_str)
                    mod_translated = True
                    break
        if not mod_translated:
            aa_mass_diffs.append(f"{mod_mass:.5f}")
            aa_mass_diff_sites.append(site_str)
    return (
        ";".join(mods),
        ";".join(mod_sites),
        ";".join(aa_mass_diffs),
        ";".join(aa_mass_diff_sites),
    )


class MSFragger_PSM_TSV_Reader(PSMReaderBase):  # noqa: N801 name should use CapWords convention TODO: refactor
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

        See PSMReaderBase documentation for parameters.

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
        pass

    def _load_file(self, filename: str) -> pd.DataFrame:
        """Load MSFragger PSM TSV file."""
        return pd.read_csv(filename, sep="\t", keep_default_na=False)

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """MSFragger PSM TSV preprocessing."""
        df.fillna("", inplace=True)
        df[PsmDfCols.RAW_NAME] = df["Spectrum"].str.split(".").apply(lambda x: x[0])
        df[PsmDfCols.SPEC_IDX] = (
            df["Spectrum"].str.split(".").apply(lambda x: int(x[1]))
        )
        return df

    def _translate_decoy(self) -> None:
        self._psm_df[PsmDfCols.DECOY] = (
            self._psm_df[PsmDfCols.DECOY] == "true"
        ).astype(np.int8)

    def _translate_score(self) -> None:
        """MSFragger Hyperscore is already in correct format (larger = better)."""

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:  # noqa: ARG002
        """Parse modifications from Assigned Modifications column."""
        translator = MSFraggerModificationTranslation(
            mass_mapped_mods=self._mass_mapped_mods,
            mod_mass_tol=self._mod_mass_tol,
        )
        self._psm_df = translator(self._psm_df)


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
        pass

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
            self._psm_df[PsmDfCols.PROTEINS].apply(_is_fragger_decoy).astype(np.int8)
        )

        self._psm_df[PsmDfCols.PROTEINS] = self._psm_df[PsmDfCols.PROTEINS].apply(
            lambda x: ";".join(x)
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
    psm_reader_provider.register_reader("msfragger_psm_tsv", MSFragger_PSM_TSV_Reader)
    psm_reader_provider.register_reader("msfragger", MSFragger_PSM_TSV_Reader)
    psm_reader_provider.register_reader("msfragger_pepxml", MSFraggerPepXMLReader)
