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

    def __init__(
        self,
        **kwargs,
    ):
        """Constructor."""
        raise NotImplementedError("MSFragger_PSM_TSV_Reader for psm.tsv")


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
