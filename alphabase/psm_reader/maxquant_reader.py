"""Reader for MaxQuant data."""

import warnings
from typing import List, Optional

import numba
import numpy as np
import pandas as pd

from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import (
    PSMReaderBase,
    psm_reader_provider,
)

# make sure all warnings are shown
warnings.filterwarnings("always")


@numba.njit
def replace_parentheses_with_brackets(
    modseq: str,
) -> str:
    """Replace parentheses with brackets in the modified sequence."""
    mod_depth = 0
    for i, aa in enumerate(modseq):
        if aa == "(":
            if mod_depth <= 0:
                modseq = modseq[:i] + "[" + modseq[i + 1 :]
            mod_depth += 1
        elif aa == "[":
            mod_depth += 1
        elif aa == ")":
            mod_depth -= 1
            if mod_depth <= 0:
                modseq = modseq[:i] + "]" + modseq[i + 1 :]
        elif aa == "]":
            mod_depth -= 1
    return modseq


@numba.njit
def parse_mod_seq(
    modseq: str,
    mod_sep: str = "()",
    fixed_C57: bool = True,  # noqa: FBT001, FBT002, N803 TODO: make this  *,fixed_c57  (breaking)
) -> tuple:
    """Extract modifications and sites from the modified sequence (modseq).

    Parameters
    ----------
    modseq : str
        modified sequence to extract modifications.

    mod_sep : str, optional
        separator to indicate the modification section.
        Defaults to '()'

    fixed_C57 : bool
        If Carbamidomethyl@C is a fixed modification
        and not displayed in the sequence. Defaults to True.

    Returns
    -------
    tuple
        str: naked peptide sequence

        str: modification names, separated by ';'

        str: modification sites, separated by ';'.
        0 for N-term; -1 for C-term; 1 to N for normal modifications.

    """
    peptide_mod_seq = modseq
    underscore_for_ncterm = modseq[0] == "_"
    mod_list = []
    site_list = []
    site = peptide_mod_seq.find(mod_sep[0])
    while site != -1:
        site_end = peptide_mod_seq.find(mod_sep[1], site + 1) + 1
        if site_end < len(peptide_mod_seq) and peptide_mod_seq[site_end] == mod_sep[1]:
            site_end += 1
        if underscore_for_ncterm:
            site_list.append(site - 1)
        else:
            site_list.append(site)
        start_mod = site
        if start_mod > 0:
            start_mod -= 1
        mod_list.append(peptide_mod_seq[start_mod:site_end])
        peptide_mod_seq = peptide_mod_seq[:site] + peptide_mod_seq[site_end:]
        site = peptide_mod_seq.find(mod_sep[0], site)

    # patch for phos. How many other modification formats does MQ have?
    site = peptide_mod_seq.find("p")
    while site != -1:
        mod_list.append(peptide_mod_seq[site : site + 2])
        site_list = [i - 1 if i > site else i for i in site_list]
        if underscore_for_ncterm:
            site_list.append(site)
        else:
            site_list.append(site + 1)
        peptide_mod_seq = peptide_mod_seq[:site] + peptide_mod_seq[site + 1 :]
        site = peptide_mod_seq.find("p", site)

    if fixed_C57:
        site = peptide_mod_seq.find("C")
        while site != -1:
            if underscore_for_ncterm:
                site_list.append(site)
            else:
                site_list.append(site + 1)
            mod_list.append("C" + "Carbamidomethyl (C)".join(mod_sep))
            site = peptide_mod_seq.find("C", site + 1)
    sequence = peptide_mod_seq.strip("_")
    n_aa = len(sequence)
    return (
        sequence,
        ";".join(mod_list),
        ";".join([str(i) if i <= n_aa else "-1" for i in site_list]),
    )


class MaxQuantReader(PSMReaderBase):
    """Reader for MaxQuant data."""

    _reader_type = "maxquant"
    _add_unimod_to_mod_mapping = True
    _modification_type = "maxquant"

    def __init__(  # noqa: PLR0913 many arguments in function definition
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        fixed_C57: bool = True,  # noqa: N803 TODO: make this  *,fixed_c57  (breaking)
        mod_seq_columns: Optional[List[str]] = None,
        rt_unit: str = "minute",
        **kwargs,
    ):
        """Reader for MaxQuant msms.txt and evidence.txt.

        Parameters
        ----------
        column_mapping : dict, optional
            By default None. If None, use
            `psm_reader_yaml['maxquant']['column_mapping']`
            (alphabase.psm_reader.psm_reader_yaml).

        modification_mapping : dict, optional
            By default None. If None, use
            `psm_reader_yaml['maxquant']['modification_mapping']`
            (alphabase.psm_reader.psm_reader_yaml).

        fdr : float, optional
            Load PSMs with FDR < this fdr, by default 0.01

        keep_decoy : bool, optional
            If keep decoy PSMs, by default False

        fixed_C57 : bool, optional
            If true, the search engine will not show `Carbamidomethyl`
            in the modified sequences.
            by default True

        mod_seq_columns : list, optional
            The columns to find modified sequences,
            by default ['Modified sequence']

        rt_unit : str, optional
            The unit of RT in the search engine result.
            Defaults to 'minute'.

        **kwargs : dict
            deprecated

        """
        if mod_seq_columns is None:
            mod_seq_columns = [
                "Modified sequence"
            ]  # TODO: why not take from psm_reader.yaml?

        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            rt_unit=rt_unit,
            **kwargs,
        )

        self.fixed_C57 = fixed_C57
        self._mod_seq_columns = mod_seq_columns
        self.mod_seq_column = "Modified sequence"

    def _translate_decoy(self) -> None:
        if PsmDfCols.DECOY in self._psm_df.columns:
            self._psm_df[PsmDfCols.DECOY] = (
                self._psm_df[PsmDfCols.DECOY] == "-"
            ).astype(np.int8)

    def _load_file(self, filename: str) -> pd.DataFrame:
        csv_sep = self._get_table_delimiter(filename)
        df = pd.read_csv(filename, sep=csv_sep, keep_default_na=False)

        self._find_mod_seq_column(df)
        df = df[~pd.isna(df["Retention time"])]
        df.fillna("", inplace=True)

        # remove MBR PSMs as they are currently not supported and will crash import
        mapped_columns = self._find_mapped_columns(df)
        if PsmDfCols.SCAN_NUM in mapped_columns:
            scan_num_col = mapped_columns[PsmDfCols.SCAN_NUM]
            no_ms2_mask = df[scan_num_col] == ""
            if (num_no_ms2_mask := np.sum(no_ms2_mask)) > 0:
                warnings.warn(
                    f"MaxQuant PSM file contains {num_no_ms2_mask} MBR PSMs without MS2 scan. This is not yet supported and rows containing MBR PSMs will be removed."
                )
                df = df[~no_ms2_mask]
                df.reset_index(drop=True, inplace=True)
            df[scan_num_col] = df[scan_num_col].astype(int)

        # if 'K0' in df.columns:
        #     df['Mobility'] = df['K0'] # Bug in MaxQuant? It should be 1/K0
        # min_rt = df['Retention time'].min()
        return df

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:
        if origin_df[self.mod_seq_column].str.contains("[", regex=False).any():
            if origin_df[self.mod_seq_column].str.contains("(", regex=False).any():
                origin_df[self.mod_seq_column] = origin_df[self.mod_seq_column].apply(
                    replace_parentheses_with_brackets
                )
            mod_sep = "[]"
        else:
            mod_sep = "()"

        (seqs, self._psm_df[PsmDfCols.MODS], self._psm_df[PsmDfCols.MOD_SITES]) = zip(
            *origin_df[self.mod_seq_column].apply(
                parse_mod_seq,
                mod_sep=mod_sep,
                fixed_C57=self.fixed_C57,
            )
        )
        if PsmDfCols.SEQUENCE not in self._psm_df.columns:
            self._psm_df[PsmDfCols.SEQUENCE] = seqs


def register_readers() -> None:
    """Register MaxQuant reader."""
    psm_reader_provider.register_reader("maxquant", MaxQuantReader)
