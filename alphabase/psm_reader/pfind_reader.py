from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

import alphabase.constants.modification as ap_mod
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import (
    PSMReaderBase,
    psm_reader_provider,
    psm_reader_yaml,
)


def _convert_one_pfind_mod(mod: str) -> Optional[str]:  # noqa:  C901 too complex (11 > 10) TODO: refactor
    if mod[-1] == ")":
        mod = mod[: (mod.find("(") - 1)]
        idx = mod.rfind("[")
        name = mod[:idx]
        site = mod[(idx + 1) :]
    else:
        idx = mod.rfind("[")
        name = mod[:idx]
        site = mod[(idx + 1) : -1]

    if len(site) == 1:
        return_value = name + "@" + site
    elif site == "AnyN-term":
        return_value = name + "@" + "Any_N-term"
    elif site == "ProteinN-term":
        return_value = name + "@" + "Protein_N-term"
    elif site.startswith("AnyN-term"):
        return_value = name + "@" + site[-1] + "^Any_N-term"
    elif site.startswith("ProteinN-term"):
        return_value = name + "@" + site[-1] + "^Protein_N-term"
    elif site == "AnyC-term":
        return_value = name + "@" + "Any_C-term"
    elif site == "ProteinC-term":
        return_value = name + "@" + "Protein_C-term"
    elif site.startswith("AnyC-term"):
        return_value = name + "@" + site[-1] + "^Any_C-term"
    elif site.startswith("ProteinC-term"):
        return_value = name + "@" + site[-1] + "^Protein_C-term"
    else:
        return_value = None

    return return_value


def translate_pFind_mod(mod_str: str) -> Union[str, pd.NA]:  # noqa: N802 name `get_pFind_mods` should be lowercase TODO: used by peptdeep
    if not mod_str:
        return ""
    ret_mods = []
    for mod_ in mod_str.split(";"):
        mod = _convert_one_pfind_mod(mod_)
        if not mod or mod not in ap_mod.MOD_INFO_DICT:
            return pd.NA
        ret_mods.append(mod)
    return ";".join(ret_mods)


def get_pFind_mods(pfind_mod_str: str) -> Tuple[str, str]:  # noqa: N802 name `get_pFind_mods` should be lowercase TODO: used by peptdeep
    pfind_mod_str = pfind_mod_str.strip(";")
    if not pfind_mod_str:
        return "", ""

    items = [item.split(",", 3) for item in pfind_mod_str.split(";")]

    items = [
        ("-1", mod)
        if (mod.endswith("C-term]") or mod[:-2].endswith("C-term"))
        # else ('0', mod) if mod.endswith('N-term]')
        else (site, mod)
        for site, mod in items
    ]
    items = list(zip(*items))
    return ";".join(items[1]), ";".join(items[0])


def parse_pfind_protein(protein: str, keep_reverse: bool = True) -> str:
    proteins = protein.strip("/").split("/")
    return ";".join(
        [
            protein
            for protein in proteins
            if (not protein.startswith("REV_") or keep_reverse)
        ]
    )


class pFindReader(PSMReaderBase):  # noqa: N801 name `pFindReader` should use CapWords convention TODO: used by peptdeep, alpharaw
    def __init__(
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        **kwargs,
    ):
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            **kwargs,
        )

    def _init_column_mapping(self) -> None:
        self.column_mapping = psm_reader_yaml["pfind"]["column_mapping"]

    def _translate_modifications(self) -> None:
        pass

    def _load_file(self, filename: str) -> pd.DataFrame:
        pfind_df = pd.read_csv(
            filename, index_col=False, sep="\t", keep_default_na=False
        )
        pfind_df.fillna("", inplace=True)
        pfind_df = pfind_df[pfind_df.Sequence != ""]
        pfind_df[PsmDfCols.RAW_NAME] = (
            pfind_df["File_Name"].str.split(".").apply(lambda x: x[0])
        )
        pfind_df["Proteins"] = pfind_df["Proteins"].apply(parse_pfind_protein)
        return pfind_df

    def _translate_decoy(self) -> None:
        self._psm_df[PsmDfCols.DECOY] = (
            self._psm_df[PsmDfCols.DECOY] == "decoy"
        ).astype(np.int8)

    def _translate_score(self) -> None:
        self._psm_df[PsmDfCols.SCORE] = -np.log(
            self._psm_df[PsmDfCols.SCORE].astype(float) + 1e-100
        )

    def _load_modifications(self, pfind_df: pd.DataFrame) -> None:
        if len(pfind_df) == 0:
            self._psm_df[PsmDfCols.MODS] = ""
            self._psm_df[PsmDfCols.MOD_SITES] = ""
            return

        (self._psm_df[PsmDfCols.MODS], self._psm_df[PsmDfCols.MOD_SITES]) = zip(
            *pfind_df["Modification"].apply(get_pFind_mods)
        )

        self._psm_df[PsmDfCols.MODS] = self._psm_df[PsmDfCols.MODS].apply(
            translate_pFind_mod
        )


def register_readers() -> None:
    psm_reader_provider.register_reader("pfind", pFindReader)
    psm_reader_provider.register_reader("pfind3", pFindReader)
