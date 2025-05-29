"""pFind reader."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas._libs.missing import NAType

import alphabase.constants.modification as ap_mod
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import (
    PSMReaderBase,
    psm_reader_provider,
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


def translate_pFind_mod(mod_str: str) -> Union[str, NAType]:  # noqa: N802 name `get_pFind_mods` should be lowercase TODO: used by peptdeep
    """Translate pFind modification string."""
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
    """Parse pFind modification string."""
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


def parse_pfind_protein(protein: str, *, keep_reverse: bool = True) -> str:
    """Parse pFind protein string."""
    proteins = protein.strip("/").split("/")
    return ";".join(
        [
            protein
            for protein in proteins
            if (not protein.startswith("REV_") or keep_reverse)
        ]
    )


class pFindReader(PSMReaderBase):  # noqa: N801 name `pFindReader` should use CapWords convention TODO: used by peptdeep, alpharaw
    """Reader for pFind's .txt files."""

    _reader_type = "pfind"

    def _translate_modifications(self) -> None:
        pass

    def _load_file(self, filename: str) -> pd.DataFrame:
        """Load a pFind output file to a DataFrame."""
        return pd.read_csv(filename, index_col=False, sep="\t", keep_default_na=False)

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """pFind-specific preprocessing of output data."""
        df.fillna("", inplace=True)
        df = df[df["Sequence"] != ""]
        df[PsmDfCols.RAW_NAME] = df["File_Name"].str.split(".").apply(lambda x: x[0])
        df["Proteins"] = df["Proteins"].apply(parse_pfind_protein)
        return df

    def _translate_decoy(self) -> None:
        self._psm_df[PsmDfCols.DECOY] = (
            self._psm_df[PsmDfCols.DECOY] == "decoy"
        ).astype(np.int8)

    def _translate_score(self) -> None:
        """Translate pFind pvalue to AlphaBase score: the larger the better."""
        self._psm_df[PsmDfCols.SCORE] = -np.log(
            self._psm_df[PsmDfCols.SCORE].astype(float) + 1e-100
        )

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:
        mods, mod_sites = zip(*origin_df["Modification"].apply(get_pFind_mods))

        self._psm_df[PsmDfCols.MODS] = [translate_pFind_mod(mod) for mod in mods]
        self._psm_df[PsmDfCols.MOD_SITES] = mod_sites


def register_readers() -> None:
    """Register pFind readers."""
    psm_reader_provider.register_reader("pfind", pFindReader)
    psm_reader_provider.register_reader("pfind3", pFindReader)
