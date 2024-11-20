"""Utility functions for PSM readers."""

from typing import Dict, List, Tuple, Union

import pandas as pd
from pandas._libs.missing import NAType

from alphabase.constants.modification import MOD_DF

MOD_TO_UNIMOD_DICT = {}
for mod_name, unimod_id_ in MOD_DF[["mod_name", "unimod_id"]].to_numpy():
    unimod_id = int(unimod_id_)
    if unimod_id in (-1, "-1"):
        continue
    if mod_name[-2] == "@":
        MOD_TO_UNIMOD_DICT[mod_name] = f"{mod_name[-1]}(UniMod:{unimod_id})"
    else:
        MOD_TO_UNIMOD_DICT[mod_name] = f"_(UniMod:{unimod_id})"


def translate_modifications(
    mod_str: str, mod_dict: Dict
) -> Tuple[Union[str, NAType], List[str]]:
    """Translate modifications of `mod_str` to the AlphaBase format mapped by mod_dict.

    Parameters
    ----------
    mod_str : str
        mod list in str format, seperated by ';',
        e.g. ModA;ModB
    mod_dict : dict
        translate mod dict from others to AlphaBase,
        e.g. for pFind, key=['Phospho[S]','Oxidation[M]'],
        value=['Phospho@S','Oxidation@M']

    Returns
    -------
    str
        new mods in AlphaBase format seperated by ';'. if any
        modification is not in `mod_dict`, return pd.NA.

    """
    if mod_str == "":
        return "", []

    ret_mods = []
    unknown_mods = []

    for mod in mod_str.split(";"):
        if mod in mod_dict:
            ret_mods.append(mod_dict[mod])
        else:
            unknown_mods.append(mod)

    if len(unknown_mods) > 0:
        return pd.NA, unknown_mods

    return ";".join(ret_mods), []


def keep_modifications(mod_str: str, mod_set: set) -> Union[str, NAType]:
    """Return modifications if they are in mod_set, pd.NA otherwise.

    Parameters
    ----------
    mod_str : str
        mod list in str format, seperated by ';',
        e.g. Oxidation@M;Phospho@S.
    mod_set : set
        mod set to check

    Returns
    -------
    str
        original `mod_str` if all modifications are in mod_set
        else pd.NA.

    """
    if not mod_str:
        return ""
    for mod in mod_str.split(";"):
        if mod not in mod_set:
            return pd.NA
    return mod_str


def get_extended_modifications(mod_list: List[str]) -> List[str]:
    """Get an extended set of modifications from a list of modifications.

    Extend bracket types of modifications and strip off underscore, e.g.
      'K(Acetyl)' -> 'K(Acetyl)', 'K[Acetyl]'
      '_[Phospho]' -> '[Phospho]', '_[Phospho]', '_(Phospho)'
    """

    mod_set = set(mod_list)

    for mod in mod_list:
        if mod[1] == "(":
            mod_set.add(f"{mod[0]}[{mod[2:-1]}]")
        elif mod[1] == "[":
            mod_set.add(f"{mod[0]}({mod[2:-1]})")

        if mod.startswith("_"):
            mod_set.add(f"{mod[1:]}")
        elif mod.startswith("("):
            mod_set.add(f"_{mod}")
            mod_set.add(f"[{mod[1:-1]}]")
            mod_set.add(f"_[{mod[1:-1]}]")
        elif mod.startswith("["):
            mod_set.add(f"_{mod}")
            mod_set.add(f"({mod[1:-1]})")
            mod_set.add(f"_({mod[1:-1]})")
    return sorted(list(mod_set))


def get_column_mapping_for_df(column_mapping: dict, df: pd.DataFrame) -> Dict[str, str]:
    """Determine the mapping of AlphaBase columns to the columns in the given DataFrame.

    For each AlphaBase column name, check if the corresponding search engine-specific
    name is in the DataFrame columns. If it is, add it to the mapping.
    If the searchengine-specific name is a list, use the first column name in the list.
    """
    mapped_columns = {}
    for col_alphabase, col_other in column_mapping.items():
        if isinstance(col_other, str):
            if col_other in df.columns:
                mapped_columns[col_alphabase] = col_other
        elif isinstance(col_other, (list, tuple)):
            for other_col in col_other:
                if other_col in df.columns:
                    mapped_columns[col_alphabase] = other_col
                    break
                    # TODO: warn if there's more
    return mapped_columns
