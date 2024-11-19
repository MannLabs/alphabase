"""Utility functions for PSM readers."""

from typing import Dict, List, Set, Tuple, Union

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


def translate_other_modification(
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


def get_mod_set(mod_list: List[str]) -> Set[str]:
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
    return mod_set
