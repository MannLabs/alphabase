import os
from typing import List, Union

import numba
import numpy as np
import pandas as pd

from alphabase.constants._const import CONST_FILE_FOLDER
from alphabase.constants.atom import (
    calc_mass_from_formula,
    parse_formula,
)


class ModificationContext:
    """
    A container class that holds all modification-related data and state.

    This class encapsulates the global modification state, providing a clear
    interface for accessing and modifying modification data.

    Attributes
    ----------
    mod_df : pd.DataFrame
        Main DataFrame storing all modification information
    mod_info_dict : dict
        Dictionary containing all modification information
    mod_chem : dict
        Maps modification names to chemical formulas ('H(1)C(2)O(3)')
    mod_mass : dict
        Maps modification names to masses
    mod_loss_mass : dict
        Maps modifications to neutral loss masses
    mod_composition : dict
        Maps modifications to parsed formula dictionaries
    mod_loss_importance : dict
        Maps modifications to their neutral loss importance
    """

    _MOD_CLASSIFICATION_USER_ADDED = "User-added"

    def __init__(self):
        """Initialize an empty modification context."""
        self.mod_df = pd.DataFrame()
        self.mod_info_dict = {}
        self.mod_chem = {}
        self.mod_mass = {}
        self.mod_loss_mass = {}
        self.mod_composition = {}
        self.mod_loss_importance = {}

    def update_all_by_mod_df(self):
        """
        Update all dictionaries based on the current state of mod_df.

        As DataFrame is more convenient in data operation,
        we process mod_df and then update all dictionaries.
        """
        self.mod_info_dict.clear()
        self.mod_info_dict.update(self.mod_df.to_dict(orient="index"))
        self.mod_chem.clear()
        self.mod_chem.update(self.mod_df["composition"].to_dict())
        self.mod_mass.clear()
        self.mod_mass.update(self.mod_df["mass"].to_dict())
        self.mod_loss_mass.clear()
        self.mod_loss_mass.update(self.mod_df["modloss"].to_dict())
        self.mod_loss_importance.clear()
        self.mod_loss_importance.update(self.mod_df["modloss_importance"].to_dict())

        self.mod_composition.clear()
        for mod, chem in self.mod_chem.items():
            self.mod_composition[mod] = dict(parse_formula(chem))

    def add_modifications_for_lower_case_aa(self):
        """Add modifications for lower-case AAs for advanced usages"""
        lower_case_df = self.mod_df.copy()

        def _mod_lower_case(modname):
            modname, site = modname.split("@")
            if len(site) == 1:
                return modname + "@" + site.lower()
            elif "^" in site:
                site = site[0].lower() + site[1:]
                return modname + "@" + site
            else:
                return ""

        lower_case_df["mod_name"] = lower_case_df["mod_name"].apply(_mod_lower_case)
        lower_case_df = lower_case_df[lower_case_df["mod_name"] != ""]
        lower_case_df.set_index("mod_name", drop=False, inplace=True)
        lower_case_df["lower_case_AA"] = True
        self.mod_df["lower_case_AA"] = False
        self.mod_df = pd.concat([self.mod_df, lower_case_df])
        self.update_all_by_mod_df()

    def keep_modloss_by_importance(self, modloss_importance_level: float = 1.0):
        """Filter modloss values based on importance level"""
        self.mod_df["modloss"] = self.mod_df["modloss_original"]
        self.mod_df.loc[
            self.mod_df.modloss_importance < modloss_importance_level, "modloss"
        ] = 0
        self.mod_loss_mass.clear()
        self.mod_loss_mass.update(self.mod_df["modloss"].to_dict())

    def load_mod_df(
        self,
        tsv: str = os.path.join(CONST_FILE_FOLDER, "modification.tsv"),
        *,
        modloss_importance_level=1,
    ):
        """
        Load modifications from a TSV file.

        Parameters
        ----------
        tsv : str
            Path to the TSV file
        modloss_importance_level : float
            Importance level threshold for modloss filtering
        """
        self.mod_df = pd.read_table(tsv, keep_default_na=False)

        if any(mask := self.mod_df["mod_name"].str.contains(" ", regex=False)):
            raise ValueError(
                f"Modification names must not contain spaces: {self.mod_df[mask]['mod_name'].values}"
            )

        self.mod_df.drop_duplicates("mod_name", inplace=True)
        self.mod_df.fillna("", inplace=True)
        self.mod_df["unimod_id"] = self.mod_df["unimod_id"].astype(np.int32)
        self.mod_df.set_index("mod_name", drop=False, inplace=True)
        self.mod_df["mass"] = self.mod_df["composition"].apply(calc_mass_from_formula)
        self.mod_df["modloss_original"] = self.mod_df["modloss_composition"].apply(
            calc_mass_from_formula
        )
        self.mod_df["modloss"] = self.mod_df["modloss_original"]
        self.keep_modloss_by_importance(modloss_importance_level)
        self.update_all_by_mod_df()

    def _check_mass_sanity(
        self,
        mod_name: str,
        composition: str,
        smiles: str,
    ):
        """
        Check if the mass of the modification is consistent with the formula.

        Parameters
        ----------
        mod_name : str
            Modification name (e.g. Mod@S)
        composition : str
            Composition formula (e.g. "H(4)O(2)"), used to calculate the mass
        smiles : str
            SMILES string of the modification, used for comparison

        Raises
        ------
        ValueError
            If the mass of the modification is inconsistent with the formula
        """
        if not smiles or mod_name not in self.mod_mass:
            return
        composition_mass = calc_mass_from_formula(composition)
        if not np.allclose(composition_mass, self.mod_mass[mod_name], atol=1e-5):
            raise ValueError(
                f"Modification mass of {mod_name} is inconsistent with the composition formula: {composition}, "
                f"df version {self.mod_df.loc[mod_name,['composition']]}"
                f" calculated_mass={composition_mass}, mod_mass={self.mod_mass[mod_name]}"
            )

    def _add_a_new_modification(
        self,
        mod_name: str,
        composition: str,
        modloss_composition: str = "",
        smiles: str = "",
    ):
        """
        Add a new modification into mod_df or update SMILES if modification already exists.

        Parameters
        ----------
        mod_name : str
            Modification name (e.g. Mod@S)
        composition : str
            Composition formula (e.g. "H(4)O(2)")
        modloss_composition : str
            Composition formula of the modification loss (e.g. "H(2)O(1)")
        smiles : str
            SMILES string of the modification
        """
        self._check_mass_sanity(mod_name, composition, smiles)

        if mod_name in self.mod_df.index:
            # If the modification already exists, only update the SMILES
            self.mod_df.loc[mod_name, "smiles"] = smiles
            return

        self.mod_df.loc[
            mod_name,
            [
                "mod_name",
                "composition",
                "modloss_composition",
                "classification",
                "unimod_id",
                "smiles",
            ],
        ] = [
            mod_name,
            composition,
            modloss_composition,
            self._MOD_CLASSIFICATION_USER_ADDED,
            0,
            smiles,
        ]
        composition_mass = calc_mass_from_formula(composition)
        modloss_mass = calc_mass_from_formula(modloss_composition)
        self.mod_df.loc[mod_name, ["mass", "modloss"]] = (
            composition_mass,
            modloss_mass,
        )
        if self.mod_df.loc[mod_name, "modloss"] > 0:
            self.mod_df.loc[mod_name, "modloss_importance"] = 1e10
        self.mod_df.fillna(0, inplace=True)

    def add_new_modifications(self, new_mods: Union[list, dict]):
        """
        Add new modifications to the modification context.

        Parameters
        ----------
        new_mods : list or dict
            List of tuples example:
            ```
            [(
                mod@site:str (e.g. Mod@S),
                composition:str (e.g. "H(4)O(2)"),
                [optional] modloss composition:str (e.g. "H(2)O(1)"),
            ), ...]
            ```,
            Dict example:
            ```
            {
                "mod@site": {
                    "composition":"H(4)O(2)",
                    "modloss_composition":"H(2)O(1)"
                }, ...
            }
            ```
        """
        if isinstance(new_mods, list):
            for items in new_mods:
                self._add_a_new_modification(*items)
        else:
            for mod_name, mod_info in new_mods.items():
                self._add_a_new_modification(mod_name, **mod_info)
        self.update_all_by_mod_df()

    def has_custom_mods(self):
        """
        Returns whether mod_df has user-defined modifications or not.

        Returns
        -------
        bool
            True if user-defined modifications exist, False otherwise
        """
        return (
            len(
                self.mod_df[
                    self.mod_df["classification"] == self._MOD_CLASSIFICATION_USER_ADDED
                ]
            )
            > 0
        )

    def get_custom_mods(self):
        """
        Returns a dictionary of user-defined modifications that can be serialized.

        Returns
        -------
        dict
            Dictionary with modification names as keys and modification details as values
        """
        if not self.has_custom_mods():
            return {}

        custom_mods = self.mod_df[
            self.mod_df["classification"] == self._MOD_CLASSIFICATION_USER_ADDED
        ]
        result = {}

        for mod_name, row in custom_mods.iterrows():
            result[mod_name] = {
                "composition": row["composition"],
                "modloss_composition": row["modloss_composition"],
                "smiles": row["smiles"],
            }

        return result

    def init_custom_mods(self, custom_mods_dict):
        """
        Initialize custom modifications from a dictionary.

        Parameters
        ----------
        custom_mods_dict : dict
            Dictionary of custom modifications as returned by get_custom_mods()
        """
        if not custom_mods_dict:
            return

        for mod_name, mod_info in custom_mods_dict.items():
            self._add_a_new_modification(
                mod_name=mod_name,
                composition=mod_info["composition"],
                modloss_composition=mod_info["modloss_composition"],
                smiles=mod_info["smiles"],
            )

        # Update all dictionaries after adding modifications
        self.update_all_by_mod_df()

    def calc_modification_mass(
        self, nAA: int, mod_names: List[str], mod_sites: List[int]
    ) -> np.ndarray:
        """
        Calculate modification masses for the given peptide length and modified sites.

        Parameters
        ----------
        nAA : int
            Peptide length
        mod_names : list
            List[str]. Modification name list
        mod_sites : list
            List[int]. Modification site list corresponding to `mod_names`.
            * `site=0` refers to an N-term modification
            * `site=-1` refers to a C-term modification
            * `1<=site<=peplen` refers to a normal modification

        Returns
        -------
        np.ndarray
            1-D array with length=`nAA`.
            Masses of modifications through the peptide,
            `0` if sites has no modifications
        """
        masses = np.zeros(nAA)
        for site, mod in zip(mod_sites, mod_names):
            if site == 0 or site == -1:
                masses[site] += self.mod_mass[mod]
            else:
                masses[site - 1] += self.mod_mass[mod]
        return masses

    def calc_mod_masses_for_same_len_seqs(
        self, nAA: int, mod_names_list: List[List[str]], mod_sites_list: List[List[int]]
    ) -> np.ndarray:
        """
        Calculate modification masses for multiple peptides with the same length.

        Parameters
        ----------
        nAA : int
            Peptide length
        mod_names_list : List[List[str]]
            List (pep_count) of modification list (n_mod on each peptide)
        mod_sites_list : List[List[int]]
            List of modification site list corresponding to `mod_names_list`.
            * `site=0` refers to an N-term modification
            * `site=-1` refers to a C-term modification
            * `1<=site<=peplen` refers to a normal modification

        Returns
        -------
        np.ndarray
            2-D array with shape=`(nAA, pep_count or len(mod_names_list)))`.
            Masses of modifications through all the peptides,
            `0` if sites without modifications.
        """
        masses = np.zeros((len(mod_names_list), nAA))
        for i, (mod_names, mod_sites) in enumerate(zip(mod_names_list, mod_sites_list)):
            for mod, site in zip(mod_names, mod_sites):
                if site == 0 or site == -1:
                    masses[i, site] += self.mod_mass[mod]
                else:
                    masses[i, site - 1] += self.mod_mass[mod]
        return masses

    def calc_modification_mass_sum(self, mod_names: List[str]) -> float:
        """
        Calculate summed mass of the given modifications.

        Parameters
        ----------
        mod_names : List[str]
            Modification name list

        Returns
        -------
        float
            Total mass
        """
        return np.sum([self.mod_mass[mod] for mod in mod_names])

    def calc_modloss_mass_with_importance(
        self,
        nAA: int,
        mod_names: List,
        mod_sites: List,
        for_nterm_frag: bool,
    ) -> np.ndarray:
        """
        Calculate modification loss masses with importance-based selection.

        Parameters
        ----------
        nAA : int
            Peptide length
        mod_names : List[str]
            Modification name list
        mod_sites : List[int]
            Modification site list
        for_nterm_frag : bool
            If True, loss will be on N-term fragments (mainly b ions)
            If False, loss will be on C-term fragments (mainly y ions)

        Returns
        -------
        np.ndarray
            mod_loss masses
        """
        if not mod_names:
            return np.zeros(nAA - 1)
        mod_losses = np.zeros(nAA + 2)
        mod_losses[mod_sites] = [self.mod_loss_mass[mod] for mod in mod_names]
        _loss_importance = np.zeros(nAA + 2)
        _loss_importance[mod_sites] = [
            self.mod_loss_importance.get(mod, 0) for mod in mod_names
        ]

        # Will not consider the modloss if the corresponding modloss_importance is 0
        mod_losses[_loss_importance == 0] = 0

        if for_nterm_frag:
            return _calc_modloss_with_importance(mod_losses, _loss_importance)[1:-2]
        else:
            return _calc_modloss_with_importance(
                mod_losses[::-1], _loss_importance[::-1]
            )[-3:0:-1]

    def calc_modloss_mass(
        self,
        nAA: int,
        mod_names: List,
        mod_sites: List,
        for_nterm_frag: bool,
    ) -> np.ndarray:
        """
        Calculate modification loss masses based on proximity.

        Parameters
        ----------
        nAA : int
            Peptide length
        mod_names : List[str]
            Modification name list
        mod_sites : List[int]
            Modification site list
        for_nterm_frag : bool
            If True, loss will be on N-term fragments (mainly b ions)
            If False, loss will be on C-term fragments (mainly y ions)

        Returns
        -------
        np.ndarray
            mod_loss masses
        """
        if len(mod_names) == 0:
            return np.zeros(nAA - 1)
        mod_losses = np.zeros(nAA + 2)
        mod_losses[mod_sites] = [self.mod_loss_mass[mod] for mod in mod_names]

        if for_nterm_frag:
            return _calc_modloss(mod_losses)[1:-2]
        else:
            return _calc_modloss(mod_losses[::-1])[-3:0:-1]


# Create the singleton instance
_global_context = ModificationContext()
_global_context.load_mod_df()

# For backward compatibility, expose the context's attributes as global variables
MOD_DF = _global_context.mod_df
MOD_INFO_DICT = _global_context.mod_info_dict
MOD_CHEM = _global_context.mod_chem
MOD_MASS = _global_context.mod_mass
MOD_LOSS_MASS = _global_context.mod_loss_mass
MOD_Composition = _global_context.mod_composition
MOD_LOSS_IMPORTANCE = _global_context.mod_loss_importance

# For backward compatibility, expose context methods as module functions
_MOD_CLASSIFICATION_USER_ADDED = ModificationContext._MOD_CLASSIFICATION_USER_ADDED


def update_all_by_MOD_DF():
    """Update all global dictionaries from MOD_DF."""
    _global_context.update_all_by_mod_df()

    # Update module-level variables to maintain backward compatibility
    global \
        MOD_DF, \
        MOD_INFO_DICT, \
        MOD_CHEM, \
        MOD_MASS, \
        MOD_LOSS_MASS, \
        MOD_Composition, \
        MOD_LOSS_IMPORTANCE
    MOD_DF = _global_context.mod_df
    MOD_INFO_DICT = _global_context.mod_info_dict
    MOD_CHEM = _global_context.mod_chem
    MOD_MASS = _global_context.mod_mass
    MOD_LOSS_MASS = _global_context.mod_loss_mass
    MOD_Composition = _global_context.mod_composition
    MOD_LOSS_IMPORTANCE = _global_context.mod_loss_importance


def add_modifications_for_lower_case_AA():
    """Add modifications for lower-case AAs for advanced usages."""
    _global_context.add_modifications_for_lower_case_aa()
    update_all_by_MOD_DF()


def keep_modloss_by_importance(modloss_importance_level: float = 1.0):
    """Filter modloss values based on importance level."""
    _global_context.keep_modloss_by_importance(modloss_importance_level)
    update_all_by_MOD_DF()


def load_mod_df(
    tsv: str = os.path.join(CONST_FILE_FOLDER, "modification.tsv"),
    *,
    modloss_importance_level=1,
):
    """Load modifications from a TSV file."""
    _global_context.load_mod_df(tsv, modloss_importance_level=modloss_importance_level)
    update_all_by_MOD_DF()


def add_new_modifications(new_mods: Union[list, dict]):
    """Add new modifications to the global context."""
    _global_context.add_new_modifications(new_mods)
    update_all_by_MOD_DF()


def has_custom_mods():
    """Returns whether global MOD_DF has user-defined modifications or not."""
    return _global_context.has_custom_mods()


def get_custom_mods():
    """Returns a dictionary of user-defined modifications that can be serialized."""
    return _global_context.get_custom_mods()


def init_custom_mods(custom_mods_dict):
    """Initialize custom modifications in a child process from a dictionary."""
    _global_context.init_custom_mods(custom_mods_dict)
    update_all_by_MOD_DF()


def calc_modification_mass(
    nAA: int, mod_names: List[str], mod_sites: List[int]
) -> np.ndarray:
    """Calculate modification masses for the given peptide length and modified sites."""
    return _global_context.calc_modification_mass(nAA, mod_names, mod_sites)


def calc_mod_masses_for_same_len_seqs(
    nAA: int, mod_names_list: List[List[str]], mod_sites_list: List[List[int]]
) -> np.ndarray:
    """Calculate modification masses for multiple peptides with the same length."""
    return _global_context.calc_mod_masses_for_same_len_seqs(
        nAA, mod_names_list, mod_sites_list
    )


def calc_modification_mass_sum(mod_names: List[str]) -> float:
    """Calculate summed mass of the given modifications."""
    return _global_context.calc_modification_mass_sum(mod_names)


@numba.jit(nopython=True, nogil=True)
def _calc_modloss_with_importance(
    mod_losses: np.ndarray, _loss_importance: np.ndarray
) -> np.ndarray:
    """
    Calculate modification loss masses with importance-based selection.

    Modification with higher `_loss_importance` has higher priorities.
    For example, `AM(Oxidation@M)S(Phospho@S)...`,
    importance of Phospho@S > importance of Oxidation@M, so the modloss of
    b3 ion will be -98 Da, not -64 Da.

    Parameters
    ----------
    mod_losses : np.ndarray
        Mod loss masses of each AA position
    _loss_importance : np.ndarray
        Mod loss importance of each AA position

    Returns
    -------
    np.ndarray
        New mod_loss masses selected by `_loss_importance`
    """
    prev_importance = _loss_importance[0]
    prev_most = 0
    for i, _curr_imp in enumerate(_loss_importance[1:], 1):
        if _curr_imp > prev_importance:
            prev_most = i
            prev_importance = _curr_imp
        else:
            mod_losses[i] = mod_losses[prev_most]
    return mod_losses


@numba.njit
def _calc_modloss(mod_losses: np.ndarray) -> np.ndarray:
    """
    Calculate modification loss masses (e.g. -98 Da for Phospho@S/T).

    Parameters
    ----------
    mod_losses : np.ndarray
        Mod loss masses of each AA position

    Returns
    -------
    np.ndarray
        New mod_loss masses
    """
    for i, _curr_loss in enumerate(mod_losses[1:], 1):
        if _curr_loss == 0:
            mod_losses[i] = mod_losses[i - 1]
        else:
            mod_losses[i] = _curr_loss
    return mod_losses


def calc_modloss_mass_with_importance(
    nAA: int,
    mod_names: List,
    mod_sites: List,
    for_nterm_frag: bool,
) -> np.ndarray:
    """Calculate modification loss masses with importance-based selection."""
    return _global_context.calc_modloss_mass_with_importance(
        nAA, mod_names, mod_sites, for_nterm_frag
    )


def calc_modloss_mass(
    nAA: int,
    mod_names: List,
    mod_sites: List,
    for_nterm_frag: bool,
) -> np.ndarray:
    """Calculate modification loss masses based on proximity."""
    return _global_context.calc_modloss_mass(nAA, mod_names, mod_sites, for_nterm_frag)


# For test purposes - provide a way to get a new context
def get_new_context() -> ModificationContext:
    """Create and return a new ModificationContext instance."""
    return ModificationContext()
