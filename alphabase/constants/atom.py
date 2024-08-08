import os
import re
import typing
from collections import defaultdict

import numba
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from alphabase.constants._const import CONST_FILE_FOLDER, common_const_dict
from alphabase.yaml_utils import load_yaml

MASS_PROTON: float = common_const_dict["MASS_PROTON"]
MASS_ISOTOPE: float = common_const_dict["MASS_ISOTOPE"]

MAX_ISOTOPE_LEN: int = common_const_dict["MAX_ISOTOPE_LEN"]
EMPTY_DIST: np.ndarray = np.zeros(MAX_ISOTOPE_LEN)
EMPTY_DIST[0] = 1


@numba.njit
def truncate_isotope(isotopes: np.ndarray, mono_idx: int) -> tuple:
    """
    For a given isotope distribution (intensity patterns),
    this function truncates the distribution by top
    `MAX_ISOTOPE_LEN` neighbors those contain the monoisotopic
    peak pointed by `mono_idx`.

    Parameters
    ----------
    isotopes : np.ndarray

        Isotope patterns with size > `MAX_ISOTOPE_LEN`.

    mono_idx : int

        Monoisotopic peak position (index) in the isotope patterns

    Returns
    -------
    int

        the new position of `mono_idx`

    int

        the start position of the truncated isotopes

    int

        the end position of the truncated isotopes
    """
    trunc_start = mono_idx - 1
    trunc_end = mono_idx + 1
    while (
        trunc_start >= 0
        and trunc_end < len(isotopes)
        and (trunc_end - trunc_start - 1) < MAX_ISOTOPE_LEN
    ):
        if isotopes[trunc_end] >= isotopes[trunc_start]:
            trunc_end += 1
        else:
            trunc_start -= 1
    if trunc_end - trunc_start - 1 < MAX_ISOTOPE_LEN:
        if trunc_start == -1:
            trunc_end = MAX_ISOTOPE_LEN
        elif trunc_end == len(isotopes):
            trunc_start = len(isotopes) - MAX_ISOTOPE_LEN - 1
    return mono_idx - trunc_start - 1, trunc_start + 1, trunc_end


#: chemical element information in dict defined by `nist_element.yaml`
CHEM_INFO_DICT = {}

#: {element: mass}
CHEM_MONO_MASS = {}

#: {element: np.ndarray of abundance distribution}
CHEM_ISOTOPE_DIST: numba.typed.Dict = numba.typed.Dict.empty(
    key_type=numba.types.unicode_type, value_type=numba.types.float64[:]
)

#: {element: int (mono position)}
CHEM_MONO_IDX: numba.typed.Dict = numba.typed.Dict.empty(
    key_type=numba.types.unicode_type, value_type=numba.types.int64
)

MASS_H: int = None
MASS_C: int = None
MASS_O: int = None
MASS_N: int = None
MASS_H2O: int = None  # raise errors if the value is not reset
MASS_NH3: int = None


def update_atom_infos(new_atom_info: typing.Dict):
    """
    Args:
        atom_dict (Dict): Example, replacing N with 15N
          {"N":
            {"abundance": [0.01,0.99]},
            {"mass": [14.00307400443, 15.00010889888]},
          }
    """
    for atom, info in new_atom_info.items():
        CHEM_INFO_DICT[atom] = info

    reset_elements()


def reset_elements():
    global MASS_C, MASS_H, MASS_O, MASS_N
    global MASS_H2O, MASS_NH3

    for elem, items in CHEM_INFO_DICT.items():
        isotopes = np.array(items["abundance"])
        masses = np.array(items["mass"])
        _sort_idx = np.argsort(masses)
        masses = masses[_sort_idx]
        isotopes = isotopes[_sort_idx]
        _mass_pos = np.round(masses).astype(int)
        _mass_pos = _mass_pos - _mass_pos[0]
        if _mass_pos[-1] - _mass_pos[0] + 1 <= MAX_ISOTOPE_LEN:
            _isos = np.zeros(MAX_ISOTOPE_LEN)
            _isos[_mass_pos] = isotopes
            _masses = np.zeros(MAX_ISOTOPE_LEN)
            _masses[_mass_pos] = masses
            mono_idx = np.argmax(_isos)

            CHEM_MONO_MASS[elem] = _masses[mono_idx]
            CHEM_ISOTOPE_DIST[elem] = _isos
            CHEM_MONO_IDX[elem] = mono_idx
        else:
            _isos = np.zeros(_mass_pos[-1] - _mass_pos[0] + 1)
            _isos[_mass_pos] = isotopes
            _masses = np.zeros(_mass_pos[-1] - _mass_pos[0] + 1)
            _masses[_mass_pos] = masses
            mono_idx = np.argmax(_isos)
            CHEM_MONO_MASS[elem] = _masses[mono_idx]

            _mono_idx, start, end = truncate_isotope(_isos, mono_idx)

            CHEM_ISOTOPE_DIST[elem] = _isos[start:end]
            CHEM_MONO_IDX[elem] = _mono_idx

    MASS_C = CHEM_MONO_MASS["C"]
    MASS_H = CHEM_MONO_MASS["H"]
    MASS_N = CHEM_MONO_MASS["N"]
    MASS_O = CHEM_MONO_MASS["O"]
    MASS_H2O = CHEM_MONO_MASS["H"] * 2 + CHEM_MONO_MASS["O"]
    MASS_NH3 = CHEM_MONO_MASS["H"] * 3 + CHEM_MONO_MASS["N"]


def load_elem_yaml(yaml_file: str):
    """Load built-in or user-defined element yaml file. Default yaml is:
    os.path.join(_base_dir, 'nist_element.yaml')
    """
    global CHEM_INFO_DICT
    global CHEM_MONO_MASS
    global CHEM_ISOTOPE_DIST
    global CHEM_MONO_IDX

    CHEM_INFO_DICT = load_yaml(yaml_file)

    CHEM_MONO_MASS = {}
    CHEM_ISOTOPE_DIST = numba.typed.Dict.empty(
        key_type=numba.types.unicode_type, value_type=numba.types.float64[:]
    )

    CHEM_MONO_IDX = numba.typed.Dict.empty(
        key_type=numba.types.unicode_type, value_type=numba.types.int64
    )

    reset_elements()


load_elem_yaml(os.path.join(CONST_FILE_FOLDER, "nist_element.yaml"))


def parse_formula(formula: str) -> list:
    """
    Given a formula (str, e.g. `H(1)C(2)O(3)`),
    it generates `[('H', 2), ('C', 2), ('O', 1)]`
    """
    if not formula:
        return []
    items = [item.split("(") for item in formula.strip(")").split(")")]
    return [(elem, int(n)) for elem, n in items]


def calc_mass_from_formula(formula: str):
    """
    Calculates the mass of the formula`

    Parameters
    ----------
    formula : str
        e.g. `H(1)C(2)O(3)`

    Returns
    -------
    float
        mass of the formula
    """
    return np.sum([CHEM_MONO_MASS[elem] * n for elem, n in parse_formula(formula)])


class ChemicalCompositonFormula:
    """
    Initialize the ChemicalCompositonFormula with a given formula.

    Parameters
    ----------
    formula : str
        The chemical formula as a string.

    Returns
    -------
    None
    """

    def __init__(self, formula=None):
        self.elements = (
            defaultdict(int) if formula is None else self._parse_formula(formula)
        )

    @classmethod
    def from_smiles(cls, smiles: str) -> "ChemicalCompositonFormula":
        """
        Create a ChemicalCompositonFormula instance from a SMILES string.

        Parameters
        ----------
        smiles : str
            The SMILES representation of the molecule.

        Returns
        -------
        ChemicalCompositonFormula
            An instance of the class based on the SMILES string.

        Raises
        ------
        ValueError
            If the SMILES string is invalid and can't be converted to an RDKit molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid RDKit molecule: {smiles}")
        formula = rdMolDescriptors.CalcMolFormula(
            mol, separateIsotopes=True, abbreviateHIsotopes=False
        )
        formula = formula.replace("[1H]", "H")
        return cls._from_rdkit_formula(formula)

    @classmethod
    def _from_rdkit_formula(cls, formula: str) -> "ChemicalCompositonFormula":
        """
        Create a ChemicalCompositonFormula instance from an RDKit formula.

        Parameters
        ----------
        formula : str
            The chemical formula as generated by RDKit.

        Returns
        -------
        ChemicalCompositonFormula
            An instance of the class based on the RDKit formula.
        """
        instance = cls.__new__(cls)
        instance.elements = instance._parse_rdkit_formula(formula)
        return instance

    def _parse_formula(self, formula) -> dict:
        """
        Parse a chemical formula string into a dictionary of elements and their counts.

        Parameters
        ----------
        formula : str
            The chemical formula to parse.

        Returns
        -------
        dict
            A dictionary with elements as keys and their counts as values.
        """
        # Expected pattern: (\d+)?: optional isotope number, ([A-Z][a-z]*): element symbol, (?:\(([-]?\d+)\))?: optional count in parentheses
        # Example: 13C(2)H(3)O(-1) -> [('13', 'C', '2'), ('', 'H', '3'), ('', 'O', '-1')]
        pattern = r"(\d+)?([A-Z][a-z]*)(?:\(([-]?\d+)\))?"
        matches = re.findall(pattern, formula)
        element_counts = defaultdict(int)

        for isotope, element, count in matches:
            if isotope:
                element = f"{isotope}{element}"
            count = int(count) if count else 1
            element_counts[element] += count

        self._validate_atoms(element_counts)
        return element_counts

    def _parse_rdkit_formula(self, formula: str) -> dict:
        """
        Parse an RDKit-generated formula string into a dictionary of elements and their counts.

        Parameters
        ----------
        formula : str
            The RDKit-generated chemical formula to parse.

        Returns
        -------
        dict
            A dictionary with elements as keys and their counts as values.
        """
        # Expected pattern: (\[(\d+)([A-Z][a-z]*)\]|([A-Z][a-z]*)): isotope in square brackets or element symbol, followed by (\d*): optional count
        # Example: [13C]C2H5OH -> [('[13C]', '13', 'C', '', ''), ('C', '', '', 'C', '2'), ('H', '', '', 'H', '5'), ('O', '', '', 'O', ''), ('H', '', '', 'H', '')]
        pattern = r"(\[(\d+)([A-Z][a-z]*)\]|([A-Z][a-z]*))(\d*)"
        matches = re.findall(pattern, formula)
        element_counts = defaultdict(int)

        for match in matches:
            count = int(match[4]) if match[4] else 1
            if match[1]:  # noqa: SIM108
                # Isotope, see 0th element in the example above
                element = f"{match[1]}{match[2]}"
            else:
                # Regular element, see the rest in the example above
                element = match[3]
            element_counts[element] += count

        self._validate_atoms(element_counts)
        return element_counts

    def _validate_atoms(self, element_counts):
        """
        Validate the elements in the formula.

        Parameters
        ----------
        element_counts : dict
            The elements and their counts in the formula.

        Raises
        ------
        ValueError
            If the formula contains an unknown element.
        """
        for element in element_counts:
            if element not in CHEM_MONO_MASS:
                raise ValueError(f"Unknown element: {element}")

    def __str__(self):
        """
        Return a string representation of the chemical formula.

        Returns
        -------
        str
            The chemical formula as a string.
        """
        return "".join(
            f"{element}({count})"
            for element, count in sorted(self.elements.items())
            if count != 0
        )

    def __repr__(self):
        """
        Return a string representation of the ChemicalCompositonFormula instance.

        Returns
        -------
        str
            A string representation of the instance.
        """
        return f"ChemicalCompositonFormula('{self.__str__()}')"

    def __add__(self, other):
        """
        Add two ChemicalCompositonFormula instances.

        Parameters
        ----------
        other : ChemicalCompositonFormula
            The other instance to add.

        Returns
        -------
        ChemicalCompositonFormula
            A new instance representing the sum of the two formulas.
        """
        result = ChemicalCompositonFormula()
        for element in set(self.elements.keys()) | set(other.elements.keys()):
            result.elements[element] = self.elements[element] + other.elements[element]
        return result

    def __sub__(self, other):
        """
        Subtract one ChemicalCompositonFormula instance from another.

        Parameters
        ----------
        other : ChemicalCompositonFormula
            The instance to subtract.

        Returns
        -------
        ChemicalCompositonFormula
            A new instance representing the difference of the two formulas.
        """
        result = ChemicalCompositonFormula()
        for element in set(self.elements.keys()) | set(other.elements.keys()):
            result.elements[element] = self.elements[element] - other.elements[element]
        return result
