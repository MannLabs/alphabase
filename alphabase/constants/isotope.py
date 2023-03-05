import numba
import numpy as np
import typing

from alphabase.constants.element import (
    MAX_ISOTOPE_LEN, EMPTY_DIST,
    CHEM_ISOTOPE_DIST, CHEM_MONO_IDX, CHEM_MONO_MASS,
    truncate_isotope, parse_formula
)

@numba.njit
def abundance_convolution(
    d1:np.ndarray,
    mono1:int,
    d2:np.ndarray,
    mono2:int,
)->typing.Tuple[np.ndarray, int]:
    '''
    If we have two isotope distributions, 
    we can convolute them into one distribution. 
    
    Parameters
    ----------
    d1 : np.ndarray
        isotope distribution to convolute

    mono1 : int
        mono position of d1.

    d2 : np.ndarray
        isotope distribution to convolute

    mono2 : int
        mono position of d2
        
    Returns
    -------
    tuple[np.ndarray,int]
        np.ndarray, convoluted isotope distribution
        int, new mono position.
    '''
    mono_idx = mono1 + mono2
    ret = np.zeros(MAX_ISOTOPE_LEN*2-1)
    for i in range(len(d1)):
        for j in range(len(d2)):
            ret[i+j] += d1[i]*d2[j]

    mono_idx, start, end = truncate_isotope(ret, mono_idx)
    return ret[start:end], mono_idx

@numba.njit
def one_element_dist(
    elem: str,
    n: int,
    chem_isotope_dist: numba.typed.Dict,
    chem_mono_idx: numba.typed.Dict,
)->typing.Tuple[np.ndarray, int]:
    '''
    Calculate the isotope distribution for 
    an element and its numbers.
    
    Parameters
    ----------
    elem : str
        element.

    n : int
        element number.

    chem_isotope_dist : numba.typed.Dict
        use `CHEM_ISOTOPE_DIST` as parameter.

    chem_mono_idx : numba.typed.Dict
        use `CHEM_MONO_IDX` as parameter.

    Returns
    -------
    tuple[np.ndarray, int]
        np.ndarray, isotope distribution of the element.
        int, mono position in the distribution
    '''
    if n == 0: return EMPTY_DIST.copy(), 0
    elif n == 1: return chem_isotope_dist[elem], chem_mono_idx[elem]
    tmp_dist, mono_idx = one_element_dist(elem, n//2, chem_isotope_dist, chem_mono_idx)
    tmp_dist, mono_idx = abundance_convolution(tmp_dist, mono_idx, tmp_dist, mono_idx)
    if n%2 == 0:
        return tmp_dist, mono_idx
    else:
        return abundance_convolution(tmp_dist, mono_idx, chem_isotope_dist[elem], chem_mono_idx[elem])

def formula_dist(
    formula: typing.Union[list, str]
)->typing.Tuple[np.ndarray, int]:
    '''
    Generate the isotope distribution and the mono index for 
    a given formula (as a list, e.g. `[('H', 2), ('C', 2), ('O', 1)]`).

    Parameters
    ----------
    formula : Union[list, str]
        chemical formula, could be str or list.
        If str: "H(1)N(2)O(3)".
        If list: "[('H',1),('H',2),('H',3)]".
            
    Returns
    -------
    tuple[np.ndarray,int]
        np.ndarray, isotope distribution
        int, mono position
    '''
    if isinstance(formula, str):
        formula = parse_formula(formula)
    calc_dist = EMPTY_DIST.copy()
    mono_idx = 0
    for elem, n in formula:
        _dist, _mono = one_element_dist(elem, n, CHEM_ISOTOPE_DIST, CHEM_MONO_IDX)
        calc_dist, mono_idx = abundance_convolution(calc_dist, mono_idx, _dist, _mono)
    return calc_dist, mono_idx

def _calc_one_elem_cum_dist(
    element_cum_dist:np.ndarray, 
    element_cum_mono:np.ndarray
):
    """Pre-build isotope abundance distribution for an element for fast calculation.
    Internel function.
    
    Added information inplace into element_cum_dist and element_cum_mono

    Parameters
    ----------
    element_cum_dist : np.ndarray
        Cumulated element abundance distribution

    element_cum_mono : np.ndarray
        Cumulated element mono position in the distribution
    """
    for n in range(2, len(element_cum_dist)):
        (
            element_cum_dist[n], 
            element_cum_mono[n]
        ) = abundance_convolution(
            element_cum_dist[n-1],
            element_cum_mono[n-1],
            element_cum_dist[1],
            element_cum_mono[1]
        )

class IsotopeDistribution:
    def __init__(self, 
        max_elem_num_dict:dict = {
            'C': 2000,
            'H': 5000,
            'N': 1000,
            'O': 1000,
            'S': 200,
            'P': 200,
        }
    ):
        """Faster calculation of isotope abundance distribution by pre-building
        isotope distribution tables for most common elements.

        We have considered large enough number of elements for shotgun proteomics.
        We can increase `max_elem_num_dict` to support larger peptide or top-down 
        in the future. However, current `MAX_ISOTOPE_LEN` is not suitable for top-down,
        it must be extended to a larger number (100?).
        Note that non-standard amino acids have 1000000 C elements in AlphaBase,
        We clip 1000000 C to the maximal number of C in `max_elem_num_dict`.
        As they have very large masses thus impossible to identify,
        their isotope distributions do not matter.

        Parameters
        ----------
        max_elem_num_dict : dict, optional
            Define the maximal number of the elements. 
            Defaults to { 'C': 2000, 'H': 5000, 'N': 1000, 'O': 1000, 'S': 200, 'P': 200, } 
        
        Attributes
        ----------
        element_to_cum_dist_dict : dict 
            {element: cumulated isotope distribution array},
            and the cumulated isotope distribution array is a 2-D float np.ndarray with 
            shape (element_max_number, MAX_ISOTOPE_LEN).

        element_to_cum_mono_idx : dict
            {element: mono position array of cumulated isotope distribution},
            and mono position array is a 1-D int np.ndarray.
        """
        self.element_to_cum_dist_dict = {}
        self.element_to_cum_mono_idx = {}
        for elem, n in max_elem_num_dict.items():
            if n < 2: n = 2
            self.element_to_cum_dist_dict[elem] = np.zeros((n, MAX_ISOTOPE_LEN))
            self.element_to_cum_mono_idx[elem] = -np.ones(n,dtype=np.int64)
            self.element_to_cum_dist_dict[elem][0,:] = EMPTY_DIST
            self.element_to_cum_mono_idx[elem][0] = 0
            self.element_to_cum_dist_dict[elem][1,:] = CHEM_ISOTOPE_DIST[elem]
            self.element_to_cum_mono_idx[elem][1] = CHEM_MONO_IDX[elem]
            _calc_one_elem_cum_dist(
                self.element_to_cum_dist_dict[elem],
                self.element_to_cum_mono_idx[elem]
            )

    def calc_formula_distribution(self,
        formula: typing.List[typing.Tuple[str,int]],
    )->typing.Tuple[np.ndarray, int]:
        """Calculate isotope abundance distribution for a given formula

        Parameters
        ----------
        formula : List[tuple(str,int)]
            chemical formula: "[('H',1),('C',2),('O',3)]".

        Returns
        -------
        tuple[np.ndarray, int]
            np.ndarray, isotope abundance distribution
            int, monoisotope position in the distribution array
    
        Examples
        --------
        >>> from alphabase.constants import IsotopeDistribution, parse_formula
        >>> iso = IsotopeDistribution()
        >>> formula = 'C(100)H(100)O(10)Na(1)Fe(1)'
        >>> formula = parse_formula(formula)
        >>> dist, mono = iso.calc_formula_distribution(formula)
        >>> dist
        array([1.92320044e-02, 2.10952666e-02, 3.13753566e-01, 3.42663681e-01,
                1.95962632e-01, 7.69157517e-02, 2.31993814e-02, 5.71948249e-03,
                1.19790438e-03, 2.18815385e-04])
        >>> # Fe's mono position is 2 Da larger than its smallest mass, 
        >>> # so the mono position of this formula shifts by +2 (Da).
        >>> mono 
        2

        >>> formula = 'C(100)H(100)O(10)13C(1)Na(1)' 
        >>> formula = parse_formula(formula)
        >>> dist, mono = iso.calc_formula_distribution(formula)
        >>> dist
        array([3.29033438e-03, 3.29352217e-01, 3.59329960e-01, 2.01524592e-01,
                7.71395498e-02, 2.26229845e-02, 5.41229894e-03, 1.09842389e-03,
                1.94206388e-04, 3.04911585e-05])
        >>> # 13C's mono position is +1 Da shifted
        >>> mono
        1

        >>> formula = 'C(100)H(100)O(10)Na(1)' 
        >>> formula = parse_formula(formula)
        >>> dist, mono = iso.calc_formula_distribution(formula)
        >>> dist
        array([3.29033438e-01, 3.60911319e-01, 2.02775462e-01, 7.76884706e-02,
                2.27963906e-02, 5.45578135e-03, 1.10754072e-03, 1.95857410e-04,
                3.07552058e-05, 4.35047710e-06])
        >>> # mono position is normal (=0) for regular formulas
        >>> mono
        0
            
        """
        mono = 0
        dist = EMPTY_DIST.copy()
        for elem, n in formula:
            if elem in self.element_to_cum_dist_dict:
                if n >= len(self.element_to_cum_mono_idx[elem]):
                    n = len(self.element_to_cum_mono_idx[elem])-1
                dist, mono = abundance_convolution(
                    dist, mono,
                    self.element_to_cum_dist_dict[elem][n],
                    self.element_to_cum_mono_idx[elem][n],
                )
            else:
                dist, mono = abundance_convolution(
                    dist, mono, *one_element_dist(
                        elem,n,CHEM_ISOTOPE_DIST, CHEM_MONO_IDX
                    )
                )
        return dist, mono

