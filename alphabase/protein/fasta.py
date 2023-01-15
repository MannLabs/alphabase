import regex as re
import numpy as np
import pandas as pd
import numba
import os
import itertools
import copy

from Bio import SeqIO
from typing import Union

from alphabase.yaml_utils import load_yaml
from alphabase.io.hdf import HDF_File
from alphabase.utils import explode_multiple_columns

from alphabase.constants._const import CONST_FILE_FOLDER

from alphabase.spectral_library.base import SpecLibBase

def get_uniprot_gene_name(description:str):
    idx = description.find(' GN=')
    if idx == -1: return ''
    else: idx += 4
    return description[idx:description.find(' ', idx)]

def read_fasta_file(fasta_filename:str=""):
    """
    Read a FASTA file line by line

    Parameters
    ----------
    fasta_filename : str
        fasta.
        
    Yields
    ------
    dict 
        protein information, 
        {protein_id:str, full_name:str, gene_name:str, description:str, sequence:str}
    """
    with open(fasta_filename, "rt") as handle:
        iterator = SeqIO.parse(handle, "fasta")
        while iterator:
            try:
                record = next(iterator)
                parts = record.id.split("|")  # pipe char
                if len(parts) > 1:
                    id = parts[1]
                else:
                    id = record.name
                sequence = str(record.seq)
                entry = {
                    "protein_id": id,
                    "full_name": record.name,
                    "gene_name": get_uniprot_gene_name(record.description),
                    "description": record.description,
                    "sequence": sequence,
                }

                yield entry
            except StopIteration:
                break

def load_all_proteins(fasta_file_list:list):
    protein_dict = {}
    for fasta in fasta_file_list:
        for protein in read_fasta_file(fasta):
            protein_dict[protein['full_name']] = protein
    return protein_dict

def concat_proteins(protein_dict:dict, sep='$')->str:
    """Concatenate all protein sequences into a single sequence, 
    seperated by `sep ($ by default)`.

    Parameters
    ----------
    protein_dict : dict
        protein_dict by read_fasta_file()

    Returns
    -------
    str
        concatenated sequence seperated by `sep`.
    """
    seq_list = ['']
    seq_count = 1
    for key in protein_dict:
        protein_dict[key]['offset'] = seq_count
        seq_list.append(protein_dict[key]['sequence'])
        seq_count += protein_dict[key]['sequence']+1
    seq_list.append('')
    return '$'.join(seq_list)

protease_dict = load_yaml(
    os.path.join(
        CONST_FILE_FOLDER, 
        'protease.yaml'
    )
)
"""
Pre-built protease dict with regular expression.
"""

@numba.njit
def cleave_sequence_with_cut_pos(
    sequence:str,
    cut_pos:np.ndarray,
    n_missed_cleavages:int=2,
    pep_length_min:int=6,
    pep_length_max:int=45,
)->tuple:
    """
    Cleave a sequence with cut postions (cut_pos). 
    Filters to have a minimum and maximum length.

    Parameters
    ----------
    sequence : str
        protein sequence

    cut_pos : np.ndarray
        cut postions determined by a given protease.

    n_missed_cleavages : int
        the number of max missed cleavages.

    pep_length_min : int
        min peptide length.

    pep_length_max :int
        max peptide length.

    Returns
    -------
    tuple:
        List[str]. Cleaved peptide sequences with missed cleavages.

        List[int]. Number of miss cleavage of each peptide.

        List[bool]. If N-term peptide

        List[bool]. If C-term pepetide
    """
    seq_list = []
    miss_list = []
    nterm_list = []
    cterm_list = []
    for i,start_pos in enumerate(cut_pos):
        for n_miss,end_pos in enumerate(
            cut_pos[i+1:i+2+n_missed_cleavages]
        ):
            if end_pos > start_pos + pep_length_max:
                break
            elif end_pos < start_pos + pep_length_min:
                continue
            else:
                seq_list.append(sequence[start_pos:end_pos])
                miss_list.append(n_miss)
                if start_pos == 0:
                    nterm_list.append(True)
                else:
                    nterm_list.append(False)
                if end_pos == len(sequence):
                    cterm_list.append(True)
                else:
                    cterm_list.append(False)
    return seq_list, miss_list, nterm_list, cterm_list

class Digest(object):
    def __init__(self,
        protease:str='trypsin/P',
        max_missed_cleavages:int=2,
        peptide_length_min:int=6,
        peptide_length_max:int=45,
    ):
        """Digest a protein sequence

        Parameters
        ----------
        protease : str, optional
            protease name, could be pre-defined name defined in `protease_dict`
            or a regular expression. By default 'trypsin/P'

        max_missed_cleavages : int, optional
            Max number of misses cleavage sites.
            By default 2

        peptide_length_min : int, optional
            Minimal cleaved peptide length, by default 6
            
        peptide_length_max : int, optional
            Maximal cleaved peptide length, by default 45
        """

        self.n_miss_cleave = max_missed_cleavages
        self.peptide_length_min = peptide_length_min
        self.peptide_length_max = peptide_length_max
        if protease.lower() in protease_dict:
            self.regex_pattern = re.compile(
                protease_dict[protease.lower()]
            )
        else:
            self.regex_pattern = re.compile(
                protease
            )

    def cleave_sequence(self,
        sequence:str,
    )->tuple:
        """
        Cleave a sequence.

        Parameters
        ----------
        sequence : str
            the given (protein) sequence.

        Returns
        -------
        tuple[list]
            list[str]: cleaved peptide sequences with missed cleavages
            list[int]: miss cleavage list
            list[bool]: is protein N-term
            list[bool]: is protein C-term
        """

        cut_pos = [0]
        cut_pos.extend([
            m.start()+1 for m in 
            self.regex_pattern.finditer(sequence)
        ])
        cut_pos.append(len(sequence))
        cut_pos = np.array(cut_pos, dtype=np.int64)

        (
            seq_list, miss_list, nterm_list, cterm_list
        ) = cleave_sequence_with_cut_pos(
            sequence, cut_pos, 
            self.n_miss_cleave,
            self.peptide_length_min,
            self.peptide_length_max,
        )
        # Consider M loss at protein N-term
        if sequence.startswith('M'):
            for seq,miss,cterm in zip(
                seq_list,miss_list,cterm_list
            ):
                if (
                    sequence.startswith(seq) 
                    and len(seq)>self.peptide_length_min
                ):
                    seq_list.append(seq[1:])
                    miss_list.append(miss)
                    nterm_list.append(True)
                    cterm_list.append(cterm)
        return seq_list, miss_list, nterm_list, cterm_list

def get_fix_mods(
    sequence:str,
    fix_mod_aas:str,
    fix_mod_dict:dict
)->tuple:
    """
    Generate fix modifications for the sequence
    """
    mods = []
    mod_sites = []
    for i,aa in enumerate(sequence):
        if aa in fix_mod_aas:
            mod_sites.append(i+1)
            mods.append(fix_mod_dict[aa])
    return ';'.join(mods), ';'.join(str(i) for i in mod_sites)

def get_candidate_sites(
    sequence:str, target_mod_aas:str
)->list:
    """get candidate modification sites

    Parameters
    ----------
    sequence : str
        peptide sequence

    target_mod_aas : str
        AAs that may have modifications

    Returns
    -------
    list
        candiadte mod sites in alphabase format (0: N-term, -1: C-term, 1-n:others)
    """
    candidate_sites = []
    for i,aa in enumerate(sequence):
        if aa in target_mod_aas:
            candidate_sites.append(i+1) #alphabase mod sites
    return candidate_sites

def get_var_mod_sites(
    sequence:str,
    target_mod_aas:str,
    min_var_mod: int,
    max_var_mod: int,
    max_combs: int,
)->list:
    """get all combinations of variable modification sites

    Parameters
    ----------
    sequence : str
        peptide sequence

    target_mod_aas : str
        AAs that may have modifications

    min_var_mod : int
        max number of mods in a sequence

    max_var_mod : int
        max number of mods in a sequence

    max_combs : int
        max number of combinations for a sequence

    Returns
    -------
    list
        list of combinations (tuple) of modification sites 
    """
    candidate_sites = get_candidate_sites(
        sequence, target_mod_aas
    )
    if min_var_mod <= 1 and max_var_mod >= 1:
        mod_sites = [(s,) for s in candidate_sites]
    else:
        mod_sites = []
    for n_var_mod in range(max(2,min_var_mod), max_var_mod+1):
        if len(mod_sites)>=max_combs: break
        mod_sites.extend(
            itertools.islice(
                itertools.combinations(
                    candidate_sites, n_var_mod
                ),
                max_combs-len(mod_sites)
            )
        )
    return mod_sites

def get_var_mods_per_sites_multi_mods_on_aa(
    sequence:str,
    mod_sites:tuple,
    var_mod_dict:dict
)->list:
    """
    Used only when the var mod list contains 
    more than one mods on the same AA, for example:
    Mod1@A, Mod2@A ...
    """
    mods_str_list = ['']
    for i,site in enumerate(mod_sites):
        if len(var_mod_dict[sequence[site-1]]) == 1:
            for i in range(len(mods_str_list)):
                mods_str_list[i] += var_mod_dict[sequence[site-1]][0]+';'
        else:
            _new_list = []
            for mod in var_mod_dict[sequence[site-1]]:
                _lst = copy.deepcopy(mods_str_list)
                for i in range(len(_lst)):
                    _lst[i] += mod+';'
                _new_list.extend(_lst)
            mods_str_list = _new_list
    return [mod[:-1] for mod in mods_str_list]

def get_var_mods_per_sites_single_mod_on_aa(
    sequence:str,
    mod_sites:tuple,
    var_mod_dict:dict
)->list:
    """
    Used when the var mod list contains 
    only one mods on the each AA, for example:
    Mod1@A, Mod2@D ...
    """
    mod_str = ''
    for site in mod_sites:
            mod_str += var_mod_dict[sequence[site-1]]+';'
    return [mod_str[:-1]]

get_var_mods_per_sites = get_var_mods_per_sites_single_mod_on_aa

def get_var_mods(
    sequence:str,
    var_mod_aas:str,
    mod_dict:dict,
    min_var_mod:int,
    max_var_mod:int,
    max_combs:int,
)->tuple:
    """
    Generate all modification combinations and associated sites
    for the sequence.
    """
    mod_sites_list = get_var_mod_sites(
        sequence, var_mod_aas, 
        min_var_mod, max_var_mod, max_combs
    )
    ret_mods = []
    ret_sites_list = []
    for mod_sites in mod_sites_list:
        _mods = get_var_mods_per_sites(
            sequence,mod_sites,mod_dict
        )
        mod_sites_str = ';'.join([str(i) for i in mod_sites])
        ret_mods.extend(_mods)
        ret_sites_list.extend([mod_sites_str]*len(_mods))
    if min_var_mod == 0:
        ret_mods.append('')
        ret_sites_list.append('')
    return ret_mods, ret_sites_list

def parse_term_mod(term_mod_name:str):
    _mod, term = term_mod_name.split('@')
    if '^' in term:
        return tuple(term.split('^'))
    else:
        return '', term

def add_single_peptide_labeling(
    seq:str,
    mods:str,
    mod_sites:str, 
    label_aas:str, 
    label_mod_dict:dict, 
    nterm_label_mod:str, 
    cterm_label_mod:str
):
    add_nterm_label = True if nterm_label_mod else False
    add_cterm_label = True if cterm_label_mod else False
    if mod_sites:
        _sites = mod_sites.split(';')
        if '0' in _sites: add_nterm_label = False
        if '-1' in _sites: add_cterm_label = False
        mod_list = [mods]
        mod_site_list = [mod_sites]
    else:
        mod_list = []
        mod_site_list = []
    if add_nterm_label:
        mod_list.append(nterm_label_mod)
        mod_site_list.append('0')
    if add_cterm_label:
        mod_list.append(cterm_label_mod)
        mod_site_list.append('-1')
    aa_labels, aa_label_sites = get_fix_mods(seq, label_aas, label_mod_dict)
    if aa_labels:
        mod_list.append(aa_labels)
        mod_site_list.append(aa_label_sites)
    return ';'.join(mod_list), ';'.join(mod_site_list)

def parse_labels(labels:list):
    label_aas = ''
    label_mod_dict = {}
    nterm_label_mod = ''
    cterm_label_mod = ''
    for label in labels:
        _, aa = label.split('@')
        if len(aa) == 1:
            label_aas += aa
            label_mod_dict[aa] = label
        elif aa == 'Any N-term':
            nterm_label_mod = label
        elif aa == 'Any C-term':
            cterm_label_mod = label
    return label_aas, label_mod_dict, nterm_label_mod, cterm_label_mod
        
def create_labeling_peptide_df(peptide_df:pd.DataFrame, labels:list):
    if len(peptide_df) == 0: return peptide_df

    df = peptide_df.copy()

    (
        label_aas, label_mod_dict, 
        nterm_label_mod, cterm_label_mod
    ) = parse_labels(labels)

    (
        df['mods'],
        df['mod_sites']
    ) = zip(*df[
        ['sequence','mods','mod_sites']
    ].apply(lambda x:
        add_single_peptide_labeling(
            *x, label_aas, label_mod_dict, 
            nterm_label_mod, cterm_label_mod
        ), axis=1,
    ))

    return df

def protein_idxes_to_names(protein_idxes:str, protein_names:list):
    if len(protein_idxes) == 0: return ''
    proteins = [protein_names[int(i)] for i in protein_idxes.split(';')]
    proteins = [protein for protein in proteins if protein]
    return ';'.join(proteins)

def append_special_modifications(
    df:pd.DataFrame, 
    var_mods:list = ['Phospho@S','Phospho@T','Phospho@Y'], 
    min_mod_num:int=0, max_mod_num:int=1, 
    max_peptidoform_num:int=100,
    cannot_modify_pep_nterm_aa:bool=False,
    cannot_modify_pep_cterm_aa:bool=False,
)->pd.DataFrame:
    """
    Append special (not N/C-term) variable modifications to the 
    exsiting modifications of each sequence in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Precursor dataframe
    
    var_mods : list, optional
        Considered varialbe modification list. 
        Defaults to ['Phospho@S','Phospho@T','Phospho@Y'].

    min_mod_num : int, optional
        Minimal modification number for 
        each sequence of the `var_mods`. Defaults to 0.

    max_mod_num : int, optional
        Maximal modification number for 
        each sequence of the `var_mods`. Defaults to 1.

    max_peptidoform_num : int, optional
        One sequence is only allowed to explode 
        to `max_peptidoform_num` number of modified peptides. Defaults to 100.

    cannot_modify_pep_nterm_aa : bool, optional
        Similar to `cannot_modify_pep_cterm_aa`, by default False

    cannot_modify_pep_cterm_aa : bool, optional
        If the modified AA is at C-term, then the modification cannot modified it.
        For example GlyGly@K, for a peptide `ACDKEFGK`, if GlyGly is at the C-term, 
        trypsin cannot cleave the C-term K, hence there will be no such a modified peptide ACDKEFGK(GlyGly).
        by default False

    Returns
    -------
    pd.DataFrame
        The precursor_df with new modification added.
    """
    if len(var_mods) == 0 or len(df) == 0: 
        return df

    if cannot_modify_pep_nterm_aa:
        df['sequence'] = df['sequence'].apply(
            lambda seq: seq[0].lower()+seq[1:]
        )
    
    if cannot_modify_pep_cterm_aa:
        df['sequence'] = df['sequence'].apply(
            lambda seq: seq[:-1]+seq[-1].lower()
        )

    mod_dict = dict([(mod[-1],mod) for mod in var_mods])
    var_mod_aas = ''.join(mod_dict.keys())
    
    (
        df['mods_app'],
        df['mod_sites_app']
    ) = zip(*df.sequence.apply(get_var_mods,
            var_mod_aas=var_mod_aas, mod_dict=mod_dict, 
            min_var_mod=min_mod_num, max_var_mod=max_mod_num, 
            max_combs=max_peptidoform_num,
        )
    )

    if cannot_modify_pep_nterm_aa:
        df['sequence'] = df['sequence'].apply(
            lambda seq: seq[0].upper()+seq[1:]
        )
    
    if cannot_modify_pep_cterm_aa:
        df['sequence'] = df['sequence'].apply(
            lambda seq: seq[:-1]+seq[-1].upper()
        )
    
    if min_mod_num==0:
        df = df.explode(['mods_app','mod_sites_app'])
        df.fillna('', inplace=True)
    else:
        df.drop(df[df.mods_app.apply(lambda x: len(x)==0)].index, inplace=True)
        df = df.explode(['mods_app','mod_sites_app'])
    df['mods'] = df[['mods','mods_app']].apply(
        lambda x: ';'.join(i for i in x if i), axis=1
    )
    df['mod_sites'] = df[['mod_sites','mod_sites_app']].apply(
        lambda x: ';'.join(i for i in x if i), axis=1
    )
    df.drop(columns=['mods_app', 'mod_sites_app'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

class SpecLibFasta(SpecLibBase):
    """
    This is the main entry of AlphaBase when generating spectral libraries from fasta files
    It includes functionalities to:
    
    - Load protein sequences
    - Digest protein sequences
    - Append decoy peptides
    - Add fixed, variable or labeling modifications to the peptide sequences
    - Add charge states
    - Save libraries into hdf file

    Attributes
    ----------
    max_peptidoform_num : int, 100 by default
        For some modifications such as Phospho, there may be 
        thousands of peptidoforms generated for some peptides, 
        so we use this attribute to control the overall number of 
        peptidoforms of a peptide.
    
    protein_df : pd.DataFrame
        Protein dataframe with columns 'protein_id', 
        'sequence', 'description', 'gene_name', etc.
    """
    def __init__(self,
        charged_frag_types:list = [
            'b_z1','b_z2','y_z1', 'y_z2'
        ],
        *,
        protease:str = 'trypsin',
        max_missed_cleavages:int = 2,
        peptide_length_min:int = 7,
        peptide_length_max:int = 35,
        precursor_charge_min:int = 2,
        precursor_charge_max:int = 4,
        precursor_mz_min:float = 400.0, 
        precursor_mz_max:float = 2000.0,
        var_mods:list = ['Acetyl@Protein N-term','Oxidation@M'],
        min_var_mod_num:int = 0,
        max_var_mod_num:int = 2,
        fix_mods:list = ['Carbamidomethyl@C'],
        labeling_channels:dict = None,
        special_mods:list = [],
        min_special_mod_num:int = 0,
        max_special_mod_num:int = 1,
        special_mods_cannot_modify_pep_n_term:bool=False,
        special_mods_cannot_modify_pep_c_term:bool=False,
        decoy: str = None,
        include_contaminants:bool=False,
        I_to_L:bool=False,
    ):
        """
        Parameters
        ----------
        charged_frag_types : list, optional
            Fragment types with charge, 
            by default [ 'b_z1','b_z2','y_z1', 'y_z2' ]

        protease : str, optional
            Could be pre-defined protease name defined in :data:`protease_dict`,
            or a regular expression. 
            By default 'trypsin'

        max_missed_cleavages : int, optional
            Maximal missed cleavages, by default 2
            
        peptide_length_min : int, optional
            Minimal cleaved peptide length, by default 7

        peptide_length_max : int, optional
            Maximal cleaved peptide length, by default 35

        precursor_charge_min : int, optional
            Minimal precursor charge, by default 2

        precursor_charge_max : int, optional
            Maximal precursor charge, by default 4

        precursor_mz_min : float, optional
            Minimal precursor mz, by default 200.0

        precursor_mz_max : float, optional
            Maximal precursor mz, by default 2000.0

        var_mods : list, optional
            list of variable modifications, 
            by default ['Acetyl@Protein N-term','Oxidation@M']

        max_var_mod_num : int, optional
            Minimal number of variable modifications on a peptide sequence, 
            by default 0

        max_var_mod_num : int, optional
            Maximal number of variable modifications on a peptide sequence, 
            by default 2

        fix_mods : list, optional
            list of fixed modifications, by default ['Carbamidomethyl@C']

        labeling_channels : dict, optional
            Add isotope labeling with different channels, 
            see :meth:`add_peptide_labeling()`. 
            Defaults to None

        special_mods : list, optional
            Modifications with special occurance per peptide.
            It is useful for modificaitons like Phospho which may largely 
            explode the number of candidate modified peptides.
            The number of special_mods per peptide 
            is controlled by `max_append_mod_num`.
            Defaults to [].

        min_special_mod_num : int, optional
            Control the min number of special_mods per peptide, by default 0.

        max_special_mod_num : int, optional
            Control the max number of special_mods per peptide, by default 1.

        special_mods_cannot_modify_pep_c_term : bool, optional
            Some modifications cannot modify the peptide C-term, 
            this will be useful for GlyGly@K as if C-term is di-Glyed, 
            it cannot be cleaved/digested. 
            Defaults to False.

        special_mods_cannot_modify_pep_n_term : bool, optional
            Similar to `special_mods_cannot_modify_pep_c_term`, but at N-term.
            Defaults to False.

        decoy : str, optional
            Decoy type (see :meth:`alphabase.spectral_library.base.append_decoy_sequence()`)

            - `protein_reverse`: Reverse on target protein sequences
            - `pseudo_reverse`: Pseudo-reverse on target peptide sequences
            - `diann`: DiaNN-like decoy
            - None: no decoy

            by default None

        include_contaminants : bool, optional
            If include contaminants.fasta, by default False
        """
        super().__init__(
            charged_frag_types=charged_frag_types,
            precursor_mz_min=precursor_mz_min,
            precursor_mz_max=precursor_mz_max,
            decoy=decoy
        )
        self.protein_df:pd.DataFrame = pd.DataFrame()
        self.I_to_L = I_to_L
        self.include_contaminants = include_contaminants
        self.max_peptidoform_num = 100
        self._digest = Digest(
            protease, max_missed_cleavages,
            peptide_length_min, peptide_length_max
        )
        self.min_precursor_charge = precursor_charge_min
        self.max_precursor_charge = precursor_charge_max

        self.var_mods = var_mods
        self.fix_mods = fix_mods
        self.min_var_mod_num = min_var_mod_num
        self.max_var_mod_num = max_var_mod_num
        self.labeling_channels = labeling_channels
        self.special_mods = special_mods
        self.min_special_mod_num = min_special_mod_num
        self.max_special_mod_num = max_special_mod_num
        self.special_mods_cannot_modify_pep_n_term = special_mods_cannot_modify_pep_n_term
        self.special_mods_cannot_modify_pep_c_term = special_mods_cannot_modify_pep_c_term

        self._parse_fix_and_var_mods()
    
    def _parse_fix_and_var_mods(self):
        self.fix_mod_aas = ''
        self.fix_mod_prot_nterm_dict = {}
        self.fix_mod_prot_cterm_dict = {}
        self.fix_mod_pep_nterm_dict = {}
        self.fix_mod_pep_cterm_dict = {}
        self.fix_mod_dict = {}

        def _set_term_mod(term_mod,
            prot_nterm, prot_cterm, pep_nterm, pep_cterm,
            allow_conflicts
        ):
            def _set_dict(term_dict,site,mod,
                allow_conflicts
            ):
                if allow_conflicts:
                    if site in term_dict:
                        term_dict[site].append(term_mod)
                    else:
                        term_dict[site] = [term_mod]
                else:
                    term_dict[site] = term_mod
            site, term = parse_term_mod(term_mod)
            if term == "Any N-term":
                _set_dict(pep_nterm, site, term_mod, 
                    allow_conflicts
                )
            elif term == 'Protein N-term':
                _set_dict(prot_nterm, site, term_mod, 
                    allow_conflicts
                )
            elif term == 'Any C-term':
                _set_dict(pep_cterm, site, term_mod, 
                    allow_conflicts
                )
            elif term == 'Protein C-term':
                _set_dict(prot_cterm, site, term_mod, 
                    allow_conflicts
                )
        
        for mod in self.fix_mods:
            if mod.find('@')+2 == len(mod):
                self.fix_mod_aas += mod[-1]
                self.fix_mod_dict[mod[-1]] = mod
            else:
                _set_term_mod(
                    mod, 
                    self.fix_mod_prot_nterm_dict,
                    self.fix_mod_prot_cterm_dict,
                    self.fix_mod_pep_nterm_dict,
                    self.fix_mod_pep_cterm_dict,
                    allow_conflicts=False
                )

        self.var_mod_aas = ''
        self.var_mod_prot_nterm_dict = {}
        self.var_mod_prot_cterm_dict = {}
        self.var_mod_pep_nterm_dict = {}
        self.var_mod_pep_cterm_dict = {}
        self.var_mod_dict = {}

        global get_var_mods_per_sites
        if self._check_if_multi_mods_on_aa(self.var_mods):
            for mod in self.var_mods:
                if mod.find('@')+2 == len(mod):
                    if mod[-1] in self.fix_mod_dict: continue
                    self.var_mod_aas += mod[-1]
                    if mod[-1] in self.var_mod_dict:
                        self.var_mod_dict[mod[-1]].append(mod)
                    else:
                        self.var_mod_dict[mod[-1]] = [mod]
            get_var_mods_per_sites = get_var_mods_per_sites_multi_mods_on_aa
        else:
            for mod in self.var_mods:
                if mod.find('@')+2 == len(mod):
                    if mod[-1] in self.fix_mod_dict: continue
                    self.var_mod_aas += mod[-1]
                    self.var_mod_dict[mod[-1]] = mod
            get_var_mods_per_sites = get_var_mods_per_sites_single_mod_on_aa
        
        for mod in self.var_mods:
            if mod.find('@')+2 < len(mod):
                _set_term_mod(
                    mod, 
                    self.var_mod_prot_nterm_dict,
                    self.var_mod_prot_cterm_dict,
                    self.var_mod_pep_nterm_dict,
                    self.var_mod_pep_cterm_dict,
                    allow_conflicts=True
                )

    def _check_if_multi_mods_on_aa(self, var_mods):
        mod_set = set()
        for mod in var_mods:
            if mod.find('@')+2 == len(mod):
                if mod[-1] in mod_set: return True
                mod_set.add(mod[-1])
        return False

    def import_and_process_fasta(self, 
        fasta_files:list,
    ):
        """
        Import and process a fasta file or a list of fasta files.
        It includes 3 steps:

        1. Digest and get peptide sequences, it uses `self.get_peptides_from_...()`
        2. Process the peptides including add modifications, 
        it uses :meth:`process_from_naked_peptide_seqs()`.

        Parameters
        ----------
        fasta_files : list
            A fasta file or a list of fasta files
        """
        if self.include_contaminants:
            fasta_files.append(os.path.join(
                CONST_FILE_FOLDER, 'contaminants.fasta'
            ))
        protein_dict = load_all_proteins(fasta_files)
        self.import_and_process_protein_dict(protein_dict)

    def import_and_process_protein_dict(self, protein_dict:dict):
        """ 
        Import and process the protein_dict.
        The processing step is in :meth:`process_from_naked_peptide_seqs()`.
        ```
        protein_dict = load_all_proteins(fasta_files)
        ```

        Parameters
        ----------
        protein_dict : dict
            Format:
            {
            'prot_id1': {'protein_id': 'prot_id1', 'sequence': string, 'gene_name': string, 'description': string
            'prot_id2': {...}
            ...
            }
        """
        self.get_peptides_from_protein_dict(protein_dict)
        self.process_from_naked_peptide_seqs()

    def import_and_process_peptide_sequences(self, 
        pep_seq_list:list, protein_list:list=None,
    ):
        """ 
        Importing and process peptide sequences instead of proteins.
        The processing step is in :meth:`process_from_naked_peptide_seqs()`.

        Parameters
        ----------
        pep_seq_list : list
            Peptide sequence list

        protein_list : list, optional
            Protein id list which maps to pep_seq_list one-by-one, 
            by default None
        """
        self.get_peptides_from_peptide_sequence_list(
            pep_seq_list, protein_list
        )
        self.process_from_naked_peptide_seqs()

    def process_from_naked_peptide_seqs(self):
        """
        The peptide processing step which is 
        called by `import_and_process_...` methods.
        """
        self.append_decoy_sequence()
        self.add_modifications()
        self.add_special_modifications()
        self.add_peptide_labeling()
        self.add_charge()

    def get_peptides_from_fasta(self, fasta_file:Union[str,list]):
        """Load peptide sequences from fasta files.

        Parameters
        ----------
        fasta_file : Union[str,list]
            Could be a fasta file (str) or a list of fasta files (list[str])
        """
        if isinstance(fasta_file, str):
            self.get_peptides_from_fasta_list([fasta_file])
        else:
            self.get_peptides_from_fasta_list(fasta_file)

    def get_peptides_from_fasta_list(self, fasta_files:list):
        """Load peptide sequences from fasta file list

        Parameters
        ----------
        fasta_files : list
            fasta file list
        """
        if self.include_contaminants:
            fasta_files.append(os.path.join(
                CONST_FILE_FOLDER, 'contaminants.fasta'
            ))
        protein_dict = load_all_proteins(fasta_files)
        self.get_peptides_from_protein_dict(protein_dict)

    def _get_peptides_from_protein_df(self):
        if self.I_to_L:
            self.protein_df[
                'sequence_I2L'
            ] = self.protein_df.sequence.str.replace('I','L')
            digest_seq = 'sequence_I2L'
        else:
            digest_seq = 'sequence'
        self._cleave_to_peptides(
            self.protein_df,
            protein_seq_column=digest_seq
        )

    def get_peptides_from_protein_dict(self, protein_dict:dict):
        """Cleave the protein sequences in protein_dict.

        Parameters
        ----------
        protein_dict : dict
            Format:
            ```
            {
            'prot_id1': {'protein_id': 'prot_id1', 'sequence': string, 'gene_name': string, 'description': string
            'prot_id2': {...}
            ...
            }
            ```
        """
        self.protein_df = pd.DataFrame.from_dict(
            protein_dict, orient='index'
        ).reset_index(drop=True)
        self._get_peptides_from_protein_df()

    def _cleave_to_peptides(self, 
        protein_df:pd.DataFrame,
        protein_seq_column:str='sequence'
    ):
        """Cleave protein sequences in protein_df

        Parameters
        ----------
        protein_df : pd.DataFrame
            Protein DataFrame containing `protein_seq_column`
        protein_seq_column : str, optional
            Target column containing protein sequences, by default 'sequence'
        """
        pep_dict = {}

        for i,prot_seq in enumerate(
            protein_df[protein_seq_column].values
        ):
            (
                seq_list, miss_list, nterm_list, cterm_list
            ) = self._digest.cleave_sequence(prot_seq)
            for seq,miss,nterm,cterm in zip(
                seq_list,miss_list,nterm_list, cterm_list
            ):
                prot_id = str(i)
                if seq in pep_dict:
                    if not pep_dict[seq][0].endswith(prot_id):
                        pep_dict[seq][0] += ';'+prot_id
                    if nterm:
                        pep_dict[seq][2] = nterm
                    if cterm:
                        pep_dict[seq][3] = cterm
                else:
                    pep_dict[seq] = [prot_id,miss,nterm,cterm]
        self._precursor_df = pd.DataFrame().from_dict(
            pep_dict, orient='index', columns = [
                'protein_idxes','miss_cleavage',
                'is_prot_nterm','is_prot_cterm'
            ]
        )
        self._precursor_df.reset_index(drop=False, inplace=True)
        self._precursor_df.rename(
            columns={'index':'sequence'}, inplace=True
        )
        self._precursor_df['mods'] = ''
        self._precursor_df['mod_sites'] = ''
        self.refine_df()

    def append_protein_name(self):
        if (
            'protein_id' not in self.protein_df or 
            'protein_idxes' not in self._precursor_df
        ): 
            return

        self._precursor_df['proteins'] = self._precursor_df['protein_idxes'].apply(
            protein_idxes_to_names,
            protein_names=self.protein_df['protein_id'].values
        )

        if 'gene_name' in self.protein_df.columns:
            self._precursor_df['genes'] = self._precursor_df['protein_idxes'].apply(
                protein_idxes_to_names,
                protein_names=self.protein_df['gene_name'].values
            )

    def get_peptides_from_peptide_sequence_list(self, 
        pep_seq_list:list,
        protein_list:list = None
    ):
        self._precursor_df = pd.DataFrame()
        self._precursor_df['sequence'] = pep_seq_list
        if protein_list is not None:
            self._precursor_df['protein_name'] = protein_list
        self._precursor_df['is_prot_nterm'] = False
        self._precursor_df['is_prot_cterm'] = False
        self.refine_df()

    def add_mods_for_one_seq(self, sequence:str, 
        is_prot_nterm, is_prot_cterm
    )->tuple:
        """Add fixed and variable modifications to a sequence

        Parameters
        ----------
        sequence : str
            Peptide sequence
        is_prot_nterm : bool
            if protein N-term
        is_prot_cterm : bool
            if protein C-term

        Returns
        -------
        tuple
            list[str]: list of modification names
            list[str]: list of modification sites
        """
        fix_mods, fix_mod_sites = get_fix_mods(
            sequence, self.fix_mod_aas, self.fix_mod_dict
        )
        #TODO add prot and pep C-term fix mods
        #TODO add prot and pep N-term fix mods

        if len(fix_mods) == 0:
            fix_mods = ['']
            fix_mod_sites = ['']
        else:
            fix_mods = [fix_mods]
            fix_mod_sites = [fix_mod_sites]

        var_mods_list, var_mod_sites_list = get_var_mods(
            sequence, self.var_mod_aas, self.var_mod_dict, 
            self.min_var_mod_num, self.max_var_mod_num, 
            self.max_peptidoform_num-1, # 1 for unmodified
        )

        nterm_var_mods = ['']
        nterm_var_mod_sites = ['']
        if is_prot_nterm and len(self.var_mod_prot_nterm_dict)>0:
            if '' in self.var_mod_prot_nterm_dict:
                nterm_var_mods.extend(self.var_mod_prot_nterm_dict[''])
            if sequence[0] in self.var_mod_prot_nterm_dict:
                nterm_var_mods.extend(self.var_mod_prot_nterm_dict[sequence[0]])
        if len(self.var_mod_pep_nterm_dict)>0:
            if '' in self.var_mod_pep_nterm_dict:
                nterm_var_mods.extend(self.var_mod_pep_nterm_dict[''])
            if sequence[0] in self.var_mod_pep_nterm_dict:
                nterm_var_mods.extend(self.var_mod_pep_nterm_dict[sequence[0]])
        nterm_var_mod_sites.extend(['0']*(len(nterm_var_mods)-1))

        #TODO add prot and pep C-term var mods

        return (
            list(
                ';'.join([i for i in items if i]) for items in itertools.product(
                    fix_mods, nterm_var_mods, var_mods_list
                )
            ),
            list(
                ';'.join([i for i in items if i]) for items in itertools.product(
                    fix_mod_sites, nterm_var_mod_sites, var_mod_sites_list
                )
            ),
        )

    def add_modifications(self):
        """Add fixed and variable modifications to all peptide sequences in `self.precursor_df`
        """
        if 'is_prot_nterm' not in self._precursor_df.columns:
            self._precursor_df['is_prot_nterm'] = False
        if 'is_prot_cterm' not in self._precursor_df.columns:
            self._precursor_df['is_prot_cterm'] = False
        
        if len(self._precursor_df) == 0:
            self._precursor_df['mods'] = ""
            self._precursor_df['mod_sites'] = ""
            return
            
        (
            self._precursor_df['mods'],
            self._precursor_df['mod_sites']
        ) = zip(*self._precursor_df[
            ['sequence','is_prot_nterm','is_prot_cterm']
        ].apply(lambda x:
            self.add_mods_for_one_seq(*x), axis=1
        ))
        self._precursor_df = explode_multiple_columns(
            self._precursor_df,
            ['mods','mod_sites']
        )
        self._precursor_df.reset_index(drop=True, inplace=True)

    def add_special_modifications(self):
        """
        Add external defined variable modifications to 
        all peptide sequences in `self._precursor_df`.
        See :meth:`append_special_modifications()` for details.
        """
        if len(self.special_mods) == 0: return
        self._precursor_df = append_special_modifications(
            self._precursor_df, self.special_mods,
            self.min_special_mod_num, self.max_special_mod_num, 
            self.max_peptidoform_num,
            cannot_modify_pep_nterm_aa=self.special_mods_cannot_modify_pep_n_term,
            cannot_modify_pep_cterm_aa=self.special_mods_cannot_modify_pep_c_term,
        )

    def add_peptide_labeling(self, labeling_channel_dict:dict=None):
        """ 
        Add labeling onto peptides inplace of self._precursor_df

        Parameters
        ----------
        labeling_channel_dict : dict, optional
            For example:
            ```
            {
            -1: [], # not labeled
            0: ['Dimethyl@Any N-term','Dimethyl@K'],
            4: ['Dimethyl:2H(4)@Any N-term','Dimethyl:2H(4)@K'],
            8: ['Dimethyl:2H(6)13C(2)@Any N-term','Dimethyl:2H(6)13C(2)@K'],
            }
            ```.
            The key name could be int (highly recommended or 
            must be in the future) or str, and the value must be 
            a list of modification names (str) in alphabase format.
            It is set to `self.labeling_channels` if None.
            Defaults to None
    
        """
        if labeling_channel_dict is None:
            labeling_channel_dict = self.labeling_channels
        if labeling_channel_dict is None or len(labeling_channel_dict) == 0:
            return
        df_list = []
        for channel, labels in labeling_channel_dict.items():
            df = create_labeling_peptide_df(self._precursor_df, labels)
            df['labeling_channel'] = channel
            df_list.append(df)
        self._precursor_df = pd.concat(df_list, ignore_index=True)
        try:
            self._precursor_df[
                'labeling_channel'
            ] = self._precursor_df.labeling_channel.astype(np.int32)
            if 'labeling_channel' not in self.key_numeric_columns:
                self.key_numeric_columns.append('labeling_channel')
        except:
            if 'labeling_channel' in self.key_numeric_columns:
                self.key_numeric_columns.remove('labeling_channel')


    def add_charge(self):
        """Add charge states
        """
        self._precursor_df['charge'] = [
            np.arange(
                self.min_precursor_charge, 
                self.max_precursor_charge+1
            )
        ]*len(self._precursor_df)
        self._precursor_df = self._precursor_df.explode('charge')
        self._precursor_df['charge'] = self._precursor_df.charge.astype(np.int8)
        self._precursor_df.reset_index(drop=True, inplace=True)

    def save_hdf(self, hdf_file:str):
        """Save the contents into hdf file (attribute -> hdf_file):
        - self.precursor_df -> library/precursor_df
        - self.protein_df -> library/protein_df
        - self.fragment_mz_df -> library/fragment_mz_df
        - self.fragment_intensity_df -> library/fragment_intensity_df

        Parameters
        ----------
        hdf_file : str
            The hdf file path
        """
        super().save_hdf(hdf_file)
        _hdf = HDF_File(
            hdf_file,
            read_only=False,
            truncate=True,
            delete_existing=False
        )
        _hdf.library.protein_df = self.protein_df

    def load_hdf(self, hdf_file:str, load_mod_seq:bool=False):
        """Load contents from hdf file:
        - self.precursor_df <- library/precursor_df
        - self.precursor_df <- library/mod_seq_df if load_mod_seq is True
        - self.protein_df <- library/protein_df
        - self.fragment_mz_df <- library/fragment_mz_df
        - self.fragment_intensity_df <- library/fragment_intensity_df

        Parameters
        ----------
        hdf_file : str
            hdf file path

        load_mod_seq : bool, optional
            After library is generated with hash values (int64) for sequences (str) and modifications (str),
            we don't need sequence information for searching. 
            So we can skip loading sequences to make the loading much faster.
            By default False
        """
        super().load_hdf(hdf_file, load_mod_seq=load_mod_seq)
        try:
            _hdf = HDF_File(
                hdf_file,
            )
            self.protein_df = _hdf.library.protein_df.values
        except (AttributeError, KeyError, ValueError, TypeError):
            print(f"No protein_df in {hdf_file}")
