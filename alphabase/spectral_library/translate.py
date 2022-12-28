import pandas as pd
import numpy as np
import tqdm
import typing
import numba
import multiprocessing as mp

from alphabase.constants.modification import MOD_DF

from alphabase.spectral_library.base import SpecLibBase

from alphabase.utils import explode_multiple_columns

#@numba.njit #(cannot use numba for pd.Series)
def create_modified_sequence(
    seq_mods_sites:typing.Tuple, # must be ('sequence','mods','mod_sites')
    translate_mod_dict:dict=None,
    mod_sep='()',
    nterm = '_',
    cterm = '_'
):
    '''
    Translate `(sequence, mods, mod_sites)` into a modified sequence. Used by `df.apply()`.
    For example, `('ABCDEFG','Mod1@A;Mod2@E','1;5')`->`_A[Mod1@A]BCDE[Mod2@E]FG_`.

    Parameters
    ----------
    seq_mods_sites : List
        must be `(sequence, mods, mod_sites)`

    translate_mod_dict : dict
        A dict to map AlphaX modification names to other software,
        use unimod name if None.
        Defaults to None.

    mod_sep : str
        '[]' or '()', default '()'

    '''
    mod_seq = seq_mods_sites[0]
    if seq_mods_sites[1]:
        mods = seq_mods_sites[1].split(';')
        mod_sites = [int(i) for i in seq_mods_sites[2].split(';')]
        rev_order = np.argsort(mod_sites)[::-1]
        mod_sites = [mod_sites[rev_order[i]] for i in range(len(mod_sites))]
        mods = [mods[rev_order[i]] for i in range(len(mods))]
        if translate_mod_dict is None:
            mods = [mod[:mod.find('@')] for mod in mods]
        else:
            mods = [translate_mod_dict[mod] for mod in mods]
        for _site, mod in zip(mod_sites, mods):
            if _site > 0:
                mod_seq = mod_seq[:_site] + mod_sep[0]+mod+mod_sep[1] + mod_seq[_site:]
            elif _site == -1:
                cterm += mod_sep[0]+mod+mod_sep[1]
            elif _site == 0:
                nterm += mod_sep[0]+mod+mod_sep[1]
            else:
                mod_seq = mod_seq[:_site] + mod_sep[0]+mod+mod_sep[1] + mod_seq[_site:]
    return nterm + mod_seq + cterm

@numba.njit
def _get_frag_info_from_column_name(column:str):
    '''
    Only used when converting alphabase libraries into other libraries
    '''
    idx = column.rfind('_')
    frag_type = column[:idx]
    charge = column[idx+2:]
    if len(frag_type)==1:
        loss_type = 'noloss'
    else:
        idx = frag_type.find('_')
        loss_type = frag_type[idx+1:]
        frag_type = frag_type[0]
    return frag_type, loss_type, charge

def _get_frag_num(columns, rows, frag_len):
    frag_nums = []
    for r,c in zip(rows, columns):
        if is_nterm_frag(c):
            frag_nums.append(r+1)
        else:
            frag_nums.append(frag_len-r)
    return frag_nums

def merge_precursor_fragment_df(
    precursor_df:pd.DataFrame, 
    fragment_mz_df:pd.DataFrame, 
    fragment_inten_df:pd.DataFrame, 
    top_n_inten:int,
    frag_type_head:str='FragmentType',
    frag_mass_head:str='FragmentMz',
    frag_inten_head:str='RelativeIntensity',
    frag_charge_head:str='FragmentCharge',
    frag_loss_head:str='FragmentLossType',
    frag_num_head:str='FragmentNumber',
    verbose=True,
):
    '''
    Convert alphabase library into a single dataframe. 
    This method is not important, as it will be only 
    used by DiaNN, or spectronaut, or others
    '''
    df = precursor_df.copy()
    frag_columns = fragment_mz_df.columns.values.astype('U')
    frag_type_list = []
    frag_loss_list = []
    frag_charge_list = []
    frag_mass_list = []
    frag_inten_list = []
    frag_num_list = []
    iters =  enumerate(df[['frag_start_idx','frag_stop_idx']].values)
    if verbose:
        iters = tqdm.tqdm(iters)
    for i,(start, end) in iters:
        intens = fragment_inten_df.iloc[start:end,:].values # is loc[start:end-1,:] faster?
        max_inten = np.amax(intens)
        if max_inten > 0:
            intens /= max_inten
        masses = fragment_mz_df.iloc[start:end,:].values
        sorted_idx = np.argsort(intens.reshape(-1))[-top_n_inten:][::-1]
        idx_in_df = np.unravel_index(sorted_idx, masses.shape)

        frag_len = end-start
        rows = np.arange(frag_len, dtype=np.int32)[idx_in_df[0]]
        columns = frag_columns[idx_in_df[1]]

        frag_types, loss_types, charges = zip(
            *[_get_frag_info_from_column_name(_) for _ in columns]
        )

        frag_nums = _get_frag_num(columns, rows, frag_len)

        frag_type_list.append(frag_types)
        frag_loss_list.append(loss_types)
        frag_charge_list.append(charges)
        frag_mass_list.append(masses[idx_in_df])
        frag_inten_list.append(intens[idx_in_df])
        frag_num_list.append(frag_nums)
    
    df[frag_type_head] = frag_type_list
    df[frag_mass_head] = frag_mass_list
    df[frag_inten_head] = frag_inten_list
    df[frag_charge_head] = frag_charge_list
    df[frag_loss_head] = frag_loss_list
    df[frag_num_head] = frag_num_list

    return explode_multiple_columns(df, 
        [
            frag_type_head,
            frag_mass_head,
            frag_inten_head,
            frag_charge_head,
            frag_loss_head,
            frag_num_head
        ]
    )

    # try:
    #     return df.explode([
    #         frag_type_head,
    #         frag_mass_head,
    #         frag_inten_head,
    #         frag_charge_head,
    #         frag_loss_head,
    #         frag_num_head
    #     ])
    # except ValueError:
    #     # df.explode does not allow mulitple columns before pandas version 1.x.x.
    #     df = df.explode(frag_type_head)

    #     df[frag_mass_head] = _flatten(frag_mass_list)
    #     df[frag_inten_head] = _flatten(frag_inten_list)
    #     df[frag_charge_head] = _flatten(frag_charge_list)
    #     df[frag_loss_head] = _flatten(frag_loss_list)
    #     df[frag_num_head] = _flatten(frag_num_list)
    #     return df

mod_to_unimod_dict = {}
for mod_name,unimod_id in MOD_DF[['mod_name','unimod_id']].values:
    if unimod_id==-1 or unimod_id=='-1': continue
    mod_to_unimod_dict[mod_name] = f"UniMod:{unimod_id}"

def is_nterm_frag(frag_type:str):
    return frag_type[0] in 'abc'

def mask_fragment_intensity_by_mz_(
    fragment_mz_df:pd.DataFrame, 
    fragment_intensity_df:pd.DataFrame,
    min_frag_mz, max_frag_mz
):
    fragment_intensity_df.mask(
        (fragment_mz_df>max_frag_mz)|(fragment_mz_df<min_frag_mz),
        0, inplace=True
    )

def mask_fragment_intensity_by_frag_nAA(
    fragment_intensity_df:pd.DataFrame,
    precursor_df:pd.DataFrame,
    max_mask_frag_nAA
):
    if max_mask_frag_nAA <= 0: return
    b_mask = np.zeros(
        len(fragment_intensity_df), 
        dtype=np.bool_
    )
    y_mask = b_mask.copy()
    for i_frag in range(max_mask_frag_nAA):
        b_mask[precursor_df.frag_start_idx.values+i_frag] = True
        y_mask[precursor_df.frag_stop_idx.values-i_frag-1] = True
    
    masks = np.zeros(
        (
            len(fragment_intensity_df),
            len(fragment_intensity_df.columns)
        ), dtype=np.bool_
    )
    for i,col in enumerate(fragment_intensity_df.columns.values):
        if is_nterm_frag(col):
            masks[:,i] = b_mask
        else:
            masks[:,i] = y_mask
    
    fragment_intensity_df.mask(
        masks, 0, inplace=True
    )

def speclib_to_single_df(
    speclib:SpecLibBase,
    *,
    translate_mod_dict:dict = None,
    keep_k_highest_fragments:int=12,
    min_frag_mz = 200,
    max_frag_mz = 2000,
    min_frag_intensity = 0.01,
    min_frag_nAA = 0,
    modloss='H3PO4',
    frag_type_head:str='FragmentType',
    frag_mass_head:str='FragmentMz',
    frag_inten_head:str='RelativeIntensity',
    frag_charge_head:str='FragmentCharge',
    frag_loss_head:str='FragmentLossType',
    frag_num_head:str='FragmentNumber',
    verbose = True,
)->pd.DataFrame:
    '''
    Convert alphabase library to diann (or Spectronaut) library dataframe
    This method is not important, as it will be only 
    used by DiaNN, or spectronaut, or others

    Parameters
    ----------
    translate_mod_dict : dict
        A dict to map AlphaX modification names to other software,
        use unimod name if None.
        Defaults to None.
    
    keep_k_highest_peaks : int
        only keep highest fragments for each precursor. Default: 12

    Returns
    -------
    pd.DataFrame
        a single dataframe in the SWATH-like format

    '''
    df = pd.DataFrame()
    df['ModifiedPeptide'] = speclib._precursor_df[
        ['sequence','mods','mod_sites']
    ].apply(
        create_modified_sequence, 
        axis=1,
        translate_mod_dict=translate_mod_dict,
        mod_sep='()'
    )

    df['frag_start_idx'] = speclib._precursor_df['frag_start_idx']
    df['frag_stop_idx'] = speclib._precursor_df['frag_stop_idx']
    
    df['PrecursorCharge'] = speclib._precursor_df['charge']
    if 'irt_pred' in speclib._precursor_df.columns:
        df['Tr_recalibrated'] = speclib._precursor_df['irt_pred']
    elif 'rt_pred' in speclib._precursor_df.columns:
        df['Tr_recalibrated'] = speclib._precursor_df['rt_pred']
    elif 'rt_norm' in speclib._precursor_df.columns:
        df['Tr_recalibrated'] = speclib._precursor_df['rt_norm']
    else:
        raise ValueError('precursor_df must contain the "rt_pred" or "rt_norm" column')

    if 'mobility_pred' in speclib._precursor_df.columns:
        df['IonMobility'] = speclib._precursor_df.mobility_pred
    elif 'mobility' in speclib._precursor_df.columns:
        df['IonMobility'] = speclib._precursor_df.mobility
    
    # df['LabelModifiedSequence'] = df['ModifiedPeptide']
    df['StrippedPeptide'] = speclib._precursor_df['sequence']

    if 'precursor_mz' not in speclib._precursor_df.columns:
        speclib.calc_precursor_mz()
    df['PrecursorMz'] = speclib._precursor_df['precursor_mz']

    if 'uniprot_ids' in speclib._precursor_df.columns:
        df['ProteinID'] = speclib._precursor_df.uniprot_ids
    elif 'proteins' in speclib._precursor_df.columns:
        df['ProteinID'] = speclib._precursor_df.proteins

    if 'genes' in speclib._precursor_df.columns:
        df['Genes'] = speclib._precursor_df['genes']

    # if 'protein_group' in speclib._precursor_df.columns:
    #     df['ProteinGroups'] = speclib._precursor_df['protein_group']

    if min_frag_mz > 0 or max_frag_mz > 0:
        mask_fragment_intensity_by_mz_(
            speclib._fragment_mz_df,
            speclib._fragment_intensity_df,
            min_frag_mz, max_frag_mz
        )

    if min_frag_nAA > 0:
        mask_fragment_intensity_by_frag_nAA(
            speclib._fragment_intensity_df,
            speclib._precursor_df,
            max_mask_frag_nAA=min_frag_nAA-1
        )

    df = merge_precursor_fragment_df(
        df,
        speclib._fragment_mz_df,
        speclib._fragment_intensity_df,
        top_n_inten=keep_k_highest_fragments,
        frag_type_head=frag_type_head,
        frag_mass_head=frag_mass_head,
        frag_inten_head=frag_inten_head,
        frag_charge_head=frag_charge_head,
        frag_loss_head=frag_loss_head,
        frag_num_head=frag_num_head,
        verbose=verbose
    )
    df = df[df['RelativeIntensity']>min_frag_intensity]
    df.loc[df[frag_loss_head]=='modloss',frag_loss_head] = modloss

    return df.drop(['frag_start_idx','frag_stop_idx'], axis=1)

def speclib_to_swath_df(
    speclib:SpecLibBase,
    *,
    keep_k_highest_fragments:int=12,
    min_frag_mz = 200,
    max_frag_mz = 2000,
    min_frag_intensity = 0.01,
)->pd.DataFrame:
    speclib_to_single_df(
        speclib, 
        translate_mod_dict=None,
        keep_k_highest_fragments=keep_k_highest_fragments,
        min_frag_mz = min_frag_mz,
        max_frag_mz = max_frag_mz,
        min_frag_intensity = min_frag_intensity,
    )

class WritingProcess(mp.Process):
    def __init__(self, task_queue, tsv, *args, **kwargs):
        self.task_queue:mp.Queue = task_queue
        self.tsv = tsv
        super().__init__(*args, **kwargs)

    def run(self):
        while True:
            df, batch = self.task_queue.get()
            if df is None: break
            df.to_csv(self.tsv, header=(batch==0), sep="\t", mode="a", index=False)


def translate_to_tsv(
    speclib:SpecLibBase,
    tsv:str,
    *,
    keep_k_highest_fragments:int=12,
    min_frag_mz:float = 200,
    max_frag_mz:float = 2000,
    min_frag_intensity:float = 0.01,
    min_frag_nAA:int = 0,
    batch_size:int = 100000,
    translate_mod_dict:dict = None,
    multiprocessing:bool=True
):
    if multiprocessing:
        queue_size = 1000000//batch_size
        if queue_size < 2:
            queue_size = 2
        elif queue_size > 10:
            queue_size = 10
        df_head_queue = mp.Queue(maxsize=queue_size)
        writing_process = WritingProcess(df_head_queue, tsv)
        writing_process.start()
    mask_fragment_intensity_by_mz_(
        speclib._fragment_mz_df,
        speclib._fragment_intensity_df,
        min_frag_mz, max_frag_mz
    )
    if min_frag_nAA > 0:
        mask_fragment_intensity_by_frag_nAA(
            speclib._fragment_intensity_df,
            speclib._precursor_df,
            max_mask_frag_nAA=min_frag_nAA-1
        )
    if isinstance(tsv, str):
        with open(tsv, "w"): pass
    _speclib = SpecLibBase()
    _speclib._fragment_intensity_df = speclib._fragment_intensity_df
    _speclib._fragment_mz_df = speclib._fragment_mz_df
    precursor_df = speclib._precursor_df
    for i in tqdm.tqdm(range(0, len(precursor_df), batch_size)):
        _speclib._precursor_df = precursor_df.iloc[i:i+batch_size]
        df = speclib_to_single_df(
            _speclib, translate_mod_dict=translate_mod_dict,
            keep_k_highest_fragments=keep_k_highest_fragments,
            min_frag_mz=0,
            max_frag_mz=0,
            min_frag_intensity=min_frag_intensity,
            min_frag_nAA=0,
            verbose=False
        )
        if multiprocessing:
            df_head_queue.put((df, i))
        else:
            df.to_csv(tsv, header=(i==0), sep="\t", mode='a', index=False)
    if multiprocessing:
        df_head_queue.put((None, None))
        print("Translation finished, it will take several minutes to export the rest precursors to the tsv file...")
        writing_process.join()
        
