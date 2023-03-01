import tqdm
import os
import sys
import pandas as pd
import itertools
import io

# from alphatims
def process_bar(iterator, len_iter):
    with tqdm.tqdm(total=len_iter) as bar:
        i = 0
        for i,iter in enumerate(iterator):
            yield iter
            bar.update()
        bar.update(len_iter-i-1)

def _flatten(list_of_lists):
    '''
    Flatten a list of lists
    '''
    return list(
        itertools.chain.from_iterable(list_of_lists)
    )

def explode_multiple_columns(df:pd.DataFrame, columns:list):
    try:
        return df.explode(columns)
    except ValueError:
        # pandas < 1.3.0
        print(f'pandas=={pd.__version__} cannot explode multiple columns')
        ret_df = df.explode(columns[0])
        for col in columns[1:]:
            ret_df[col] = _flatten(df[col].values)
        return ret_df

def get_delimiter(tsv_file:str):
    if isinstance(tsv_file, io.StringIO):
        # for unit tests
        line = tsv_file.readline().strip()
        tsv_file.seek(0)
    else:
        with open(tsv_file, "r") as f:
            line = f.readline().strip()
    if '\t' in line: return '\t'
    elif ',' in line: return ','
    else: return '\t'
