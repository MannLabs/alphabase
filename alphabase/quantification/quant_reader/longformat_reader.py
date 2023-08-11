import pandas as pd
import os
import shutil
import glob
import dask.dataframe as dd
import os.path
import sys

from . import config_dict_loader
from . import quantreader_utils
from . import table_reformatter
from . import plexdia_reformatter




def reformat_and_write_longtable_according_to_config(input_file, outfile_name, config_dict_for_type, sep = "\t",decimal = ".", enforce_largefile_processing = False, 
                                                     chunksize =1000_000, use_alphaquant_format = False):
    """Reshape a long format proteomics results table (e.g. Spectronaut or DIA-NN) to a wide format table.
    :param file input_file: long format proteomic results table
    :param string input_type: the configuration key stored in the config file (e.g. "diann_precursor")
    """


    file_is_large = check_if_file_is_large(input_file, enforce_largefile_processing)

    if file_is_large:
        process_out_of_memory(input_file=input_file, outfile_name=outfile_name, config_dict_for_type=config_dict_for_type, sep=sep, decimal=decimal,
                              chunksize=chunksize, use_alphaquant_format=use_alphaquant_format)
    else:
        process_in_memory(input_file=input_file, outfile_name=outfile_name, config_dict_for_type=config_dict_for_type, sep=sep, decimal=decimal,
                              chunksize=chunksize, use_alphaquant_format=use_alphaquant_format)


def check_if_file_is_large(input_file, enforce_largefile_processing):
    filesize = os.path.getsize(input_file)/(1024**3) #size in gigabyte
    file_is_large = (filesize>10 and str(input_file).endswith(".zip")) or filesize>50 or enforce_largefile_processing
    return file_is_large

        

def process_out_of_memory(input_file, outfile_name, config_dict_for_type, sep = "\t",decimal = ".", 
                                                     chunksize =1000_000, use_alphaquant_format = False):

    tmpfile_large = get_tmpfile_location(input_file)
    remove_possible_old_tmpfiles(tmpfile_large=tmpfile_large, outfile_name=outfile_name)
    
    relevant_cols = config_dict_loader.get_relevant_columns_config_dict(config_dict_for_type)
    input_df_it = pd.read_csv(input_file, sep = sep, decimal=decimal, usecols = relevant_cols, encoding ='latin1', chunksize = chunksize)
    header = True
    
    for input_df_subset in input_df_it:
        input_df_subset = adapt_subtable(input_df_subset, config_dict_for_type, use_alphaquant_format)
        write_chunk_to_file(input_df_subset,tmpfile_large, header, write_index=True)
        header = False

    process_with_dask(tmpfile_columnfilt=tmpfile_large , outfile_name = outfile_name, config_dict_for_type=config_dict_for_type, use_alphaquant_format=use_alphaquant_format)

def get_tmpfile_location(input_file):
    return f"{input_file}.tmp.longformat.columnfilt.tsv" #only needed when file is large

def remove_possible_old_tmpfiles(tmpfile_large, outfile_name):
    #remove potential leftovers from previous processings
    if os.path.exists(tmpfile_large):
        os.remove(tmpfile_large)
    if os.path.exists(outfile_name):
        os.remove(outfile_name)


def process_in_memory(input_file, outfile_name, config_dict_for_type, sep = "\t",decimal = ".",
                                                     chunksize =1000_000, use_alphaquant_format = False):

    relevant_cols = config_dict_loader.get_relevant_columns_config_dict(config_dict_for_type)
    input_df_it = pd.read_csv(input_file, sep = sep, decimal=decimal, usecols = relevant_cols, encoding ='latin1', chunksize = chunksize)
    input_df_list = []
    for input_df_subset in input_df_it:
        input_df_subset = adapt_subtable(input_df_subset, config_dict_for_type, use_alphaquant_format)
        input_df_list.append(input_df_subset)

    input_df = pd.concat(input_df_list)
    input_reshaped = reshape_input_df(input_df, config_dict_for_type)
    input_reshaped.to_csv(outfile_name, sep = "\t", index = None)


def adapt_subtable(input_df_subset, config_dict, use_alphaquant_format):
    input_df_subset = quantreader_utils.filter_input(config_dict.get("filters", {}), input_df_subset)
    if "ion_hierarchy" in config_dict.keys():
        return table_reformatter.merge_protein_cols_and_config_dict(input_df_subset, config_dict, use_alphaquant_format)
    else:
        return table_reformatter.merge_protein_and_ion_cols(input_df_subset, config_dict)

def write_chunk_to_file(chunk, filepath ,write_header, write_index):
    """write chunk of pandas dataframe to a file"""
    chunk.to_csv(filepath, header=write_header, mode='a', sep = "\t", index = write_index)

def reshape_input_df(input_df, config_dict):
    input_df = input_df.astype({'quant_val': 'float'})
    input_df = plexdia_reformatter.adapt_input_df_columns_in_case_of_mDIA(input_df=input_df, config_dict_for_type=config_dict)
    column_names = get_column_names(input_df)
    input_df = adapt_input_df(input_df, column_names)
    input_reshaped = pd.pivot_table(input_df, index = ['protein', 'quant_id']+column_names, columns = config_dict.get("sample_ID"), values = 'quant_val', fill_value=0)
    input_reshaped = input_reshaped.reset_index()
    return input_reshaped

def get_column_names(input_df):
    if input_df.index.names[0] is None:
        return []
    else:
        return input_df.index.names

def adapt_input_df(input_df, column_names):
    if len(column_names)==0:
        return input_df.reset_index()
    else:
        input_df = input_df.reset_index()
        input_df[column_names] = input_df[column_names].astype(str)
        return input_df




def process_with_dask(*, tmpfile_columnfilt, outfile_name, config_dict_for_type, use_alphaquant_format):
    df = dd.read_csv(tmpfile_columnfilt, sep = "\t")
    allcols = df[config_dict_for_type.get("sample_ID")].drop_duplicates().compute() # the columns of the output table are the sample IDs
    allcols = plexdia_reformatter.extend_sample_allcolumns_for_mDIA_case(allcols_samples=allcols, config_dict_for_type=config_dict_for_type)
    allcols = ['protein', 'quant_id'] + sorted(allcols)

    if not use_alphaquant_format:
        hierarchy_columns = get_hierarchy_names_from_config_dict(config_dict_for_type)
        allcols = allcols + hierarchy_columns
    df = df.set_index('protein')
    sorted_filedir = f"{tmpfile_columnfilt}_sorted"
    df.to_csv(sorted_filedir, sep = "\t")
    #now the files are sorted and can be pivoted chunkwise (multiindex pivoting at the moment not possible in dask)
    files_dask = glob.glob(f"{sorted_filedir}/*part")
    header = True
    for file in files_dask:
        if use_alphaquant_format:
            input_df = pd.read_csv(file, sep = "\t")
        else:
            input_df = pd.read_csv(file, sep = "\t", index_col=hierarchy_columns)
        if len(input_df.index) <2:
            continue
        input_reshaped = reshape_input_df(input_df, config_dict_for_type)
        input_reshaped = sort_and_add_columns(input_reshaped, allcols)
        write_chunk_to_file(input_reshaped, outfile_name, header, write_index = False)
        header = False
    os.remove(tmpfile_columnfilt)
    shutil.rmtree(sorted_filedir)

def get_hierarchy_names_from_config_dict(config_dict_for_type):
    hierarchy_names = []
    if "ion_hierarchy" in config_dict_for_type.keys():
        ion_hierarchy = config_dict_for_type.get("ion_hierarchy")
        for hierarchy_type in ion_hierarchy.keys():
            hierarchy_names += ion_hierarchy.get(hierarchy_type).get("order")
        return list(set(hierarchy_names))
    else:
        return []
    

def sort_and_add_columns(input_reshaped, allcols):
    missing_cols = set(allcols) - set(input_reshaped.columns)
    input_reshaped[list(missing_cols)] = 0
    input_reshaped = input_reshaped[allcols]
    return input_reshaped





