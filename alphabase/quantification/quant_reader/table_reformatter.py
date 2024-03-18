import itertools
import pandas as pd
import copy

def merge_protein_cols_and_config_dict(input_df, config_dict, use_alphaquant_format = False):
    """[summary]
    
    Args:
        input_df ([pandas dataframe]): longtable containing peptide intensity data
        confid_dict ([dict[String[]]]): nested dict containing the parse information. derived from yaml file

    Returns:
        pandas dataframe: longtable with newly assigned "protein" and "ion" columns
    """
    protein_cols = config_dict.get("protein_cols")
    ion_hierarchy = config_dict.get("ion_hierarchy")
    splitcol2sep = config_dict.get('split_cols')
    quant_id_dict = config_dict.get('quant_ID')

    ion_dfs = []
    input_df['protein'] = join_columns(input_df, protein_cols)

    input_df = input_df.drop(columns = [x for x in protein_cols if x!='protein'])
    index_names = []
    for hierarchy_type in ion_hierarchy.keys():
        df_subset = input_df.copy()
        ion_hierarchy_local = ion_hierarchy.get(hierarchy_type).get("order")
        ion_headers_merged, ion_headers_grouped = get_ionname_columns(ion_hierarchy.get(hierarchy_type).get("mapping"), ion_hierarchy_local) #ion headers merged is just a helper to select all relevant rows, ionheaders grouped contains the sets of ionstrings to be merged into a list eg [[SEQ, MOD], [CH]]
        quant_columns = get_quantitative_columns(df_subset, hierarchy_type, config_dict, ion_headers_merged)
        headers = list(set(ion_headers_merged + quant_columns + ['protein']))
        if "sample_ID" in config_dict.keys():
            headers+=[config_dict.get("sample_ID")]
        df_subset = df_subset[headers].drop_duplicates()

        if splitcol2sep is not None:
            if quant_columns[0] in splitcol2sep.keys(): #in the case that quantitative values are stored grouped in one column (e.g. msiso1,msiso2,msiso3, etc.), reformat accordingly
                df_subset = split_extend_df(df_subset, splitcol2sep)
            ion_headers_grouped = adapt_headers_on_extended_df(ion_headers_grouped, splitcol2sep)

        #df_subset = df_subset.set_index(quant_columns)

        df_subset = add_index_and_metadata_columns(df_subset, ion_hierarchy_local, ion_headers_grouped, quant_id_dict, hierarchy_type)
        index_names += df_subset.index.names
        #add_merged_ionnames(df_subset, ion_hierarchy_local, ion_headers_grouped, quant_id_dict, hierarchy_type)
        ion_dfs.append(df_subset.reset_index())
    
    input_df = pd.concat(ion_dfs, ignore_index=True)
    if use_alphaquant_format:
        input_df = input_df.drop(columns=list(set(index_names)))
    else:
        input_df = input_df.set_index(list(set(index_names)))

    return input_df

def join_columns(df, columns, separator='_'):
    if len(columns) == 1:
        return df[columns[0]].fillna('nan').astype(str)
    else:
        return df[columns].fillna('nan').astype(str).agg(separator.join, axis=1)



def get_ionname_columns(ion_dict, ion_hierarchy_local):
    ion_headers_merged = []
    ion_headers_grouped = []
    for lvl in ion_hierarchy_local:
        vals = ion_dict.get(lvl)
        ion_headers_merged.extend(vals)
        ion_headers_grouped.append(vals)
    return ion_headers_merged, ion_headers_grouped

def get_quantitative_columns(input_df, hierarchy_type, config_dict, ion_headers_merged):
    naming_columns = ion_headers_merged + ['protein']
    if config_dict.get("format") == 'longtable':
        quantcol = config_dict.get("quant_ID").get(hierarchy_type)
        return [quantcol]

    if config_dict.get("format") == 'widetable':
        quantcolumn_candidates = [x for x in input_df.columns if x not in naming_columns]
        if "quant_pre_or_suffix" in config_dict.keys():
            return [x for x in quantcolumn_candidates if x.startswith(config_dict.get("quant_pre_or_suffix")) or x.endswith(config_dict.get("quant_pre_or_suffix"))] # in the case that the quantitative columns have a prefix (like "Intensity " in MQ peptides.txt), only columns with the prefix are filtered
        else:
            return quantcolumn_candidates #in this case, we assume that all non-ionname/proteinname columns are quantitative columns


def split_extend_df(input_df, splitcol2sep, value_threshold=10):
    """reformats data that is stored in a condensed way in a single column. For example isotope1_intensity;isotope2_intensity etc. in Spectronaut

    Args:
        input_df ([type]): [description]
        splitcol2sep ([type]): [description]
        value_threshold([type]): [description]

    Returns:
        Pandas Dataframe: Pandas dataframe with the condensed items expanded to long format
    """
    if splitcol2sep==None:
        return input_df

    for split_col, separator in splitcol2sep.items():
        idx_name = f"{split_col}_idxs"
        split_col_series = input_df[split_col].str.split(separator)
        input_df = input_df.drop(columns = [split_col])

        input_df[idx_name] = [list(range(len(x))) for x in split_col_series]
        exploded_input = input_df.explode(idx_name)
        exploded_split_col_series = split_col_series.explode()

        exploded_input[split_col] = exploded_split_col_series.replace('', 0) #the column with the intensities has to come after to column with the idxs

        exploded_input = exploded_input.astype({split_col: float})
        exploded_input = exploded_input[exploded_input[split_col]>value_threshold]
        #exploded_input = exploded_input.rename(columns = {'var1': split_col})
    return exploded_input

def adapt_headers_on_extended_df(ion_headers_grouped, splitcol2sep):
    #in the case that one column has been split, we need to designate the "naming" column
    ion_headers_grouped_copy = copy.deepcopy(ion_headers_grouped)
    for vals in ion_headers_grouped_copy:
        if splitcol2sep is not None:
            for idx in range(len(vals)):
                if vals[idx] in splitcol2sep.keys():
                    vals[idx] = vals[idx] + "_idxs"
    return ion_headers_grouped_copy


def merge_protein_and_ion_cols(input_df, config_dict):
    protein_cols =  config_dict.get("protein_cols")
    ion_cols = config_dict.get("ion_cols")
    input_df['protein'] = join_columns(input_df, protein_cols)
    input_df['quant_id'] = join_columns(input_df, ion_cols)
    input_df = input_df.rename(columns = {config_dict.get('quant_ID') : "quant_val"})
    return input_df


def add_index_and_metadata_columns(df_subset, ion_hierarchy_local, ion_headers_grouped, quant_id_dict, hierarchy_type):
    """puts together the hierarchical ion names as a column in a given input dataframe"""
    
    for idx in range(len(ion_hierarchy_local)):
        hierarchy_name = ion_hierarchy_local[idx]
        headers = ion_headers_grouped[idx]
        df_subset[hierarchy_name] = join_columns(df_subset, headers)# df_subset[headers].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        df_subset[hierarchy_name] = f"{hierarchy_name}_" +df_subset[hierarchy_name]
    
    df_subset['quant_id'] = join_columns(df_subset, ion_hierarchy_local, separator='_')# df_subset[ion_hierarchy_local].apply(lambda x: '_AND_'.join(x.astype(str)), axis=1)


    df_subset = df_subset.set_index(ion_hierarchy_local)
    if quant_id_dict!= None:
        df_subset = df_subset.rename(columns = {quant_id_dict.get(hierarchy_type) : "quant_val"})
    return df_subset