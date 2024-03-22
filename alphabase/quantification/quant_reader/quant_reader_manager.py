import pandas as pd
from . import config_dict_loader
from . import longformat_reader
from . import wideformat_reader


def import_data(input_file, input_type_to_use = None, samples_subset = None, results_dir = None, use_alphaquant_format = False):
    """
    Function to import peptide level data. Depending on available columns in the provided file,
    the function identifies the type of input used (e.g. Spectronaut, MaxQuant, DIA-NN), reformats if necessary
    and returns a generic wide-format dataframe
    :param file input_file: quantified peptide/ion -level data
    :param file results_folder: the folder where the directlfq outputs are stored
    """

    samples_subset = add_ion_protein_headers_if_applicable(samples_subset)
    if "aq_reformat" in input_file:
        file_to_read = input_file
    else:
        file_to_read = reformat_and_save_input_file(input_file=input_file, input_type_to_use=input_type_to_use, use_alphaquant_format = use_alphaquant_format)
    
    input_reshaped = pd.read_csv(file_to_read, sep = "\t", encoding = 'latin1', usecols=samples_subset)
    input_reshaped = input_reshaped.drop_duplicates(subset='quant_id')
    return input_reshaped

def add_ion_protein_headers_if_applicable(samples_subset):
    if samples_subset is not None:
        return samples_subset + ["quant_id", "protein"]
    else:
        return None

def reformat_and_save_input_file(input_file, input_type_to_use = None, use_alphaquant_format = False):
    
    input_type, config_dict_for_type, sep = config_dict_loader.get_input_type_and_config_dict(input_file, input_type_to_use)
    print(f"using input type {input_type}")
    format = config_dict_for_type.get('format')
    outfile_name = f"{input_file}.{input_type}.aq_reformat.tsv"

    if format == "longtable":
        longformat_reader.reformat_and_write_longtable_according_to_config(input_file, outfile_name,config_dict_for_type, sep = sep, use_alphaquant_format=use_alphaquant_format)
    elif format == "widetable":
        wideformat_reader.reformat_and_write_wideformat_table(input_file, outfile_name, config_dict_for_type)
    else:
        raise Exception('Format not recognized!')
    return outfile_name

def set_quanttable_config_location(quanttable_config_file):
    config_dict_loader.INTABLE_CONFIG = quanttable_config_file



