import itertools
import logging
import os
import os.path
import pathlib
import re

import yaml

from . import quantreader_utils

LOGGER = logging.getLogger(__name__)
INTABLE_CONFIG = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "../../../alphabase/constants/const_files/quant_reader_config.yaml",
)  # the yaml config is located one directory below the python library files


def get_input_type_and_config_dict(input_file, input_type_to_use=None):
    all_config_dicts = _load_config(INTABLE_CONFIG)
    type2relevant_columns = _get_type2relevant_cols(all_config_dicts)

    if "aq_reformat.tsv" in input_file:
        input_file = _get_original_file_from_aq_reformat(input_file)

    sep = _get_seperator(input_file)

    uploaded_data_columns = quantreader_utils.read_columns_from_file(
        input_file, sep=sep
    )

    for input_type in type2relevant_columns:
        if (input_type_to_use is not None) and (input_type != input_type_to_use):
            continue
        relevant_columns = type2relevant_columns.get(input_type)
        relevant_columns = [x for x in relevant_columns if x]  # filter None values
        if set(relevant_columns).issubset(uploaded_data_columns):
            config_dict = all_config_dicts.get(input_type)
            return input_type, config_dict, sep

    raise TypeError("format not specified in intable_config.yaml!")


def _get_original_file_from_aq_reformat(input_file):
    matched = re.match("(.*)(\..*\.)(aq_reformat\.tsv)", input_file)
    return matched.group(1)


def _get_seperator(input_file):
    filename = str(input_file)
    if ".csv" in filename:
        return ","
    elif ".tsv" in filename:
        return "\t"
    else:
        LOGGER.info(
            f"neither of the file extensions (.tsv, .csv) detected for file {input_file}! Trying with tab separation. In the case that it fails, please provide the correct file extension"
        )
    return "\t"


def _load_config(config_yaml):
    with open(config_yaml) as stream:
        config_all = yaml.safe_load(stream)
    return config_all


def _get_type2relevant_cols(config_all):
    type2relcols = {}
    for type in config_all:
        config_typedict = config_all.get(type)
        relevant_cols = get_relevant_columns_config_dict(config_typedict)
        type2relcols[type] = relevant_cols
    return type2relcols


def get_relevant_columns_config_dict(config_typedict):
    filtcols = []
    dict_ioncols = []
    for filtconf in config_typedict.get("filters", {}).values():
        filtcols.append(filtconf.get("param"))

    if "ion_hierarchy" in config_typedict:
        for headr in config_typedict.get("ion_hierarchy").values():
            ioncols = list(itertools.chain.from_iterable(headr.get("mapping").values()))
            dict_ioncols.extend(ioncols)

    quant_ids = _get_quant_ids_from_config_dict(config_typedict)
    sample_ids = _get_sample_ids_from_config_dict(config_typedict)
    channel_ids = _get_channel_ids_from_config_dict(config_typedict)
    relevant_cols = (
        config_typedict.get("protein_cols")
        + config_typedict.get("ion_cols", [])
        + sample_ids
        + quant_ids
        + filtcols
        + dict_ioncols
        + channel_ids
    )
    relevant_cols = list(set(relevant_cols))  # to remove possible redudancies
    return relevant_cols


def _get_quant_ids_from_config_dict(config_typedict):
    quantID = config_typedict.get("quant_ID")
    if isinstance(quantID, str):
        return [config_typedict.get("quant_ID")]
    if quantID is None:
        return []
    else:
        return list(config_typedict.get("quant_ID").values())


def _get_sample_ids_from_config_dict(config_typedict):
    sampleID = config_typedict.get("sample_ID")
    if isinstance(sampleID, str):
        return [config_typedict.get("sample_ID")]
    if sampleID is None:
        return []
    else:
        return config_typedict.get("sample_ID")


def _get_channel_ids_from_config_dict(config_typedict):
    return config_typedict.get("channel_ID", [])


def import_config_dict():
    config_dict = _load_config(INTABLE_CONFIG)
    return config_dict
