def extend_sample_allcolumns_for_mDIA_case(allcols_samples, config_dict_for_type):
    if is_mDIA_table(config_dict_for_type):
        new_allcols = []
        channels = ['Dimethyl-n-0', 'Dimethyl-n-4', 'Dimethyl-n-8']
        for channel in channels:
            for sample in allcols_samples:
                new_allcols.append(merge_channel_and_sample_string(sample, channel))
        return new_allcols
    else:
        return allcols_samples


# Cell
#mDIA case

def adapt_input_df_columns_in_case_of_mDIA(input_df,config_dict_for_type):
    if is_mDIA_table(config_dict_for_type):
        input_df = extend_sampleID_column_for_mDIA_case(input_df, config_dict_for_type)
        input_df = set_mtraq_reduced_ion_column_into_dataframe(input_df)
        return input_df
    else:
        return input_df


def extend_sampleID_column_for_mDIA_case(input_df,config_dict_for_type):
    channels_per_peptide = parse_channel_from_peptide_column(input_df)
    return merge_sample_id_and_channels(input_df, channels_per_peptide, config_dict_for_type)


def set_mtraq_reduced_ion_column_into_dataframe(input_df):
    new_ions = remove_mtraq_modifications_from_ion_ids(input_df['quant_id'])
    input_df['quant_id'] = new_ions
    return input_df

def remove_mtraq_modifications_from_ion_ids(ions):
    new_ions = []
    for ion in ions:
        new_ions.append( re.sub("\(Dimethyl-\w-\d\)","", ion))
    return new_ions


def is_mDIA_table(config_dict_for_type):
    return config_dict_for_type.get('channel_ID') == ['Channel.0', 'Channel.4']


import re
def parse_channel_from_peptide_column(input_df):
    channels = []
    for pep in input_df['Modified.Sequence']:
        pattern = "(.*)(\(Dimethyl-n-.\))(.*)"
        matched = re.match(pattern, pep)
        num_appearances = pep.count("Dimethyl-n-")
        if matched and num_appearances==1:
            channels.append(matched.group(2))
        else:
            channels.append("NA")
    return channels

def merge_sample_id_and_channels(input_df, channels, config_dict_for_type):
    sample_id = config_dict_for_type.get("sample_ID")
    sample_ids = list(input_df[sample_id])
    input_df[sample_id] = [merge_channel_and_sample_string(sample_ids[idx], channels[idx]) for idx in range(len(sample_ids))]
    return input_df

def merge_channel_and_sample_string(sample, channel):
    return f"{sample}_{channel}"

