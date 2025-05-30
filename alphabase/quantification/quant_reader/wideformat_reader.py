from . import quantreader_utils, table_reformatter


def reformat_and_write_wideformat_table(peptides_tsv, outfile_name, config_dict):
    input_df = quantreader_utils.read_file(peptides_tsv, sep="\t")
    filter_dict = config_dict.get("filters")

    input_df = quantreader_utils.filter_input(filter_dict, input_df)
    # input_df = merge_protein_and_ion_cols(input_df, config_dict)
    input_df = table_reformatter.merge_protein_cols_and_config_dict(
        input_df, config_dict
    )
    if "quant_pre_or_suffix" in config_dict:
        quant_pre_or_suffix = config_dict.get("quant_pre_or_suffix")
        headers = ["protein", "quant_id"] + list(
            filter(
                lambda x: x.startswith(quant_pre_or_suffix)
                or x.endswith(quant_pre_or_suffix),
                input_df.columns,
            )
        )
        input_df = input_df[headers]
        input_df = input_df.rename(columns=lambda x: x.replace(quant_pre_or_suffix, ""))

    # input_df = input_df.reset_index()

    input_df.to_csv(outfile_name, sep="\t", index=None)
