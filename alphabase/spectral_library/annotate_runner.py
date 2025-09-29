#!/usr/bin/env python3
"""Profile script for annotate_runner.py using line_profiler."""

from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library.annotate import add_dense_lib
from alphabase.spectral_library.flat import SpecLibFlat


def main():
    """Main function to profile annotate_runner logic."""
    speclib_flat = SpecLibFlat()
    speclib_flat.load_hdf(
        "/fs/gpfs41/lv03/fileset01/pool/pool-mann-borgwardt/alphanovo/metaptcm/v0.7/muller_2020_Bacillus_subtilis_evidence_txt_9_batch_0.hdf"
    )

    speclib_flat.precursor_df["nAA"] = speclib_flat.precursor_df["nAA"].astype(int)
    seq_coverage = speclib_flat.precursor_df["sequence_coverage"]

    new_speclib_flat = add_dense_lib(
        speclib_flat,
        get_charged_frag_types(
            [
                "a",
                "x",
                "b",
                "y",
                "c",
                "z",
                "b_NH3",
                "y_NH3",
                "b_H2O",
                "y_H2O",
                "c_lossH",
                "z_addH",
            ],
            2,
        ),
    )

    assert (
        new_speclib_flat.precursor_df["sequence_coverage"] == seq_coverage
    ).sum() / len(seq_coverage) > 0.999


if __name__ == "__main__":
    main()
