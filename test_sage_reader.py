# ruff: noqa
from alphabase.psm_reader.sage_reader import *


def test_reader():
    annotated_mod_df = get_annotated_mod_df()
    register_readers()
    result = capture_modifications(
        "Q[-17.026548]DQSANEKNK[+42.010567]LEM[+15.9949]NK[+42.010567]",
        annotated_mod_df,
    )
    expected = ("0;9;12;14", "Gln->pyro-Glu@Q^Any N-term;Acetyl@K;Oxidation@M;Acetyl@K")
    assert result == expected, f"Expected: {expected}, got: {result}"
