"""Multiprocessing workers must see the same modification registry as the parent.

Workers are started with the "spawn" start method and re-import alphabase, so
they only see `modification.tsv` unless the registry is handed to them
explicitly. These tests cover each way the registry can change at runtime.
"""

import os

import pandas as pd
import pytest

from alphabase.constants._const import CONST_FILE_FOLDER
from alphabase.constants.modification import (
    add_modifications_for_lower_case_AA,
    add_new_modifications,
    get_modification_state,
    keep_modloss_by_importance,
    load_mod_df,
    set_modification_state,
)
from alphabase.peptide.precursor import (
    calc_precursor_isotope_intensity,
    calc_precursor_isotope_intensity_mp,
    update_precursor_mz,
)
from alphabase.spectral_library.base import SpecLibBase
from alphabase.utils import parallel_imap

CUSTOM_MOD = "TestCustomMod@K"
CUSTOM_MOD_COMPOSITION = "H(4)O(2)"


@pytest.fixture
def restore_registry():
    """Undo the global registry changes each test makes."""
    snapshot = get_modification_state()
    yield
    set_modification_state(snapshot)


def _worker_registry(_):
    """Return the worker process's own view of the registry."""
    return get_modification_state()


def _add_custom_mod():
    add_new_modifications({CUSTOM_MOD: {"composition": CUSTOM_MOD_COMPOSITION}})


def _filter_modloss():
    # `load_mod_df` applies level 1 at import, keeping 2 modloss values; level 0
    # keeps 867, so a worker that missed this change is detectably different.
    keep_modloss_by_importance(0.0)


def _add_lower_case_aa():
    add_modifications_for_lower_case_AA()


def _load_custom_tsv():
    """Load a TSV that differs from the default one shipped with alphabase."""
    default_tsv = os.path.join(CONST_FILE_FOLDER, "modification.tsv")
    mod_df = pd.read_table(default_tsv, keep_default_na=False)
    extra = mod_df.iloc[[0]].copy()
    extra["mod_name"] = "TestTsvMod@K"
    extra["composition"] = CUSTOM_MOD_COMPOSITION
    custom_tsv = os.path.join(
        os.path.dirname(default_tsv), "_test_modification_tmp.tsv"
    )
    pd.concat([mod_df, extra], ignore_index=True).to_csv(
        custom_tsv, sep="\t", index=False
    )
    try:
        load_mod_df(custom_tsv)
    finally:
        os.remove(custom_tsv)


@pytest.mark.parametrize(
    "mutate",
    [_add_custom_mod, _filter_modloss, _add_lower_case_aa, _load_custom_tsv],
    ids=["custom_mods", "modloss_filtering", "lower_case_AA", "custom_tsv"],
)
def test_worker_registry_matches_parent(mutate, restore_registry):
    # Given a registry changed at runtime
    mutate()
    expected = get_modification_state()

    # When a worker process reports its own registry
    registries = list(
        parallel_imap(_worker_registry, [None, None], processes=2, progress=False)
    )

    # Then it is identical to the parent's
    for registry in registries:
        pd.testing.assert_frame_equal(registry, expected)


def _precursor_df(n_precursors=40, mod=CUSTOM_MOD):
    df = pd.DataFrame(
        {
            "sequence": ["PEPTIDEK"] * n_precursors,
            "mods": [mod] * n_precursors,
            "mod_sites": ["8" if mod else ""] * n_precursors,
            "charge": [2] * n_precursors,
        }
    )
    df["nAA"] = df["sequence"].str.len()
    return update_precursor_mz(df)


def test_isotope_intensity_mp_matches_single_process(restore_registry):
    # Given a library whose precursors carry a custom modification
    _add_custom_mod()
    isotope_cols = [f"i_{i}" for i in range(6)]

    # When the isotope intensities are calculated with and without multiprocessing
    single = calc_precursor_isotope_intensity(_precursor_df(), max_isotope=6)
    multi = calc_precursor_isotope_intensity_mp(
        _precursor_df(), max_isotope=6, mp_batch_size=10, mp_process_num=2
    )

    # Then the results agree
    pd.testing.assert_frame_equal(
        single.sort_index()[isotope_cols], multi.sort_index()[isotope_cols]
    )


def test_caller_supplied_progress_bar_is_used():
    # Given a caller supplying its own progress bar rather than the default
    seen_totals = []

    def progress(iterator, total):
        seen_totals.append(total)
        return iterator

    # When isotope intensities are calculated with multiprocessing
    calc_precursor_isotope_intensity_mp(
        _precursor_df(40, mod=""),
        max_isotope=6,
        mp_batch_size=10,
        mp_process_num=2,
        progress_bar=progress,
    )

    # Then it is actually driven, rather than being treated as a plain flag
    assert seen_totals == [4]


def test_speclib_isotope_info_runs_with_multiprocessing():
    # Given a library large enough to take the multiprocessing branch
    lib = SpecLibBase()
    lib._precursor_df = _precursor_df(20_000, mod="")

    # When isotope info is calculated
    lib.calc_precursor_isotope_info(mp_process_num=2, mp_batch_size=1000)

    # Then it completes; it used to raise TypeError on an unknown keyword
    assert "isotope_apex_offset" in lib.precursor_df.columns
