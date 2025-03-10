import multiprocessing as mp
from multiprocessing import Queue

import pandas as pd
import pytest

from alphabase.constants.modification import (
    _MOD_CLASSIFICATION_USER_ADDED,
    MOD_DF,
    MOD_Composition,
    add_new_modifications,
    get_custom_mods,
    has_custom_mods,
    update_all_by_MOD_DF,
)
from alphabase.peptide.precursor import _init_worker


def worker_has_custom_mods(custom_mods_dict, result_queue):
    """Worker function to test if custom mods are available in the worker process."""
    # Initialize the worker with custom mods
    _init_worker(custom_mods_dict)

    # Check if custom mods are available
    from alphabase.constants.modification import (
        MOD_DF,
        MOD_Composition,
    )

    # Put results in the queue
    result = {
        "has_custom_mods": has_custom_mods(),
        "mod_names": list(
            MOD_DF[MOD_DF["classification"] == _MOD_CLASSIFICATION_USER_ADDED].index
        ),
        "mod_composition_keys": list(MOD_Composition.keys()),
    }
    result_queue.put(result)


@pytest.fixture
def cleanup_test_mods():
    """Clean up test modifications after test."""
    # Keep track of test modifications
    test_mods = ["WorkerMod1@N-term", "WorkerMod2@S", "CalcMod@N-term"]

    yield

    # Remove test modifications
    for mod in test_mods:
        if mod in MOD_DF.index:
            MOD_DF.drop(mod, inplace=True)
        if mod in MOD_Composition:
            del MOD_Composition[mod]

    # Update dictionaries
    update_all_by_MOD_DF()


def test_worker_init(cleanup_test_mods):
    """Test that _init_worker correctly initializes custom mods in a worker process."""
    # Skip if only 1 CPU is available
    if mp.cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Add custom modifications in the main process
    add_new_modifications(
        [
            ("WorkerMod1@N-term", "C(2)H(3)O(1)", "H(2)O(1)"),
            ("WorkerMod2@S", "C(3)H(5)O(2)", ""),
        ]
    )

    # Make sure MOD_Composition is updated
    update_all_by_MOD_DF()

    # Verify mods were added in the main process
    assert "WorkerMod1@N-term" in MOD_DF.index
    assert "WorkerMod2@S" in MOD_DF.index
    assert "WorkerMod1@N-term" in MOD_Composition
    assert "WorkerMod2@S" in MOD_Composition

    # Get the custom mods
    custom_mods = get_custom_mods()

    # Create a queue for the worker to return results
    result_queue = Queue()

    # Start a worker process
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=worker_has_custom_mods, args=(custom_mods, result_queue))
    p.start()
    p.join()

    # Get the result from the worker
    result = result_queue.get()

    # Check that the worker has the custom mods
    assert "WorkerMod1@N-term" in result["mod_names"]
    assert "WorkerMod2@S" in result["mod_names"]
    assert "WorkerMod1@N-term" in result["mod_composition_keys"]
    assert "WorkerMod2@S" in result["mod_composition_keys"]


def test_worker_init_empty(cleanup_test_mods):
    """Test that _init_worker works with an empty custom mods dict."""
    # Skip if only 1 CPU is available
    if mp.cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Create a queue for the worker to return results
    result_queue = Queue()

    # Start a worker process with empty custom mods
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=worker_has_custom_mods, args=({}, result_queue))
    p.start()
    p.join()

    # Get the result from the worker
    result = result_queue.get()

    # Check that the worker doesn't have our test mods
    assert "WorkerMod1@N-term" not in result["mod_names"]
    assert "WorkerMod2@S" not in result["mod_names"]


def worker_calc_isotope(custom_mods_dict, df_dict, result_queue):
    """Worker function to test isotope calculation with custom mods."""
    # Initialize the worker with custom mods
    _init_worker(custom_mods_dict)

    # Verify custom mods were initialized
    from alphabase.constants.modification import MOD_Composition

    for mod_name in custom_mods_dict:
        assert mod_name in MOD_Composition

    # Convert dict back to dataframe
    df = pd.DataFrame(df_dict)

    # Calculate isotope intensities
    from alphabase.peptide.precursor import calc_precursor_isotope_intensity

    result_df = calc_precursor_isotope_intensity(df, max_isotope=3)

    # Convert result to dict for passing through the queue
    result_dict = {
        "i_0": result_df["i_0"].tolist(),
        "i_1": result_df["i_1"].tolist(),
        "i_2": result_df["i_2"].tolist(),
    }

    # Put results in the queue
    result_queue.put(result_dict)


def test_worker_calc_isotope(cleanup_test_mods):
    """Test that a worker can calculate isotope intensities with custom mods."""
    # Skip if only 1 CPU is available
    if mp.cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Add custom modifications in the main process
    add_new_modifications([("CalcMod@N-term", "C(2)H(3)O(1)", "H(2)O(1)")])

    # Make sure MOD_Composition is updated
    update_all_by_MOD_DF()

    # Verify mods were added in the main process
    assert "CalcMod@N-term" in MOD_DF.index
    assert "CalcMod@N-term" in MOD_Composition

    # Get the custom mods
    custom_mods = get_custom_mods()

    # Create a test dataframe
    df = pd.DataFrame(
        {
            "sequence": ["PEPTIDE"],
            "mods": ["CalcMod@N-term"],
            "mod_sites": ["0"],
            "charge": [2],
            "nAA": [7],
            "precursor_mz": [400.2],
        }
    )

    # Convert to dict for passing through the queue
    df_dict = {
        "sequence": df["sequence"].tolist(),
        "mods": df["mods"].tolist(),
        "mod_sites": df["mod_sites"].tolist(),
        "charge": df["charge"].tolist(),
        "nAA": df["nAA"].tolist(),
        "precursor_mz": df["precursor_mz"].tolist(),
    }

    # Create a queue for the worker to return results
    result_queue = Queue()

    # Start a worker process
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=worker_calc_isotope, args=(custom_mods, df_dict, result_queue)
    )
    p.start()
    p.join()

    # Get the result from the worker
    result_dict = result_queue.get()

    # Check that the worker calculated isotope intensities
    assert len(result_dict["i_0"]) == 1
    assert len(result_dict["i_1"]) == 1
    assert len(result_dict["i_2"]) == 1

    # Check that values are reasonable
    assert result_dict["i_0"][0] > 0
    assert result_dict["i_1"][0] > 0
    assert result_dict["i_2"][0] > 0

    # Sum should be close to 1
    total = result_dict["i_0"][0] + result_dict["i_1"][0] + result_dict["i_2"][0]
    assert 0.99 <= total <= 1.01
