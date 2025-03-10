import multiprocessing as mp
from multiprocessing import Queue

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


def worker_can_access_custom_mods(custom_mods_dict, result_queue):
    """Worker function to test if custom mods can be accessed in calculations."""
    # Initialize the worker with custom mods
    _init_worker(custom_mods_dict)

    # Verify custom mods were initialized
    from alphabase.constants.modification import MOD_DF, MOD_Composition

    # Put results in the queue
    result = {
        "mod_names_available": [
            mod_name for mod_name in custom_mods_dict if mod_name in MOD_DF.index
        ],
        "mod_composition_available": [
            mod_name for mod_name in custom_mods_dict if mod_name in MOD_Composition
        ],
    }
    result_queue.put(result)


def test_worker_can_access_custom_mods(cleanup_test_mods):
    """Test that a worker can access custom mods in calculations."""
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

    # Create a queue for the worker to return results
    result_queue = Queue()

    # Start a worker process
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=worker_can_access_custom_mods, args=(custom_mods, result_queue)
    )
    p.start()
    p.join()

    # Get the result from the worker
    result = result_queue.get()

    # Check that the worker can access the custom mods
    assert "CalcMod@N-term" in result["mod_names_available"]
    assert "CalcMod@N-term" in result["mod_composition_available"]
