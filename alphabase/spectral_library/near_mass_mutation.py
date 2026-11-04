"""Near-mass mutation of peptides on zero-padded uint8 code arrays, avoiding a Python loop per sequence."""

from typing import NamedTuple

import numpy as np

from alphabase.constants.aa import AA_ASCII_MASS

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
# I/L look identical to fragments, N/D and Q/E mimic deamidation
EXCLUDED_MUTATIONS = ("IL", "LI", "ND", "DN", "QE", "EQ")
# C carries fixed labels, K defines the cleavage site
FROZEN_AAS = "CK"
# a proline before the C-terminal residue shapes fragmentation
PENULTIMATE_FROZEN_AAS = "P"
MIN_MASS_DELTA = 0.05
MAX_MASS_DELTA = 25.0

INNER_TABLE = 0
PENULTIMATE_TABLE = 1
# any uint8 code indexes the tables, so unknown residues have no candidates
_N_CODES = 256
# above any draw, so padding slots are never picked
_NO_CANDIDATE = 2.0

_HASH_BASE = np.uint64(0x100000001B3)
_SPLITMIX_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)
_FLOAT_MANTISSA_BITS = 53
_UINT64_MODULUS = 2**64


class ReplacementTables(NamedTuple):
    """Replacement candidates per residue code, stacked as [INNER_TABLE, PENULTIMATE_TABLE].

    candidates[t, code, :n_candidates[t, code]] are the replacement codes and
    cumulative[t, code] their cumulative draw probabilities, padded with _NO_CANDIDATE.
    """

    candidates: np.ndarray
    cumulative: np.ndarray
    n_candidates: np.ndarray


def encode(sequences: np.ndarray) -> np.ndarray:
    """ASCII codes of the sequences, one row each, padded with zeros."""
    fixed = np.asarray(sequences).astype(np.bytes_)
    return fixed.view(np.uint8).reshape(len(fixed), max(fixed.itemsize, 1))


def decode(codes: np.ndarray) -> np.ndarray:
    """Sequences of the zero-padded ASCII code rows."""
    codes = np.ascontiguousarray(codes)
    return codes.view(f"S{codes.shape[1]}").ravel().astype(str)


def inner_composition(codes: np.ndarray) -> np.ndarray:
    """Count of each residue code over all positions except the first and the last."""
    return np.bincount(codes[_inner_positions(codes)], minlength=_N_CODES)


def build_replacement_tables(
    composition: np.ndarray,
    min_mass_delta: float = MIN_MASS_DELTA,
    max_mass_delta: float = MAX_MASS_DELTA,
    frozen_aas: str = FROZEN_AAS,
    excluded_mutations: tuple = EXCLUDED_MUTATIONS,
) -> ReplacementTables:
    """Replacements within the mass window, weighted by their count in `composition`."""
    tables = [
        _build_replacement_table(
            composition, min_mass_delta, max_mass_delta, frozen, excluded_mutations
        )
        for frozen in (frozen_aas, frozen_aas + PENULTIMATE_FROZEN_AAS)
    ]
    return ReplacementTables(*(np.stack(arrays) for arrays in zip(*tables)))


def draw_uniforms(codes: np.ndarray, seed: int) -> np.ndarray:
    """Two uniform numbers in [0, 1) per sequence, derived from the seed and the sequence only.

    Padding zeros add nothing to the hash, so a sequence draws the same numbers in every batch.
    """
    powers = np.cumprod(np.full(codes.shape[1], _HASH_BASE, dtype=np.uint64))
    hashes = np.zeros(len(codes), dtype=np.uint64)
    for position in range(codes.shape[1]):
        hashes += codes[:, position].astype(np.uint64) * powers[position]
    seed_hash = _splitmix64(np.array([seed % _UINT64_MODULUS], dtype=np.uint64))
    mixed = hashes ^ seed_hash
    first = _splitmix64(mixed)
    second = _splitmix64(first)
    return np.stack([_to_unit_interval(first), _to_unit_interval(second)], axis=1)


def mutate(codes: np.ndarray, uniforms: np.ndarray, tables: ReplacementTables) -> tuple:
    """Replaces one inner residue per sequence.

    The site is drawn by uniforms[:, 0] and the replacement by uniforms[:, 1].
    Returns the mutated codes and whether each sequence had a replaceable residue; sequences
    without one are returned unchanged.
    """
    lengths = np.count_nonzero(codes, axis=1)
    table_at = (np.arange(codes.shape[1]) == (lengths - 2)[:, None]).astype(
        np.intp
    ) * PENULTIMATE_TABLE
    eligible = _inner_positions(codes) & (tables.n_candidates[table_at, codes] > 0)

    n_sites = eligible.sum(axis=1)
    has_site = n_sites > 0
    rows = np.flatnonzero(has_site)
    rank = (uniforms[rows, 0] * n_sites[rows]).astype(np.int64)
    sites = np.argmax(eligible[rows].cumsum(axis=1) > rank[:, None], axis=1)

    originals = codes[rows, sites]
    tables_used = table_at[rows, sites]
    passed = (tables.cumulative[tables_used, originals] <= uniforms[rows, 1, None]).sum(
        axis=1
    )
    index = np.minimum(passed, tables.n_candidates[tables_used, originals] - 1)

    mutated = codes.copy()
    mutated[rows, sites] = tables.candidates[tables_used, originals, index]
    return mutated, has_site


def _inner_positions(codes: np.ndarray) -> np.ndarray:
    lengths = np.count_nonzero(codes, axis=1)
    positions = np.arange(codes.shape[1])
    return (positions >= 1) & (positions < (lengths - 1)[:, None])


def _build_replacement_table(
    composition: np.ndarray,
    min_mass_delta: float,
    max_mass_delta: float,
    frozen_aas: str,
    excluded_mutations: tuple,
) -> tuple:
    aas = _codes_of(STANDARD_AAS)
    masses = AA_ASCII_MASS[aas]
    counts = composition[aas]
    frozen = np.isin(aas, _codes_of(frozen_aas))

    # original x replacement
    mass_deltas = np.abs(masses[None, :] - masses[:, None])
    allowed = (
        (min_mass_delta < mass_deltas)
        & (mass_deltas <= max_mass_delta)
        & (counts > 0)[None, :]
        & ~frozen[None, :]
        & ~frozen[:, None]
    )
    for original, replacement in excluded_mutations:
        allowed[STANDARD_AAS.index(original), STANDARD_AAS.index(replacement)] = False

    # allowed candidates first, so a draw indexes only the first n_candidates slots
    order = np.argsort(~allowed, axis=1, kind="stable")
    sorted_allowed = np.take_along_axis(allowed, order, axis=1)
    weights = np.where(sorted_allowed, counts[order], 0).astype(np.float64)
    totals = weights.sum(axis=1, keepdims=True)
    cumulative = np.where(
        sorted_allowed,
        np.cumsum(weights, axis=1) / np.maximum(totals, 1),
        _NO_CANDIDATE,
    )

    candidates = np.zeros((_N_CODES, len(aas)), dtype=np.uint8)
    full_cumulative = np.full((_N_CODES, len(aas)), _NO_CANDIDATE)
    n_candidates = np.zeros(_N_CODES, dtype=np.int64)
    candidates[aas] = aas[order]
    full_cumulative[aas] = cumulative
    n_candidates[aas] = sorted_allowed.sum(axis=1)
    return candidates, full_cumulative, n_candidates


def _codes_of(residues: str) -> np.ndarray:
    return np.frombuffer(residues.encode(), dtype=np.uint8)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    z = values + _SPLITMIX_INCREMENT
    z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_MULTIPLIER_1
    z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_MULTIPLIER_2
    return z ^ (z >> np.uint64(31))


def _to_unit_interval(values: np.ndarray) -> np.ndarray:
    return (values >> np.uint64(64 - _FLOAT_MANTISSA_BITS)).astype(np.float64) * (
        2.0**-_FLOAT_MANTISSA_BITS
    )
