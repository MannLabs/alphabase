import numpy as np
import pytest

from alphabase.protein.fasta import Digest

pytestmark = pytest.mark.requires_numba


class TestDigestGetCutPositions:
    """Test the get_cut_positions method of the Digest class."""

    @pytest.mark.parametrize(
        "protease, sequence, expected",
        [
            (
                "trypsin",
                "PEPTIDEK",
                np.array([0, 8], dtype=np.int64),
            ),
            (
                "trypsin",
                "MYPEPTIDER",
                np.array([0, 10], dtype=np.int64),
            ),
            (
                "trypsin",
                "PEPTIDEKPEPTIDER",
                np.array([0, 8, 16], dtype=np.int64),
            ),
            (
                "lys-c",
                "PEPTIDEK",
                np.array([0, 8], dtype=np.int64),
            ),
            (
                "lys-c",
                "PEPTIDEKPEPTIDER",
                np.array([0, 8, 16], dtype=np.int64),
            ),
            (
                "arg-c",
                "PEPTIDER",
                np.array([0, 8], dtype=np.int64),
            ),
            (
                "trypsin_not_p",
                "PEPTIDEKPEPTIDE",
                np.array([0, 15], dtype=np.int64),
            ),
            (
                "trypsin_not_p",
                "PEPTIDEKPPEPTIDE",
                np.array([0, 16], dtype=np.int64),
            ),
            (
                "trypsin",
                "PEPTIDE",
                np.array([0, 7], dtype=np.int64),
            ),
            (
                "trypsin",
                "KKRR",
                np.array([0, 1, 2, 3, 4], dtype=np.int64),
            ),
            (
                "trypsin",
                "K",
                np.array([0, 1], dtype=np.int64),
            ),
            (
                "glu-c",
                "PEPTIDE",
                np.array([0, 2, 7], dtype=np.int64),
            ),
            (
                "non-specific",
                "PEPTIDE",
                np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
            ),
        ],
    )
    def test_get_cut_positions_with_various_proteases(
        self, protease, sequence, expected
    ):
        """Test get_cut_positions returns correct cut positions for various proteases."""
        # given
        digest = Digest(protease=protease)

        # when
        result = digest.get_cut_positions(sequence)

        # then
        assert np.array_equal(result, expected)
        assert result.dtype == np.int64
        assert result[-1] == len(sequence)

    def test_get_cut_positions_with_empty_sequence(self):
        """Test get_cut_positions with an empty sequence."""
        # given
        digest = Digest(protease="trypsin")
        sequence = ""

        # when
        result = digest.get_cut_positions(sequence)

        # then
        assert np.array_equal(result, np.array([0], dtype=np.int64))

    def test_get_cut_positions_returns_sorted_positions(self):
        """Test that cut positions are returned in ascending order."""
        # given
        digest = Digest(protease="trypsin")
        sequence = "KRPEPTIDEKR"

        # when
        result = digest.get_cut_positions(sequence)

        # then
        assert np.all(result[:-1] <= result[1:])

    def test_get_cut_positions_with_custom_regex(self):
        """Test get_cut_positions with a custom regex pattern."""
        # given
        digest = Digest(protease="[ABC]")
        sequence = "XAYZBXC"

        # when
        result = digest.get_cut_positions(sequence)

        # then
        expected = np.array([0, 2, 5, 7], dtype=np.int64)
        assert np.array_equal(result, expected)
