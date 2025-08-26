from typing import Union

import pytest

from alphabase.pg_reader import AlphaPeptPGReader


class TestAlphapeptPGReader:
    """Test suite for AlphapeptPGReader._parse_alphapept_index method."""

    @pytest.fixture
    def reader(self):
        """Create a mock AlphapeptPGReader instance."""
        return AlphaPeptPGReader()

    @pytest.mark.parametrize(
        "identifier,expected",
        [
            # Test case 1: Standard UniProt Swiss-Prot format
            (
                "sp|Q9NQT4|EXOS5_HUMAN",
                {
                    "source_db": "sp",
                    "uniprot_ids": "Q9NQT4",
                    "ensembl_ids": "na",
                    "proteins": "EXOS5_HUMAN",
                    "is_decoy": False,
                },
            ),
            # Test case 2: UniProt ID only
            (
                "Q0IIK2",
                {
                    "source_db": "na",
                    "uniprot_ids": "Q0IIK2",
                    "ensembl_ids": "na",
                    "proteins": "na",
                    "is_decoy": False,
                },
            ),
            # Test case 3: Multiple UniProt entries
            (
                "sp|Q9H2K8|TAOK3_HUMAN,sp|Q7L7X3|TAOK1_HUMAN",
                {
                    "source_db": "sp;sp",
                    "uniprot_ids": "Q9H2K8;Q7L7X3",
                    "ensembl_ids": "na;na",
                    "proteins": "TAOK3_HUMAN;TAOK1_HUMAN",
                    "is_decoy": False,
                },
            ),
            # Test case 4: Ensembl format
            (
                "ENSEMBL:ENSBTAP00000024146",
                {
                    "source_db": "ENSEMBL",
                    "uniprot_ids": "na",
                    "ensembl_ids": "ENSBTAP00000024146",
                    "proteins": "na",
                    "is_decoy": False,
                },
            ),
            # Test case 5: Mixed Ensembl and UniProt
            (
                "ENSEMBL:ENSBTAP00000024146,sp|P35520|CBS_HUMAN",
                {
                    "source_db": "ENSEMBL;sp",
                    "uniprot_ids": "na;P35520",
                    "ensembl_ids": "ENSBTAP00000024146;na",
                    "proteins": "na;CBS_HUMAN",
                    "is_decoy": False,
                },
            ),
            # Test case 6: Decoy protein with REV__ prefix
            (
                "REV__sp|Q13085|ACACA_HUMAN",
                {
                    "source_db": "REV__sp",
                    "uniprot_ids": "Q13085",
                    "ensembl_ids": "na",
                    "proteins": "ACACA_HUMAN",
                    "is_decoy": True,
                },
            ),
        ],
    )
    def test_parse_alphapept_index(
        self,
        reader: AlphaPeptPGReader,
        identifier: str,
        expected: dict[str, Union[str, bool]],
    ) -> None:
        """Test _parse_alphapept_index with various identifier formats."""

        result = reader._parse_alphapept_index(identifier)

        # Assert that the result matches the expected output
        assert result == expected

    def test_parse_alphapept_index_multiple_decoys(
        self, reader: AlphaPeptPGReader
    ) -> None:
        """Test _parse_alphapept_index with multiple decoy entries."""
        identifier = "REV__sp|Q13085|ACACA_HUMAN,REV__sp|P35520|CBS_HUMAN"
        expected = {
            "source_db": "REV__sp;REV__sp",
            "uniprot_ids": "Q13085;P35520",
            "ensembl_ids": "na;na",
            "proteins": "ACACA_HUMAN;CBS_HUMAN",
            "is_decoy": True,
        }
        result = reader._parse_alphapept_index(identifier)
        assert result == expected

    def test_parse_alphapept_index_mixed_decoy_regular(
        self, reader: AlphaPeptPGReader
    ) -> None:
        """Test _parse_alphapept_index with mixed decoy and regular entries."""
        identifier = "sp|Q9NQT4|EXOS5_HUMAN,REV__sp|Q13085|ACACA_HUMAN"
        # This tests whether is_decoy is True if ANY entry is a decoy
        expected = {
            "source_db": "sp;REV__sp",
            "uniprot_ids": "Q9NQT4;Q13085",
            "ensembl_ids": "na;na",
            "proteins": "EXOS5_HUMAN;ACACA_HUMAN",
            "is_decoy": True,  # Assuming True if any entry is decoy
        }
        result = reader._parse_alphapept_index(identifier)
        assert result == expected
