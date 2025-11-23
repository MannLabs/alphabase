"""Unit tests for MSFragger PSM TSV reader."""

import pandas as pd
import pytest

from alphabase.psm_reader import psm_reader_provider
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.msfragger_reader import MSFragger_PSM_TSV_Reader


@pytest.fixture
def reader():
    """Fixture to create reader instance."""
    return psm_reader_provider.get_reader("msfragger_psm_tsv")


class TestReaderBasics:
    """Tests for reader initialization and basic functionality."""

    def test_reader_initialization(self, reader):
        """Test reader is properly initialized with correct configuration."""
        assert isinstance(reader, MSFragger_PSM_TSV_Reader)
        assert reader._reader_type == "msfragger_psm_tsv"
        assert len(reader._mass_mapped_mods) > 0
        assert reader._mod_mass_tol == 0.1
        assert isinstance(reader.column_mapping, dict)
        assert PsmDfCols.SEQUENCE in reader.column_mapping

    def test_custom_parameters(self):
        """Test reader accepts custom parameters."""
        reader = psm_reader_provider.get_reader("msfragger_psm_tsv", keep_decoy=True)
        assert reader._keep_decoy is True


class TestDataProcessing:
    """Tests for data processing and integration."""

    def test_preprocessing(self, reader):
        """Test preprocessing extracts raw names, spec_idx, and fills NAs correctly."""
        df = pd.DataFrame(
            {
                "Spectrum": ["file1.01234.01234.3", "file2.01235.01235.2"],
                "Peptide": ["PEPTIDE", None],
                "Is Decoy": ["false", "true"],
                "Assigned Modifications": ["5S(79.9663)", ""],
            }
        )

        processed = reader._pre_process(df.copy())
        assert PsmDfCols.RAW_NAME in processed.columns
        assert PsmDfCols.SPEC_IDX in processed.columns
        assert processed[PsmDfCols.RAW_NAME].tolist() == ["file1", "file2"]
        assert processed[PsmDfCols.SPEC_IDX].tolist() == [1234, 1235]
        assert processed["Peptide"].tolist() == ["PEPTIDE", ""]

    def test_decoy_translation(self, reader):
        """Test decoy translation converts string to int."""
        reader._psm_df = pd.DataFrame({PsmDfCols.DECOY: ["true", "false"]})
        reader._translate_decoy()
        assert reader._psm_df[PsmDfCols.DECOY].tolist() == [1, 0]

    def test_modification_loading_integration(self, reader):
        """Test modification loading produces complete correct results."""
        reader._psm_df = pd.DataFrame(
            {
                "Peptide": ["PEPTIDE", "SEQUENCE", "TEST"],
                "Assigned Modifications": [
                    "5S(79.9663)",
                    "",
                    "3M(15.9949), N-term(304.2071)",
                ],
            }
        )
        reader._load_modifications(reader._psm_df)

        assert PsmDfCols.MODS in reader._psm_df.columns
        assert PsmDfCols.MOD_SITES in reader._psm_df.columns
        assert len(reader._psm_df) == 3

        # Verify complete modification results for all rows
        assert reader._psm_df[PsmDfCols.MODS].tolist() == [
            "Phospho@S",
            "",
            "Oxidation@M;TMTpro@Any_N-term",
        ]
        assert reader._psm_df[PsmDfCols.MOD_SITES].tolist() == ["5", "", "3;0"]
