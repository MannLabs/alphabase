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
        """Test preprocessing extracts raw names, scan_num, and fills NAs correctly."""
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
        assert PsmDfCols.SCAN_NUM in processed.columns
        assert processed[PsmDfCols.RAW_NAME].tolist() == ["file1", "file2"]
        assert processed[PsmDfCols.SCAN_NUM].tolist() == [1234, 1235]
        assert processed["Peptide"].tolist() == ["PEPTIDE", ""]

    def test_decoy_translation(self, reader):
        """Test decoy translation converts string to int."""
        reader._psm_df = pd.DataFrame({PsmDfCols.DECOY: ["true", "false"]})
        reader._translate_decoy()
        assert reader._psm_df[PsmDfCols.DECOY].tolist() == [1, 0]

    def test_modification_loading_integration(self, reader):
        """Test modification loading produces complete correct results."""
        # Set up _psm_df with raw "Assigned Modifications" strings in PsmDfCols.ASSIGNED_MODS column
        # (this simulates what happens after _translate_columns() maps the column)
        reader._psm_df = pd.DataFrame(
            {
                "Peptide": ["PEPTIDE", "SEQUENCE", "TEST"],
                PsmDfCols.ASSIGNED_MODS: [
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

    def test_spec_idx_from_scan_num(self, reader):
        """Test that spec_idx is correctly calculated as scan_num - 1 in full workflow."""
        import io

        # Create a minimal MSFragger PSM TSV
        tsv_content = """Spectrum\tPeptide\tCharge\tRetention\tHyperscore\tProtein\tIs Decoy\tAssigned Modifications\tIntensity
file1.01234.01234.3\tPEPTIDEK\t2\t10.5\t50.0\tP12345\tfalse\t\t1000.0
file2.01235.01235.2\tSEQUENCER\t3\t15.2\t60.0\tP67890\tfalse\t\t2000.0
file3.00100.00100.2\tTESTK\t2\t5.0\t45.0\tP11111\tfalse\t\t500.0"""

        tsv_buffer = io.StringIO(tsv_content)
        result = reader.import_file(tsv_buffer)

        # Verify both scan_num and spec_idx exist
        assert PsmDfCols.SCAN_NUM in result.columns
        assert PsmDfCols.SPEC_IDX in result.columns

        # Verify spec_idx = scan_num - 1 for all rows
        assert len(result) == 3
        for _, row in result.iterrows():
            assert row[PsmDfCols.SPEC_IDX] == row[PsmDfCols.SCAN_NUM] - 1
