"""Factory class to convert PSM DataFrames to AnnData format."""

import warnings
from typing import List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd

from alphabase.psm_reader import PSMReaderBase  # noqa: TCH001
from alphabase.psm_reader.keys import PsmDfCols


class AnnDataFactory:
    """Factory class to convert AlphaBase PSM DataFrames to AnnData format."""

    def __init__(self, psm_df: pd.DataFrame):
        """Initialize AnnDataFactory.

        Parameters
        ----------
        psm_df : pd.DataFrame
            AlphaBase PSM DataFrame containing at minimum the columns:
            - PsmDfCols.RAW_NAME
            - PsmDfCols.PROTEINS
            - PsmDfCols.INTENSITY

        """
        required_cols = [PsmDfCols.RAW_NAME, PsmDfCols.PROTEINS, PsmDfCols.INTENSITY]
        missing_cols = [col for col in required_cols if col not in psm_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self._psm_df = psm_df

        duplicated_proteins = self._psm_df[PsmDfCols.PROTEINS].duplicated()
        if duplicated_proteins.sum() > 0:
            warnings.warn(
                f"Found {duplicated_proteins.sum()} duplicated protein groups. Using only first."
            )

    def create_anndata(self) -> ad.AnnData:
        """Create AnnData object from PSM DataFrame.

        Returns
        -------
        ad.AnnData
            AnnData object where:
            - obs (rows) are raw names
            - var (columns) are proteins
            - X contains intensity values

        """
        # Create pivot table: raw names x proteins with intensity values
        pivot_df = pd.pivot_table(
            self._psm_df,
            index=PsmDfCols.RAW_NAME,
            columns=PsmDfCols.PROTEINS,
            values=PsmDfCols.INTENSITY,
            aggfunc="first",  # DataFrameGroupBy.first -> will skip NA
            fill_value=np.nan,
            dropna=False,
        )

        return ad.AnnData(
            X=pivot_df.values,
            obs=pd.DataFrame(index=pivot_df.index),
            var=pd.DataFrame(index=pivot_df.columns),
        )

    @classmethod
    def from_files(
        cls,
        file_paths: Union[str, List[str]],
        reader_type: str = "maxquant",
        *,
        intensity_column: Optional[str] = None,
        protein_id_column: Optional[str] = None,
        raw_name_column: Optional[str] = None,
        **kwargs,
    ) -> "AnnDataFactory":
        """Create AnnDataFactory from PSM files.

        Parameters
        ----------
        file_paths : Union[str, List[str]]
            Path(s) to PSM file(s)
        reader_type : str, optional
            Type of PSM reader to use, by default "maxquant"
        intensity_column: str, optional
            Name of the column storing intensity data. Default is taken from `psm_reader.yaml`
        protein_id_column: str, optional
            Name of the column storing proteins ids. Default is taken from `psm_reader.yaml`
        raw_name_column: str, optional
            Name of the column storing raw (or run) name. Default is taken from `psm_reader.yaml`
        **kwargs
            Additional arguments passed to PSM reader

        Returns
        -------
        AnnDataFactory
            Initialized AnnDataFactory instance

        """
        from alphabase.psm_reader.psm_reader import psm_reader_provider

        reader: PSMReaderBase = psm_reader_provider.get_reader(reader_type, **kwargs)

        custom_column_mapping = {
            k: v
            for k, v in {
                PsmDfCols.INTENSITY: intensity_column if intensity_column else None,
                PsmDfCols.PROTEINS: protein_id_column if protein_id_column else None,
                PsmDfCols.RAW_NAME: raw_name_column if raw_name_column else None,
            }.items()
            if v is not None
        }

        if custom_column_mapping:
            reader.add_column_mapping(custom_column_mapping)

        psm_df = reader.load(file_paths)
        return cls(psm_df)
