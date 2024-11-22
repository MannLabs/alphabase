"""Factory class to convert PSM DataFrames to AnnData format."""

from typing import List, Union

import anndata as ad
import numpy as np
import pandas as pd

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
            aggfunc=np.nanmean,  # how to aggregate intensities for same protein in same raw file TODO first?
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
        cls, file_paths: Union[str, List[str]], reader_type: str = "maxquant", **kwargs
    ) -> "AnnDataFactory":
        """Create AnnDataFactory from PSM files.

        Parameters
        ----------
        file_paths : Union[str, List[str]]
            Path(s) to PSM file(s)
        reader_type : str, optional
            Type of PSM reader to use, by default "maxquant"
        **kwargs
            Additional arguments passed to PSM reader

        Returns
        -------
        AnnDataFactory
            Initialized AnnDataFactory instance

        """
        from alphabase.psm_reader.psm_reader import psm_reader_provider

        # TODO: add option to specify column names via API

        reader = psm_reader_provider.get_reader(reader_type, **kwargs)

        psm_df = reader.load(file_paths)
        return cls(psm_df)
