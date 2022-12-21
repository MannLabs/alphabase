import pandas as pd

class BaseFeatureExtractor:
    def __init__(self):
        self._feature_list = ['score','nAA','charge']

    @property
    def feature_list(self)->list:
        """
        This is a property. It tells ML scoring modules 
        what features (columns) are extracted by 
        this FeatureExtractor for scoring.

        Returns
        -------
        list
            feature names (columns) in the PSM dataframe
        """

        self._feature_list = list(set(self._feature_list))
        return self._feature_list

    def extract_features(self, 
        psm_df:pd.DataFrame, 
        *args, **kwargs
    )->pd.DataFrame:
        """
        Extract the scoring features (self._feature_list) 
        and append them inplace into candidate PSMs (psm_df).

        **All sub-classes must re-implement this method.**

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSMs to be rescored

        Returns
        -------
        pd.DataFrame
            psm_df with appended feature columns extracted by this extractor
        """
        return psm_df

    def update_features(self,psm_df:pd.DataFrame)->pd.DataFrame:
        """
        This method allow us to update adaptive features
        during the iteration of Percolator algorithm

        Parameters
        ----------
        psm_df : pd.DataFrame
            psm_df
        
        Returns
        -------
        pd.DataFrame
            psm_df with updated feature values
        """
        return psm_df

