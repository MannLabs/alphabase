import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from alphabase.scoring.feature_extraction_base import BaseFeatureExtractor
from alphabase.scoring.fdr import (
    calculate_fdr,
    calculate_fdr_from_ref,
    fdr_to_q_values,
    fdr_from_ref,
)

class Percolator:
    def __init__(self):
        self._feature_extractor:BaseFeatureExtractor = BaseFeatureExtractor()
        self._ml_model = LogisticRegression()
        
        self.fdr_level = 'psm' # psm, precursor, peptide, or sequence
        self.training_fdr = 0.01
        self.per_raw_fdr = False

        self.max_training_sample = 200000
        self.min_training_sample = 100
        self.cv_fold = 1
        self.iter_num = 1

        self._base_features = ['score','nAA','charge']

    @property
    def feature_list(self)->list:
        """ Get extracted feature_list. Property, read-only """
        return list(set(
            self._base_features+
            self.feature_extractor.feature_list
        ))

    @property
    def ml_model(self):
        """ 
        ML model in Percolator.
        It can be sklearn models or other models but implement 
        the methods `fit()` and `decision_function()` (or `predict_proba()`) 
        which are the same as sklearn models.
        """
        return self._ml_model
    
    @ml_model.setter
    def ml_model(self, model):
        self._ml_model = model

    @property
    def feature_extractor(self)->BaseFeatureExtractor:
        """
        The feature extractor inherited from `BaseFeatureExtractor`
        """
        return self._feature_extractor
    
    @feature_extractor.setter
    def feature_extractor(self, fe:BaseFeatureExtractor):
        self._feature_extractor = fe

    def extract_features(self,
        psm_df:pd.DataFrame,
        *args, **kwargs
    )->pd.DataFrame:
        """
        Extract features for rescoring.

        *args and **kwargs are used for 
        `self.feature_extractor.extract_features`.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM DataFrame

        Returns
        -------
        pd.DataFrame
            psm_df with feature columns appended inplace.
        """
        psm_df['ml_score'] = psm_df.score
        psm_df = self._estimate_psm_fdr(psm_df)
        return self._feature_extractor.extract_features(
            psm_df, *args, **kwargs
        )

    def rescore(self, 
        df:pd.DataFrame
    )->pd.DataFrame:
        """
        Estimate ML scores and then FDRs (q-values)

        Parameters
        ----------
        df : pd.DataFrame
            psm_df

        Returns
        -------
        pd.DataFrame
            psm_df with `ml_score` and `fdr` columns updated inplace
        """
        for i in range(self.iter_num):
            df = self._cv_score(df)
            df = self._estimate_fdr(df, 'psm', False)
            df = self.feature_extractor.update_features(df)
        df = self._estimate_fdr(df)
        return df

    def run_rerank_workflow(self,
        top_k_psm_df:pd.DataFrame,
        rerank_column:str='spec_idx',
        *args, **kwargs
    )->pd.DataFrame:
        """
        Run percolator workflow with reranking 
        the peptides for each spectrum.

        - self.extract_features()
        - self.rescore()

        *args and **kwargs are used for 
        `self.feature_extractor.extract_features`.

        Parameters
        ----------
        top_k_psm_df : pd.DataFrame
            PSM DataFrame

        rerank_column : str
            The column use to rerank PSMs. 
            
            For example, use the following code to select 
            the top-ranked peptide for each spectrum.
            ```
            rerank_column = 'spec_idx' # scan_num
            idx = top_k_psm_df.groupby(['raw_name',rerank_column])['ml_score'].idxmax()
            psm_df = top_k_psm_df.loc[idx].copy()
            ```
        Returns
        -------
        pd.DataFrame
            Only top-scored PSM is returned for 
            each group of the `rerank_column`.
        """
        top_k_psm_df = self.extract_features(
            top_k_psm_df, *args, **kwargs
        )
        idxmax = top_k_psm_df.groupby(
            ['raw_name',rerank_column]
        )['ml_score'].idxmax()

        df = top_k_psm_df.loc[idxmax].copy()
        self._train_and_score(df)

        top_k_psm_df = self._predict(top_k_psm_df)
        idxmax = top_k_psm_df.groupby(
            ['raw_name',rerank_column]
        )['ml_score'].idxmax()
        return top_k_psm_df.loc[idxmax].copy()

    def run_rescore_workflow(self,
        psm_df:pd.DataFrame,
        *args, **kwargs
    )->pd.DataFrame:
        """
        Run percolator workflow:

        - self.extract_features()
        - self.rescore()

        *args and **kwargs are used for 
        `self.feature_extractor.extract_features`.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM DataFrame

        Returns
        -------
        pd.DataFrame
            psm_df with feature columns appended inplace.
        """
        df = self.extract_features(
            psm_df, *args, **kwargs
        )
        return self.rescore(df)

    def _estimate_fdr_per_raw(self,
        df:pd.DataFrame,
        fdr_level:str
    )->pd.DataFrame:
        df_list = []
        for raw_name, df_raw in df.groupby('raw_name'):
            df_list.append(self._estimate_fdr(df_raw, 
                fdr_level = fdr_level,
                per_raw_fdr = False
            ))
        return pd.concat(df_list, ignore_index=True)

    def _estimate_psm_fdr(self,
        df:pd.DataFrame,
    )->pd.DataFrame:
        return calculate_fdr(df, 'ml_score', 'decoy')
        
    def _estimate_fdr(self, 
        df:pd.DataFrame,
        fdr_level:str=None,
        per_raw_fdr:bool=None,
    )->pd.DataFrame:
        if fdr_level is None: 
            fdr_level = self.fdr_level
        if per_raw_fdr is None: 
            per_raw_fdr = self.per_raw_fdr

        if per_raw_fdr:
            return self._estimate_fdr_per_raw(
                df, fdr_level=fdr_level
            )

        if fdr_level == 'psm':
            return self._estimate_psm_fdr(df)
        else:
            if fdr_level == 'precursor':
                _df = df.groupby([
                    'sequence','mods','mod_sites','charge','decoy'
                ])['ml_score'].max()
            elif fdr_level == 'peptide':
                _df = df.groupby([
                    'sequence','mods','mod_sites','decoy'
                ])['ml_score'].max()
            else:
                _df = df.groupby(['sequence','decoy'])['ml_score'].max()
            _df = self._estimate_psm_fdr(_df)
            df['fdr'] = fdr_from_ref(
                df['ml_score'].values, _df['ml_score'].values, 
                _df['fdr'].values
            )
        return df

    def _train(self, 
        train_t_df:pd.DataFrame, 
        train_d_df:pd.DataFrame
    ):
        train_t_df = train_t_df[train_t_df.fdr<=self.training_fdr]

        if len(train_t_df) > self.max_training_sample:
            train_t_df = train_t_df.sample(
                n=self.max_training_sample, 
                random_state=1337
            )
        if len(train_d_df) > self.max_training_sample:
            train_d_df = train_d_df.sample(
                n=self.max_training_sample,
                random_state=1337
            )

        train_df = pd.concat((train_t_df, train_d_df))
        train_label = np.ones(len(train_df),dtype=np.int8)
        train_label[len(train_t_df):] = 0

        self._ml_model.fit(
            train_df[self.feature_list].values, 
            train_label
        )

    def _predict(self, test_df):
        try:
            test_df['ml_score'] = self._ml_model.decision_function(
                test_df[self.feature_list].values
            )
        except AttributeError:
            test_df['ml_score'] = self._ml_model.predict_proba(
                test_df[self.feature_list].values
            )
        return test_df

    def _train_and_score(self,
        df:pd.DataFrame
    )->pd.DataFrame:

        df_target = df[df.decoy == 0]
        df_decoy = df[df.decoy != 0]

        if (
            np.sum(df_target.fdr<=self.training_fdr) < 
            self.min_training_sample or
            len(df_decoy) < self.min_training_sample
        ):
            return df
        
        self._train(df_target, df_decoy)
        test_df = pd.concat(
            [df_target, df_decoy],
            ignore_index=True
        )
    
        return self._predict(test_df)

    def _cv_score(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Apply cross-validation for rescoring.

        It will split `df` into K folds. For each fold, 
        its ML scores are predicted by a model which 
        is trained by other K-1 folds .

        Parameters
        ----------
        df : pd.DataFrame
            PSMs to be rescored

        Returns
        -------
        pd.DataFrame
            PSMs after rescoring
        """

        if self.cv_fold <= 1:
            return self._train_and_score(df)

        df = df.sample(
            frac=1, random_state=1337
        ).reset_index(drop=True)

        df_target = df[df.decoy == 0]
        df_decoy = df[df.decoy != 0]

        if (
            np.sum(df_target.fdr<=self.training_fdr) < 
            self.min_training_sample*self.cv_fold 
            or len(df_decoy) < 
            self.min_training_sample*self.cv_fold
        ):
            return df
        
        test_df_list = []
        for i in range(self.cv_fold):
            t_mask = np.ones(len(df_target), dtype=bool)
            _slice = slice(i, len(df_target), self.cv_fold)
            t_mask[_slice] = False
            train_t_df = df_target[t_mask]
            test_t_df = df_target[_slice]
            
            d_mask = np.ones(len(df_decoy), dtype=bool)
            _slice = slice(i, len(df_decoy), self.cv_fold)
            d_mask[_slice] = False
            train_d_df = df_decoy[d_mask]
            test_d_df = df_decoy[_slice]

            self._train(train_t_df, train_d_df)

            test_df = pd.concat((test_t_df, test_d_df))
            test_df_list.append(self._predict(test_df))
    
        return pd.concat(test_df_list, ignore_index=True)
    
