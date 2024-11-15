import logging
import multiprocessing as mp
import re
import typing
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from alphabase.constants.modification import MOD_DF
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import (
    PSMReaderBase,
    psm_reader_provider,
    psm_reader_yaml,
)


class SageModificationTranslation:
    def __init__(
        self,
        custom_translation_df: pd.DataFrame = None,
        ppm_tolerance: int = 10,
        mp_process_num=10,
    ):
        """Translate Sage style modifications to alphabase style modifications.
        A modified sequence like VM[+15.9949]QENSSSFSDLSER will be translated to mods: Oxidation@M, mod_sites: 2.
        By default, the translation is done by matching the observed mass and location to the UniMod database.
        If a custom translation dataframe is provided, the translation will be done based on the custom translation dataframe first.

        Parameters
        ----------
        custom_translation_df : pd.DataFrame
            A custom translation dataframe with columns 'modification' and 'matched_mod_name'.

        ppm_tolerance : int
            The ppm tolerance for matching the observed mass to the annotated modification mass.

        mp_process_num : int
            The number of processes to use for translation.

        """

        self.custom_translation_df = custom_translation_df
        self.ppm_tolerance = ppm_tolerance
        self.mp_process_num = mp_process_num

        # validate custom translation df
        if self.custom_translation_df is not None:
            valid = True
            valid &= "modification" in self.custom_translation_df.columns
            valid &= "matched_mod_name" in self.custom_translation_df.columns
            if not valid:
                raise ValueError(
                    "Custom translation df must have columns 'modification' and 'matched_mod_name'."
                )

    def __call__(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """Translate modifications in the PSMs to alphabase style modifications.
        1. Discover all modifications in the PSMs.
        2. Annotate modifications from custom translation df, if provided.
        3. Annotate all remaining modifications from UniMod.
        4. Apply translation to PSMs.
        5. Drop PSMs with missing modifications.

        Parameters
        ----------
        psm_df : pd.DataFrame
            The PSM dataframe with column 'modified_sequence'.

        Returns
        -------
        pd.DataFrame
            The PSM dataframe with columns 'mod_sites' and 'mods'.
        """

        # 1. Discover all modifications in the PSMs
        discovered_modifications_df = _discover_modifications(psm_df)
        translation_df = pd.DataFrame()

        # 2. Annotate modifications from custom translation df, if provided
        discovered_modifications_df, translation_df = (
            self._annotate_from_custom_translation(
                discovered_modifications_df, translation_df
            )
        )

        # 3. Annotate all remaining modifications from UniMod
        translation_df = self._annotate_from_unimod(
            discovered_modifications_df, translation_df
        )

        # 4. Apply translation to PSMs
        translated_psm_df = _apply_translate_modifications_mp(psm_df, translation_df)

        # 5. Drop PSMs with missing modifications
        is_null = translated_psm_df[PsmDfCols.MOD_SITES].isnull()
        translated_psm_df = translated_psm_df[~is_null]
        if np.sum(is_null) > 0:
            logging.warning(
                f"Dropped {np.sum(is_null)} PSMs with missing modifications."
            )

        return translated_psm_df

    def _annotate_from_custom_translation(
        self, discovered_modifications_df: pd.DataFrame, translation_df: pd.DataFrame
    ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """Annotate modifications from custom translation df, if provided.
        Discovered modifications are first matched using the custom translation dataframe.
        If no match is found, the modifications are returned for matching using UniMod.

        Parameters
        ----------

        discovered_modifications_df : pd.DataFrame
            The discovered modifications dataframe.

        translation_df : pd.DataFrame
            The translation dataframe.

        Returns
        -------

        typing.Tuple[pd.DataFrame, pd.DataFrame]
            The updated discovered modifications dataframe and translation dataframe.

        """
        if self.custom_translation_df is not None:
            discovered_modifications_df = discovered_modifications_df.merge(
                self.custom_translation_df, on="modification", how="left"
            )
            for _, row in discovered_modifications_df[
                discovered_modifications_df["matched_mod_name"].isnull()
            ].iterrows():
                logging.warning(
                    f"No modification found for mass {row['modification']} at position {row['previous_aa']} found in custom_translation_df, will be matched using UniMod"
                )

            translation_df = pd.concat(
                [
                    translation_df,
                    discovered_modifications_df[
                        discovered_modifications_df["matched_mod_name"].notnull()
                    ],
                ]
            )
            discovered_modifications_df = discovered_modifications_df[
                discovered_modifications_df["matched_mod_name"].isnull()
            ]

        return discovered_modifications_df, translation_df

    def _annotate_from_unimod(
        self, discovered_modifications_df: pd.DataFrame, translation_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Annotate all remaining modifications from UniMod.
        UniMod modification are used from the global MOD_DF.

        Parameters
        ----------
        discovered_modifications_df : pd.DataFrame
            The discovered modifications dataframe.

        translation_df : pd.DataFrame
            The translation dataframe.

        Returns
        -------
        pd.DataFrame
            The updated translation dataframe.
        """

        annotated_df = _get_annotated_mod_df()
        discovered_modifications_df["matched_mod_name"] = (
            discovered_modifications_df.apply(
                lambda x: _lookup_modification(
                    x["mass"],
                    x["previous_aa"],
                    annotated_df,
                    ppm_tolerance=self.ppm_tolerance,
                ),
                axis=1,
            )
        )
        for _, row in discovered_modifications_df[
            discovered_modifications_df["matched_mod_name"].isnull()
        ].iterrows():
            logging.warning(
                f"UniMod lookup failed for mass {row['modification']} at position {row['previous_aa']}, will be removed."
            )
        translation_df = pd.concat(
            [
                translation_df,
                discovered_modifications_df[
                    discovered_modifications_df["matched_mod_name"].notnull()
                ],
            ]
        )

        return translation_df


def _discover_modifications(psm_df: pd.DataFrame) -> pd.DataFrame:
    """Discover all modifications in the PSMs.

    Parameters
    ----------
    psm_df : pd.DataFrame
        The PSM dataframe with column 'modified_sequence'.

    Returns
    -------
    pd.DataFrame
        A dataframe with columns 'modification', 'previous_aa', 'is_nterm', 'is_cterm', 'mass'.

    """

    modifications = (
        psm_df[PsmDfCols.MODIFIED_SEQUENCE]
        .apply(_match_modified_sequence)
        .explode()
        .unique()
    )
    modifications = modifications[~pd.isnull(modifications)]
    return pd.DataFrame(
        list(modifications),
        columns=["modification", "previous_aa", "is_nterm", "is_cterm", "mass"],
    )


def _match_modified_sequence(
    sequence: str,
) -> typing.List[typing.Tuple[str, str, bool, bool, float]]:
    """Get all matches with the amino acid location.

    P[-100.0]EPTIDE -> [('[-100.0]', 'P', False, False, -100.0)]
    [-100.0]-PEPTIDE -> [('[-100.0]', '', True, False, -100.0)]
    PEPTIDE-[-100.0] -> [('[-100.0]', 'E', False, True, -100.0)]

    Parameters
    ----------

    sequence : str
        The sequence string.

    Returns
    -------
    typing.List[typing.Tuple[str, str, bool, bool, float]]
        A list of tuples with the matched modification.
        Each match has the structure (match, previous_aa, is_nterm, is_cterm, mass)

    """

    matches = []
    # Matches the square bracket modification pattern from modified sequences
    # [-100.0]-PEPTIDE -> [('[-100.0]', '', True, False, -100.0)]
    for m in re.finditer(r"\[(\+|-)(\d+\.\d+)\]", sequence):
        previous_char = sequence[m.start() - 1] if m.start() > 0 else ""
        next_char = sequence[m.end()] if m.end() < len(sequence) else ""

        is_nterm = next_char == "-"
        is_cterm = previous_char == "-"

        aa = previous_char if not (is_nterm | is_cterm) else ""

        mass = float(m.group().replace(r"[", "").replace(r"]", ""))

        matches += [(m.group(), aa, is_nterm, is_cterm, mass)]

    return matches


def _lookup_modification(
    mass_observed: float,
    previous_aa: str,
    mod_annotated_df: pd.DataFrame,
    ppm_tolerance: int = 10,
) -> str:
    """
    Look up a single modification based on the observed mass and the previous amino acid.

    Parameters
    ----------

    mass_observed : float
        The observed mass of the modification.

    previous_aa : str
        The previous amino acid.

    mod_annotated_df : pd.DataFrame
        The annotated modification dataframe.

    ppm_tolerance : int
        The ppm tolerance for matching the observed mass to the annotated modification mass.

    Returns
    -------

    str
        The name of the matched modification in alphabase format.

    """

    mass_distance = mod_annotated_df["mass"].values - mass_observed
    ppm_distance = mass_distance / mass_observed * 1e6
    ppm_distance = np.abs(ppm_distance)

    ppm_tolerance = min(np.min(ppm_distance), ppm_tolerance)

    # get index of matches
    mass_match = ppm_distance <= ppm_tolerance
    sequence_match = mod_annotated_df["previous_aa"] == previous_aa

    filtered_mod_df = mod_annotated_df[mass_match & sequence_match]

    if len(filtered_mod_df) == 0:
        logging.warning(
            f"No modification found for mass {mass_observed} at position {previous_aa}"
        )
        return None

    if len(filtered_mod_df) > 1:
        logging.warning(
            f"Multiple modifications found for mass {mass_observed} at position {previous_aa}, will use the one with the lowest localizer rank. Please use the custom translation df to resolve this."
        )

    matched_mod = filtered_mod_df.sort_values(by=["unimod_id", "localizer_rank"]).iloc[
        0
    ]
    return matched_mod.name


def _translate_modifications(
    sequence: str, mod_translation_df: pd.DataFrame
) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
    """Translate modifications in the sequence to alphabase style modifications.

    Parameters
    ----------
    sequence : str
        The sequence string.

    mod_translation_df : pd.DataFrame
        The annotated modification dataframe.

    Returns
    -------
    typing.Tuple[str, str]
        A tuple with the translated modification sites and names.

    """

    accumulated_non_sequence_chars = 0

    mod_sites = []
    mod_names = []

    for m in re.finditer(r"\[(\+|-)(\d+\.\d+)\]", sequence):
        group = m.group()

        previous_char = sequence[m.start() - 1] if m.start() > 0 else ""
        next_char = sequence[m.end()] if m.end() < len(sequence) else ""

        is_nterm = next_char == "-"
        is_cterm = previous_char == "-"

        if is_nterm:
            # real nterm mod with location 0
            matched_mod = mod_translation_df[
                (mod_translation_df["modification"] == group)
                & (mod_translation_df["is_nterm"])
            ]
            if len(matched_mod) == 0:
                return None, None

            matched_mod_name = matched_mod.iloc[0]["matched_mod_name"]
            mod_site = "0"
            mod_tag_len = m.end() - m.start() + 1

        elif is_cterm:
            # real cterm mod with location -1 or side chain mod at the last aa
            matched_mod = mod_translation_df[
                (mod_translation_df["modification"] == group)
                & (mod_translation_df["is_cterm"])
            ]
            if len(matched_mod) == 0:
                return None, None

            matched_mod_name = matched_mod.iloc[0]["matched_mod_name"]
            mod_site = (
                "-1"
                if matched_mod.iloc[0]["is_cterm"]
                else str(m.start() - accumulated_non_sequence_chars)
            )
            mod_tag_len = m.end() - m.start() + 1

        else:
            # side chain mod
            matched_mod = mod_translation_df[
                (mod_translation_df["modification"] == group)
                & (mod_translation_df["previous_aa"] == previous_char)
            ]
            if len(matched_mod) == 0:
                return None, None

            matched_mod_name = matched_mod.iloc[0]["matched_mod_name"]
            mod_site = str(m.start() - accumulated_non_sequence_chars)
            mod_tag_len = m.end() - m.start()

        accumulated_non_sequence_chars += mod_tag_len
        mod_sites.append(mod_site)
        mod_names.append(matched_mod_name)

    return ";".join(mod_sites), ";".join(mod_names)


def _apply_translate_modifications(
    psm_df: pd.DataFrame, mod_translation_df: pd.DataFrame
) -> pd.DataFrame:
    """Apply the translation of modifications to the PSMs.

    Parameters
    ----------

    psm_df : pd.DataFrame
        The PSM dataframe with column 'modified_sequence'.

    mod_translation_df : pd.DataFrame
        The annotated modification dataframe.

    Returns
    -------

    pd.DataFrame
        The PSM dataframe with columns 'mod_sites' and 'mods'.

    """

    psm_df[PsmDfCols.MOD_SITES], psm_df[PsmDfCols.MODS] = zip(
        *psm_df[PsmDfCols.MODIFIED_SEQUENCE].apply(
            lambda x: _translate_modifications(x, mod_translation_df)
        )
    )
    return psm_df


def _batchify_df(df: pd.DataFrame, mp_batch_size: int) -> typing.Generator:
    """Internal funciton for applying translation modifications in parallel.

    Parameters
    ----------
    df : pd.DataFrame
        The PSM dataframe.

    mp_batch_size : int
        The batch size for parallel processing.

    Returns
    -------

    typing.Generator
        A generator for the batchified dataframe.
    """

    for i in range(0, len(df), mp_batch_size):
        yield df.iloc[i : i + mp_batch_size, :]


def _apply_translate_modifications_mp(
    psm_df: pd.DataFrame,
    mod_translation_df: pd.DataFrame,
    mp_batch_size: int = 50000,
    mp_process_num: int = 10,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """Apply translate modifications with multiprocessing

    Parameters
    ----------

    psm_df : pd.DataFrame
        The PSM dataframe.

    mod_translation_df : pd.DataFrame
        Dataframe which instructs how to map modifications.

    mp_batch_size : int
        The batch size for parallel processing.

    mp_process_num : int
        The number of parallel processes
    """

    df_list = []
    with mp.get_context("spawn").Pool(mp_process_num) as p:
        processing = p.imap(
            partial(
                _apply_translate_modifications, mod_translation_df=mod_translation_df
            ),
            _batchify_df(psm_df, mp_batch_size),
        )
        if progress_bar:
            df_list = list(
                tqdm(processing, total=int(np.ceil(len(psm_df) / mp_batch_size)))
            )
        else:
            df_list = list(processing)

    return pd.concat(df_list, ignore_index=True)


def _get_annotated_mod_df() -> pd.DataFrame:
    """Annotates the modification dataframe for annotation of sage output.
    Due to the modified sequence based notation,
    C-Terminal and sidechain modifications on the last AA could be confused.

    Returns
    -------

    pd.DataFrame
        The annotated modification dataframe with columns 'mass', 'previous_aa', 'is_nterm', 'is_cterm', 'unimod_id', 'localizer_rank'.

    """
    mod_annotated_df = MOD_DF.copy()

    mod_annotated_df["previous_aa"] = (
        mod_annotated_df["mod_name"].str.split("@").str[1].str.split("^").str[0]
    )

    # we use the length of the localizer "K", "Any_N-term", "Protein_N-term" as rank to prioritize Any N-term over Protein N-term
    mod_annotated_df["localizer_rank"] = mod_annotated_df["previous_aa"].str.len()
    mod_annotated_df.loc[mod_annotated_df["localizer_rank"] > 1, "previous_aa"] = ""

    mod_annotated_df["is_nterm"] = mod_annotated_df["mod_name"].str.contains("N-term")
    mod_annotated_df["is_cterm"] = mod_annotated_df["mod_name"].str.contains("C-term")

    return mod_annotated_df[
        ["mass", "previous_aa", "is_nterm", "is_cterm", "unimod_id", "localizer_rank"]
    ]


def _sage_spec_idx_from_scan_nr(scan_indicator_str: str) -> int:
    """Extract the spectrum index from the scan_nr field in Sage output.
    Sage uses 1-based indexing for spectra, so we need to subtract 1 to convert to 0-based indexing.

    Parameters
    ----------

    scan_indicator_str : str
        The scan_indicator_str field in Sage output.
        e.g. `'controllerType=0 controllerNumber=1 scan=7846'`

    Returns
    -------

    int
        The 0-based spectrum index.

    Examples
    --------

    >>> _sage_spec_idx_from_scan_nr('controllerType=0 controllerNumber=1 scan=7846')
    7845

    """
    return int(re.search(r"scan=(\d+)", scan_indicator_str).group(1)) - 1


class SageReaderBase(PSMReaderBase):
    def __init__(
        self,
        *,
        column_mapping: dict = None,
        modification_mapping: dict = None,
        fdr=0.01,
        keep_decoy=False,
        rt_unit="second",
        custom_translation_df=None,
        mp_process_num=10,
        **kwargs,
    ):
        self.custom_translation_df = custom_translation_df
        self.mp_process_num = mp_process_num

        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            rt_unit=rt_unit,
            **kwargs,
        )

    def _init_column_mapping(self):
        self.column_mapping = psm_reader_yaml["sage"]["column_mapping"]

    def _load_file(self, filename):
        raise NotImplementedError

    def _transform_table(self, origin_df):
        self._psm_df[PsmDfCols.SPEC_IDX] = self._psm_df[PsmDfCols.SCANNR].apply(
            _sage_spec_idx_from_scan_nr
        )
        self._psm_df.drop(columns=[PsmDfCols.SCANNR], inplace=True)

    def _translate_decoy(self, origin_df):
        if not self._keep_decoy:
            self._psm_df = self._psm_df[~self._psm_df[PsmDfCols.DECOY]]

        self._psm_df = self._psm_df[self._psm_df[PsmDfCols.FDR] <= self._keep_fdr]
        self._psm_df = self._psm_df[
            self._psm_df[PsmDfCols.PEPTIDE_FDR] <= self._keep_fdr
        ]
        self._psm_df = self._psm_df[
            self._psm_df[PsmDfCols.PROTEIN_FDR] <= self._keep_fdr
        ]

        # drop peptide_fdr, protein_fdr
        self._psm_df.drop(
            columns=[PsmDfCols.PEPTIDE_FDR, PsmDfCols.PROTEIN_FDR], inplace=True
        )

    def _load_modifications(self, origin_df):
        pass

    def _translate_modifications(self):
        sage_translation = SageModificationTranslation(
            custom_translation_df=self.custom_translation_df,
            mp_process_num=self.mp_process_num,
        )
        self._psm_df = sage_translation(self._psm_df)

        # drop modified_sequence
        self._psm_df.drop(columns=[PsmDfCols.MODIFIED_SEQUENCE], inplace=True)


class SageReaderTSV(SageReaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_file(self, filename):
        return pd.read_csv(filename, sep="\t")


class SageReaderParquet(SageReaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_file(self, filename):
        return pd.read_parquet(filename)


def register_readers():
    psm_reader_provider.register_reader("sage_tsv", SageReaderTSV)
    psm_reader_provider.register_reader("sage_parquet", SageReaderParquet)
