"""Reader for spectral libraries exported by PEAKS Studio."""

from typing import Dict, List, Optional

import pandas as pd

from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import PSMReaderBase
from alphabase.spectral_library.flat import SpecLibFlat


class PEAKSLibraryReader(PSMReaderBase, SpecLibFlat):
    """Reader for spectral libraries exported by PEAKS Studio (DB search / DIA-DB export).

    Subclasses `PSMReaderBase` like every other reader in `alphabase.psm_reader`
    (rather than parsing the file standalone) so that this reader follows the
    same `import_file` -> `_pre_process` -> `_translate_columns` ->
    `_load_modifications` -> `_post_process` pipeline as the rest of the
    codebase - see `alphabase.spectral_library.reader.LibraryReaderBase` for
    the precedent of a reader that is *also* a `SpecLibBase`/`SpecLibFlat`.

    Slice 1 of this reader's implementation: precursor-level columns
    (sequence, charge, precursor_mz, rt) only, via the existing "peaks" entry
    in `psm_reader.yaml` - no override needed, the inherited
    `_translate_columns` already does this from `self.column_mapping`.
    Modifications and fragments are deliberately stubbed here (every
    precursor is treated as unmodified, `fragment_df` stays empty) - later
    slices replace the stub.

    Example:
    -------
    >>> reader = PEAKSLibraryReader()
    >>> reader.import_file("lib.tsv")
    >>> reader.precursor_df.head()

    """

    _reader_type = "peaks"
    _add_unimod_to_mod_mapping = True

    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        modification_mapping: Optional[Dict[str, List[str]]] = None,
        rt_unit: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the reader.

        Parameters
        ----------
        column_mapping : dict, optional
            PEAKS column name -> AlphaBase column name.
            Defaults to `psm_reader_yaml["peaks"]["column_mapping"]`.

        modification_mapping : dict, optional
            Additional/override PEAKS-name -> AlphaBase-name mappings, merged
            on top of the `"peaks"` entry in `psm_reader.yaml`'s
            `modification_mappings` section. See
            `alphabase.psm_reader.modification_mapper.ModificationMapper`.

        rt_unit : str, optional
            Unit of the retention time column, one of "minute", "second" or
            "irt". Defaults to `psm_reader_yaml["peaks"]["rt_unit"]`
            ("second" - the only convention observed across the available
            PEAKS exports so far).

        **kwargs : dict
            Passed through to `PSMReaderBase.__init__` (e.g. `fdr`).

        """
        # PEAKS fragments carry *observed* mz/intensity, so the dense
        # `charged_frag_types` machinery `SpecLibFlat` otherwise offers is
        # never used by this reader - no need to expose it as a parameter.
        SpecLibFlat.__init__(self)
        # `SpecLibFlat.__init__` doesn't set `_fragment_df` (only
        # `parse_base_library`/HDF loading do), so `fragment_df` would raise
        # AttributeError before the first `import_file()` call without this.
        self._fragment_df = pd.DataFrame()

        PSMReaderBase.__init__(
            self,
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            rt_unit=rt_unit,
            **kwargs,
        )

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:
        """Stub: treats every precursor as unmodified.

        Real parsing of PEAKS' "Modifications" column (Unimod mapping,
        position offset) is added in the next slice.
        """
        del origin_df  # unused for now
        self._psm_df[PsmDfCols.MODS] = ""
        self._psm_df[PsmDfCols.MOD_SITES] = ""
