"""Module to handle modification mappings for different search engines."""

import copy
from collections import defaultdict
from typing import Dict, Optional, Tuple

from alphabase.psm_reader.utils import MOD_TO_UNIMOD_DICT, get_extended_modifications


class ModificationMapper:
    """Class to handle modification mappings for different search engines."""

    def __init__(
        self,
        reader_yaml: Dict,
        modification_type: Optional[str],
        *,
        add_unimod_to_mod_mapping: bool,
    ):
        """Initialize the ModificationMapper."""
        self.modification_mapping = None
        self.rev_mod_mapping = None

        self._psm_reader_yaml = reader_yaml
        self._add_unimod_to_mod_mapping = add_unimod_to_mod_mapping
        self._modification_type = modification_type

    def init_modification_mapping(
        self, modification_mapping: Optional[Dict[str, str]]
    ) -> Tuple:
        """Initialize the modification mapping (& reverse) for the search engine."""
        self.set_modification_mapping()
        self.add_modification_mapping(modification_mapping)

        return self.modification_mapping, self.rev_mod_mapping

    def add_modification_mapping(self, modification_mapping: dict) -> Tuple[Dict, Dict]:
        """Append additional modification mappings for the search engine.

        Also creates a reverse mapping from the modification format used by the search engine to the AlphaBase format.

        Parameters
        ----------
        modification_mapping : dict
            The key of dict is a modification name in AlphaBase format;
            the value could be a str or a list, see below
            ```
            add_modification_mapping({
            'Dimethyl@K': ['K(Dimethyl)'], # list
            'Dimethyl@Any_N-term': '_(Dimethyl)', # str
            })
            ```

        """
        if not isinstance(modification_mapping, dict):
            return self.modification_mapping, self.rev_mod_mapping

        new_modification_mapping = defaultdict(list)
        for key, val in list(modification_mapping.items()):
            if isinstance(val, str):
                new_modification_mapping[key].append(val)
            else:
                new_modification_mapping[key].extend(val)

        if new_modification_mapping:
            self.set_modification_mapping(
                self.modification_mapping | new_modification_mapping
            )

        return self.modification_mapping, self.rev_mod_mapping

    def set_modification_mapping(
        self, modification_mapping: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Set the modification mapping for the search engine.

        Also creates a reverse mapping from the modification format used by the search engine to the AlphaBase format.

        Parameters
        ----------
        modification_mapping:
            If dictionary: the current modification_mapping will be overwritten by this.
            If str: the parameter will be interpreted as a reader type, and the modification_mapping is read from the
                "modification_mapping" section of the psm_reader_yaml

        """
        if modification_mapping is None:
            self._init_modification_mapping()
        elif isinstance(modification_mapping, str):
            if modification_mapping in self._psm_reader_yaml:
                self.modification_mapping = copy.deepcopy(
                    self._psm_reader_yaml[modification_mapping]["modification_mapping"]
                )
            else:
                raise ValueError(
                    f"Unknown modification mapping: {modification_mapping}"
                )
        else:
            self.modification_mapping = copy.deepcopy(modification_mapping)

        self._str_mods_to_lists()

        if self._add_unimod_to_mod_mapping:
            self._add_all_unimod()
            self._extend_mod_brackets()

        self.rev_mod_mapping = self._get_reversed_mod_mapping()

        return self.modification_mapping, self.rev_mod_mapping

    def _init_modification_mapping(self) -> None:
        if self._modification_type is not None:
            self.modification_mapping = self._psm_reader_yaml[self._modification_type][
                "modification_mapping"
            ]
        else:
            self.modification_mapping = {}

    def _add_all_unimod(self) -> None:
        for mod_name, unimod in MOD_TO_UNIMOD_DICT.items():
            if mod_name in self.modification_mapping:
                self.modification_mapping[mod_name].append(unimod)
            else:
                self.modification_mapping[mod_name] = [unimod]

    def _extend_mod_brackets(self) -> None:
        """Update modification_mapping to include different bracket types."""
        for key, mod_list in list(self.modification_mapping.items()):
            self.modification_mapping[key] = get_extended_modifications(mod_list)

    def _str_mods_to_lists(self) -> None:
        """Convert all single strings to lists containing one item in self.modification_mapping."""
        for mod, val in list(self.modification_mapping.items()):
            if isinstance(val, str):
                self.modification_mapping[mod] = [val]

    def _get_reversed_mod_mapping(self) -> Dict[str, str]:
        """Create a reverse mapping from the modification format used by the search engine to the AlphaBase format."""
        rev_mod_mapping = {}
        for mod_alphabase_format, mod_other_format in self.modification_mapping.items():
            if isinstance(mod_other_format, (list, tuple)):
                for mod_other_format_ in mod_other_format:
                    if (
                        mod_other_format_ in rev_mod_mapping
                        and mod_alphabase_format.endswith("Protein_N-term")
                    ):
                        continue

                    rev_mod_mapping[mod_other_format_] = mod_alphabase_format
            else:
                rev_mod_mapping[mod_other_format] = mod_alphabase_format

        return rev_mod_mapping
