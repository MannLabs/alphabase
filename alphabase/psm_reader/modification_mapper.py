"""Module to handle modification mappings for different search engines."""

import copy
from collections import defaultdict
from typing import Dict, Optional

from alphabase.psm_reader.utils import MOD_TO_UNIMOD_DICT, get_extended_modifications


class ModificationMapper:
    """Class to handle modification mappings for different search engines."""

    def __init__(
        self,
        custom_modification_mapping: Optional[Dict[str, str]],
        *,
        reader_yaml: Dict,
        mapping_type: str,
        add_unimod_to_mod_mapping: bool,
    ):
        """Initialize the ModificationMapper.

        Parameters
        ----------
        custom_modification_mapping:
            A custom mapping or a string referencing one of the mappings in the reader_yaml
            The key of dict is a modification name in AlphaBase format;
            the value could be a str or a list, see below
            ```
            add_modification_mapping({
                'Dimethyl@K': ['K(Dimethyl)'], # list
                'Dimethyl@Any_N-term': '_(Dimethyl)', # str
            })

        reader_yaml:
            the yaml (read from file) containing the modification mappings

        mapping_type:
            the type of modification mapping ("maxquant" or "alphapept")

        add_unimod_to_mod_mapping:
            whether unimod modifications should be added to the mapping

        """
        self._psm_reader_yaml = reader_yaml
        self._add_unimod_to_mod_mapping = add_unimod_to_mod_mapping
        self._mapping_type = mapping_type

        self.modification_mapping = None
        self.rev_mod_mapping = None
        self.set_modification_mapping()
        self.add_modification_mapping(custom_modification_mapping)

    def add_modification_mapping(self, custom_modification_mapping: dict) -> None:
        """Append additional modification mappings for the search engine.

        Also creates a reverse mapping from the modification format used by the search engine to the AlphaBase format.

        Parameters
        ----------
        custom_modification_mapping : dict
            The key of dict is a modification name in AlphaBase format;
            the value could be a str or a list, see below
            ```
            add_modification_mapping({
            'Dimethyl@K': ['K(Dimethyl)'], # list
            'Dimethyl@Any_N-term': '_(Dimethyl)', # str
            })
            ```

        """
        if not isinstance(custom_modification_mapping, dict):
            return

        new_modification_mapping = defaultdict(list)
        for key, val in list(custom_modification_mapping.items()):
            if isinstance(val, str):
                new_modification_mapping[key].append(val)
            else:
                new_modification_mapping[key].extend(val)

        if new_modification_mapping:
            self.set_modification_mapping(
                self.modification_mapping | new_modification_mapping
            )

    def set_modification_mapping(
        self, modification_mapping: Optional[Dict] = None
    ) -> None:
        """Set the modification mapping for the search engine.

        Also creates a reverse mapping from the modification format used by the search engine to the AlphaBase format.

        Parameters
        ----------
        modification_mapping:
            If dictionary: the current modification_mapping will be overwritten by this.
            If str: the parameter will be interpreted as a modification_mapping_type, and the mapping is read from the
                respective key in the "modification_mappings" section of the psm_reader_yaml

        """
        if modification_mapping is None:
            self._init_modification_mapping()
        elif isinstance(
            modification_mapping,
            str,  # interpret as modification_mapping_type
        ):
            self.modification_mapping = self._psm_reader_yaml["modification_mappings"][
                modification_mapping
            ]

        else:
            self.modification_mapping = copy.deepcopy(modification_mapping)

        self._str_mods_to_lists()

        if self._add_unimod_to_mod_mapping:
            self._add_all_unimod()
            self._extend_mod_brackets()

        self.rev_mod_mapping = self._get_reversed_mod_mapping()

    def _init_modification_mapping(self) -> None:
        """Initialize the modification mapping from the psm_reader_yaml or as an empty dictionary."""
        self.modification_mapping = (
            self._psm_reader_yaml["modification_mappings"][self._mapping_type]
            if self._mapping_type is not None
            else {}
        )

    def _add_all_unimod(self) -> None:
        """Add all unimod modifications to the modification mapping."""
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
